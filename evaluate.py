import os
import json
import numpy as np
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from acc_env import ACCEnv
from attacks import create_attack


def evaluate_single_condition(vec_env, base_env, model, attack, n_episodes, 
                              scenario_config, verbose=True):
    """
    Evaluate agent for one condition (baseline or attack).
    """
    collisions = []
    returns = []
    min_dx_values = []
    episodes_data = []
    rmse_values = []  # Track RMSE for each episode
    
    for ep in range(n_episodes):
        obs = vec_env.reset()
        
        # Configure scenario
        if scenario_config.get('use_lead_decel', False):
            base_env.activate_lead_deceleration(
                start_time=scenario_config.get('decel_start', 5.0),
                duration=scenario_config.get('decel_duration', 3.0),
                decel=scenario_config.get('decel_value', -2.0)
            )
        
        episode_return = 0.0
        episode_collision = False
        min_dx = float('inf')
        episode_rmse_steps = []  # RMSE for each step in episode
        
        # Store trajectory for first 3 episodes
        if ep < 3:
            trajectory = {
                't': [], 'dx': [], 'dv': [], 'v': [],
                'rl_action': [], 'applied_action': [], 'lead_v': [],
                'step_rmse': []  # Add RMSE tracking
            }
        else:
            trajectory = None
        
        done = False
        step = 0
        
        while not done:
            # Store original observation for RMSE calculation
            obs_original = obs.copy()
            obs_for_policy = obs.copy()
            
            # Apply attack if present
            if attack is not None:
                obs_adv = attack.perturb(obs[0])
                base_env.set_safety_obs_for_filter(obs_adv)
                obs_for_policy = obs_adv.reshape(1, -1)
                
                # Compute step RMSE (between original and perturbed observations)
                step_rmse = np.sqrt(np.mean((obs_adv - obs[0]) ** 2))
                episode_rmse_steps.append(step_rmse)
            else:
                step_rmse = 0.0  # No perturbation for baseline
            
            # Get action from policy
            action, _ = model.predict(obs_for_policy, deterministic=True)
            
            # Step environment
            obs, reward, terminated, info = vec_env.step(action)
            
            episode_return += reward[0]
            min_dx = min(min_dx, info[0]["dx"])
            
            if info[0].get("collision", False):
                episode_collision = True
            
            # Record trajectory
            if trajectory is not None:
                trajectory['t'].append(step * ACCEnv.DT)
                trajectory['dx'].append(info[0]["dx"])
                trajectory['dv'].append(info[0]["lead_v"] - info[0]["ego_v"])
                trajectory['v'].append(info[0]["ego_v"])
                trajectory['rl_action'].append(info[0]["rl_action"])
                trajectory['applied_action'].append(info[0]["applied_action"])
                trajectory['lead_v'].append(info[0]["lead_v"])
                trajectory['step_rmse'].append(step_rmse)
            
            done = terminated[0] or info[0].get("TimeLimit.truncated", False)
            step += 1
        
        collisions.append(episode_collision)
        returns.append(episode_return)
        min_dx_values.append(min_dx)
        
        # Compute episode mean RMSE
        if episode_rmse_steps:
            episode_mean_rmse = np.mean(episode_rmse_steps)
        else:
            episode_mean_rmse = 0.0
        rmse_values.append(episode_mean_rmse)
        
        if trajectory is not None:
            episodes_data.append(trajectory)
        
        if verbose and (ep + 1) % 20 == 0:
            curr_coll = sum(collisions) / len(collisions)
            curr_ret = np.mean(returns)
            curr_rmse = np.mean(rmse_values)
            print(f"  Episode {ep+1}/{n_episodes}: "
                  f"Collision={curr_coll:.3f}, Return={curr_ret:.2f}, RMSE={curr_rmse:.4f}")
    
    # Compute jerk
    jerks = []
    for traj in episodes_data:
        actions = np.array(traj["applied_action"])
        if len(actions) > 1:
            jerk = np.mean(np.abs(np.diff(actions)))
            jerks.append(jerk)
    mean_jerk = float(np.mean(jerks)) if jerks else 0.0
    
    return {
        "collision_rate": float(sum(collisions) / n_episodes),
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_jerk": mean_jerk,
        "mean_rmse": float(np.mean(rmse_values)),  # Add mean RMSE
        "std_rmse": float(np.std(rmse_values)),    # Add std RMSE
        "min_dx_mean": float(np.mean(min_dx_values)),
        "min_dx_std": float(np.std(min_dx_values)),
        "episodes": episodes_data
    }


def run_evaluation(
    model_path="models/ppo_acc_final.zip",
    vec_normalize_path="models/vec_normalize.pkl",
    n_episodes=100,
    epsilon=0.01,
    scenario="normal",
    output_dir="results"
):
    """
    Run complete evaluation with baseline, FGSM, and OIA.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure scenario
    scenario_configs = {
        'normal': {
            'use_lead_decel': False,
            'description': 'Normal driving (no lead braking)'
        },
        'challenging': {
            'use_lead_decel': True,
            'decel_value': -2.0,
            'description': 'Challenging (lead brakes at -2.0 m/s²)'
        },
        'gentle': {
            'use_lead_decel': True,
            'decel_value': -1.5,
            'description': 'Gentle challenge (lead brakes at -1.5 m/s²)'
        }
    }
    
    scenario_config = scenario_configs.get(scenario, scenario_configs['normal'])
    
    print("=" * 70)
    print(f"EVALUATION: {scenario_config['description']}")
    print("=" * 70)
    print(f"Episodes per condition: {n_episodes}")
    print(f"Attack epsilon: {epsilon}")
    print()
    
    # Load model and environment
    print("Loading model and environment...")
    vec_env = DummyVecEnv([lambda: ACCEnv()])
    vec_env = VecNormalize.load(vec_normalize_path, vec_env)
    vec_env.training = False
    vec_env.norm_reward = False
    
    model = PPO.load(model_path, env=vec_env)
    base_env = vec_env.envs[0]
    
    # Run evaluations
    results = {}
    
    # Baseline
    print("\n" + "=" * 70)
    print("BASELINE")
    print("=" * 70)
    baseline_results = evaluate_single_condition(
        vec_env, base_env, model, None, n_episodes, scenario_config
    )
    results['baseline'] = baseline_results
    
    # FGSM
    print("\n" + "=" * 70)
    print(f"FGSM ATTACK (ε={epsilon})")
    print("=" * 70)
    fgsm_attack = create_attack("fgsm", model, epsilon)
    fgsm_results = evaluate_single_condition(
        vec_env, base_env, model, fgsm_attack, n_episodes, scenario_config
    )
    results['fgsm'] = fgsm_results
    
    # OIA
    print("\n" + "=" * 70)
    print(f"OIA ATTACK (ε={epsilon})")
    print("=" * 70)
    oia_attack = create_attack("oia", model, epsilon)
    oia_results = evaluate_single_condition(
        vec_env, base_env, model, oia_attack, n_episodes, scenario_config
    )
    results['oia'] = oia_results
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nBaseline:")
    print(f"  Collision Rate: {baseline_results['collision_rate']:.3f}")
    print(f"  Mean Return: {baseline_results['mean_return']:.2f} ± {baseline_results['std_return']:.2f}")
    print(f"  Mean Jerk: {baseline_results['mean_jerk']:.4f}")
    print(f"  Mean RMSE: {baseline_results['mean_rmse']:.4f} (no attack)")
    
    print(f"\nFGSM (ε={epsilon}):")
    print(f"  Collision Rate: {fgsm_results['collision_rate']:.3f}")
    print(f"  Mean Return: {fgsm_results['mean_return']:.2f} ± {fgsm_results['std_return']:.2f}")
    print(f"  Mean Jerk: {fgsm_results['mean_jerk']:.4f}")
    print(f"  Mean RMSE: {fgsm_results['mean_rmse']:.4f} (stealth)")
    
    print(f"\nOIA (ε={epsilon}):")
    print(f"  Collision Rate: {oia_results['collision_rate']:.3f}")
    print(f"  Mean Return: {oia_results['mean_return']:.2f} ± {oia_results['std_return']:.2f}")
    print(f"  Mean Jerk: {oia_results['mean_jerk']:.4f}")
    print(f"  Mean RMSE: {oia_results['mean_rmse']:.4f} (stealth)")
    
    # Compute degradation
    fgsm_deg = baseline_results['mean_return'] - fgsm_results['mean_return']
    oia_deg = baseline_results['mean_return'] - oia_results['mean_return']
    
    print(f"\nPerformance Degradation:")
    print(f"  FGSM: {fgsm_deg:.2f}")
    print(f"  OIA: {oia_deg:.2f}")
    if oia_deg > 0 and fgsm_deg > 0:
        ratio = oia_deg / fgsm_deg
        print(f"  OIA is {ratio:.2f}x more damaging than FGSM")
    
    if oia_results['collision_rate'] > fgsm_results['collision_rate']:
        coll_ratio = oia_results['collision_rate'] / max(fgsm_results['collision_rate'], 0.01)
        print(f"  OIA causes {coll_ratio:.2f}x more collisions than FGSM")
    
    # Stealth comparison
    print(f"\nStealth (RMSE - Lower is Stealthier):")
    print(f"  FGSM: {fgsm_results['mean_rmse']:.4f}")
    print(f"  OIA: {oia_results['mean_rmse']:.4f}")
    if fgsm_results['mean_rmse'] > oia_results['mean_rmse']:
        stealth_ratio = fgsm_results['mean_rmse'] / max(oia_results['mean_rmse'], 1e-6)
        print(f"  OIA is {stealth_ratio:.2f}x more stealthy than FGSM")
    elif oia_results['mean_rmse'] > fgsm_results['mean_rmse']:
        stealth_ratio = oia_results['mean_rmse'] / max(fgsm_results['mean_rmse'], 1e-6)
        print(f"  FGSM is {stealth_ratio:.2f}x more stealthy than OIA")
    else:
        print(f"  Similar stealth levels")
    
    # Save results
    summary = {
        'scenario': scenario,
        'epsilon': float(epsilon),
        'n_episodes': n_episodes,
        'baseline': {
            'collision_rate': baseline_results['collision_rate'],
            'mean_return': baseline_results['mean_return'],
            'std_return': baseline_results['std_return'],
            'mean_jerk': baseline_results['mean_jerk'],
            'mean_rmse': baseline_results['mean_rmse'],
            'std_rmse': baseline_results['std_rmse'],
        },
        'fgsm': {
            'collision_rate': fgsm_results['collision_rate'],
            'mean_return': fgsm_results['mean_return'],
            'std_return': fgsm_results['std_return'],
            'mean_jerk': fgsm_results['mean_jerk'],
            'mean_rmse': fgsm_results['mean_rmse'],
            'std_rmse': fgsm_results['std_rmse'],
            'epsilon': float(epsilon),
        },
        'oia': {
            'collision_rate': oia_results['collision_rate'],
            'mean_return': oia_results['mean_return'],
            'std_return': oia_results['std_return'],
            'mean_jerk': oia_results['mean_jerk'],
            'mean_rmse': oia_results['mean_rmse'],
            'std_rmse': oia_results['std_rmse'],
            'epsilon': float(epsilon),
        }
    }
    
    summary_path = os.path.join(output_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to {summary_path}")
    
    # Save sample trajectories
    for condition_name in ['baseline', 'fgsm', 'oia']:
        condition_data = results[condition_name]
        for i, ep_data in enumerate(condition_data['episodes']):
            traj_path = os.path.join(output_dir, f'trajectory_{condition_name}_ep{i}.npz')
            np.savez(traj_path, **ep_data)
    
    print(f"Sample trajectories saved to {output_dir}")
    print("=" * 70)
    
    return summary


def run_multi_epsilon_evaluation(
    model_path="models/ppo_acc_final.zip",
    vec_normalize_path="models/vec_normalize.pkl",
    n_episodes=100,
    epsilons=[0.005, 0.01, 0.015, 0.02],
    scenario="challenging",
    output_dir="results_multi_epsilon"
):
    """
    Run evaluation across multiple epsilon values.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    
    for eps in epsilons:
        print(f"\n{'=' * 70}")
        print(f"TESTING EPSILON = {eps}")
        print('=' * 70)
        
        result = run_evaluation(
            model_path=model_path,
            vec_normalize_path=vec_normalize_path,
            n_episodes=n_episodes,
            epsilon=eps,
            scenario=scenario,
            output_dir=os.path.join(output_dir, f"eps_{eps}")
        )
        
        all_results[f"eps_{eps}"] = result
    
    # Save combined results
    combined_path = os.path.join(output_dir, 'multi_epsilon_summary.json')
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'=' * 70}")
    print("MULTI-EPSILON SUMMARY")
    print('=' * 70)
    
    for eps_key, result in all_results.items():
        eps_val = float(eps_key.split('_')[1])
        print(f"\nε = {eps_val}:")
        print(f"  Baseline:  Coll={result['baseline']['collision_rate']:.1%}, "
              f"Return={result['baseline']['mean_return']:.1f}")
        print(f"  FGSM:      Coll={result['fgsm']['collision_rate']:.1%}, "
              f"Return={result['fgsm']['mean_return']:.1f}")
        print(f"  OIA:       Coll={result['oia']['collision_rate']:.1%}, "
              f"Return={result['oia']['mean_return']:.1f}")
    
    print(f"\nCombined results saved to {combined_path}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate PPO agent with adversarial attacks')
    parser.add_argument('--model', type=str, default='models/ppo_acc_final.zip',
                       help='Path to trained model')
    parser.add_argument('--vec-normalize', type=str, default='models/vec_normalize.pkl',
                       help='Path to VecNormalize stats')
    parser.add_argument('--n-episodes', type=int, default=100,
                       help='Number of episodes per condition')
    parser.add_argument('--epsilon', type=float, default=0.01,
                       help='Attack perturbation budget')
    parser.add_argument('--scenario', type=str, default='normal',
                       choices=['normal', 'challenging', 'gentle'],
                       help='Evaluation scenario type')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--multi-epsilon', action='store_true',
                       help='Run evaluation with multiple epsilon values')
    parser.add_argument('--epsilons', type=float, nargs='+',
                       default=[0.005, 0.01, 0.015, 0.02],
                       help='Epsilon values for multi-epsilon mode')
    
    args = parser.parse_args()
    
    if args.multi_epsilon:
        run_multi_epsilon_evaluation(
            model_path=args.model,
            vec_normalize_path=args.vec_normalize,
            n_episodes=args.n_episodes,
            epsilons=args.epsilons,
            scenario=args.scenario,
            output_dir=args.output_dir
        )
    else:
        run_evaluation(
            model_path=args.model,
            vec_normalize_path=args.vec_normalize,
            n_episodes=args.n_episodes,
            epsilon=args.epsilon,
            scenario=args.scenario,
            output_dir=args.output_dir
        )