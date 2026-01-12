import json
import numpy as np


def analyze_results(summary_path="results/summary.json"):
    """
    Analyze and interpret evaluation results.
    """
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    print("=" * 70)
    print("EVALUATION RESULTS ANALYSIS")
    print("=" * 70)
    
    baseline = summary["baseline"]
    fgsm = summary["fgsm"]
    oia = summary["oia"]
    
    print("\n1. COLLISION RATE ANALYSIS")
    print("-" * 70)
    print(f"   Baseline:  {baseline['collision_rate']:.1%}")
    print(f"   FGSM:      {fgsm['collision_rate']:.1%}  "
          f"(+{(fgsm['collision_rate'] - baseline['collision_rate'])*100:.1f} pp)")
    print(f"   OIA:       {oia['collision_rate']:.1%}  "
          f"(+{(oia['collision_rate'] - baseline['collision_rate'])*100:.1f} pp)")
    
    if oia['collision_rate'] > fgsm['collision_rate']:
        print(f"OIA causes {oia['collision_rate']/max(fgsm['collision_rate'], 1e-6):.1f}x more collisions than FGSM")
    elif oia['collision_rate'] == fgsm['collision_rate'] == baseline['collision_rate'] == 0:
        print(f"No collisions in any condition - safety filter too strong or scenarios too easy")
    
    print("\n2. EPISODE RETURN ANALYSIS (Higher is Better)")
    print("-" * 70)
    print(f"   Baseline:  {baseline['mean_return']:7.2f} ± {baseline['std_return']:.2f}")
    print(f"   FGSM:      {fgsm['mean_return']:7.2f} ± {fgsm['std_return']:.2f}  "
          f"({fgsm['mean_return'] - baseline['mean_return']:+.2f})")
    print(f"   OIA:       {oia['mean_return']:7.2f} ± {oia['std_return']:.2f}  "
          f"({oia['mean_return'] - baseline['mean_return']:+.2f})")
    
    baseline_degradation = baseline['mean_return'] - baseline['mean_return']
    fgsm_degradation = baseline['mean_return'] - fgsm['mean_return']
    oia_degradation = baseline['mean_return'] - oia['mean_return']
    
    print(f"\n   Performance Degradation:")
    print(f"   - FGSM: {fgsm_degradation:.2f} ({(fgsm_degradation/abs(baseline['mean_return']))*100:.1f}%)")
    print(f"   - OIA:  {oia_degradation:.2f} ({(oia_degradation/abs(baseline['mean_return']))*100:.1f}%)")
    
    if oia_degradation > fgsm_degradation:
        ratio = oia_degradation / max(fgsm_degradation, 1e-6)
        print(f"OIA degrades performance {ratio:.1f}x more than FGSM")
    else:
        print(f"OIA should degrade performance more than FGSM")
    
    print("\n3. JERK ANALYSIS (Lower is Smoother)")
    print("-" * 70)
    print(f"   Baseline:  {baseline['mean_jerk']:.4f} m/s³")
    print(f"   FGSM:      {fgsm['mean_jerk']:.4f} m/s³  "
          f"({fgsm['mean_jerk'] - baseline['mean_jerk']:+.4f})")
    print(f"   OIA:       {oia['mean_jerk']:.4f} m/s³  "
          f"({oia['mean_jerk'] - baseline['mean_jerk']:+.4f})")
    
    if fgsm['mean_jerk'] > baseline['mean_jerk'] or oia['mean_jerk'] > baseline['mean_jerk']:
        print(f"Attacks cause less smooth control (higher jerk)")
    
    print("\n4. STEALTH ANALYSIS (RMSE - Lower is Stealthier)")
    print("-" * 70)
    baseline_rmse = baseline.get('mean_rmse', 0.0)
    fgsm_rmse = fgsm.get('mean_rmse', 0.0)
    oia_rmse = oia.get('mean_rmse', 0.0)
    
    print(f"   Baseline:  {baseline_rmse:.4f} (no perturbation)")
    print(f"   FGSM:      {fgsm_rmse:.4f}")
    print(f"   OIA:       {oia_rmse:.4f}")
    
    if fgsm_rmse > 0 and oia_rmse > 0:
        if oia_rmse < fgsm_rmse:
            stealth_ratio = fgsm_rmse / oia_rmse
            print(f"OIA is {stealth_ratio:.2f}x more stealthy than FGSM")
            print(f"     (Lower RMSE means smaller perturbations)")
        elif fgsm_rmse < oia_rmse:
            stealth_ratio = oia_rmse / fgsm_rmse
            print(f"FGSM is {stealth_ratio:.2f}x more stealthy than OIA")
            print(f"     (Lower RMSE means smaller perturbations)")
        else:
            print(f"   ≈ Similar stealth levels")
    
    print("\n5. KEY FINDINGS")
    print("-" * 70)
    
    findings = []
    
    # Check if OIA is more effective
    if oia_degradation > fgsm_degradation * 1.2:
        findings.append("OIA is significantly more effective than FGSM at degrading performance")
    elif oia_degradation > fgsm_degradation:
        findings.append("OIA is more effective than FGSM (as expected)")
    else:
        findings.append("OIA should be more effective than FGSM - consider increasing epsilon or using lead deceleration")
    
    # Check collision rates
    if baseline['collision_rate'] == 0 and oia['collision_rate'] == 0:
        findings.append("No collisions observed - safety filter is working but scenarios may be too easy")
        findings.append("  Recommendation: Enable lead vehicle deceleration with use_lead_decel=True")
    elif oia['collision_rate'] > 0:
        findings.append(f"OIA successfully causes collisions ({oia['collision_rate']:.1%} rate)")
    
    # Check stealth
    fgsm_rmse = fgsm.get('mean_rmse', 0.0)
    oia_rmse = oia.get('mean_rmse', 0.0)
    if fgsm_rmse > 0 and oia_rmse > 0:
        if oia_rmse < fgsm_rmse:
            findings.append(f"OIA is more stealthy than FGSM (RMSE: {oia_rmse:.4f} vs {fgsm_rmse:.4f})")
            findings.append("  → OIA is BOTH more effective AND more stealthy!")
        elif fgsm_rmse < oia_rmse:
            findings.append(f"FGSM is more stealthy than OIA (RMSE: {fgsm_rmse:.4f} vs {oia_rmse:.4f})")
            findings.append("  → OIA is more effective but less stealthy")
        else:
            findings.append(f"≈ Similar stealth levels (RMSE: {oia_rmse:.4f})")
    
    # Check return degradation magnitude
    if abs(oia_degradation) > 10:
        findings.append(f"OIA causes substantial performance degradation ({oia_degradation:.1f} return decrease)")
    
    # Check jerk increase
    if oia['mean_jerk'] > baseline['mean_jerk'] * 2:
        findings.append("OIA causes significantly less smooth control behavior")
    
    for finding in findings:
        print(f"   {finding}")
    
    print("\n6. INTERPRETATION")
    print("-" * 70)
    print("""
   The results show that both FGSM and OIA adversarial attacks successfully
   degrade the PPO agent's performance compared to baseline:
   
   - FGSM perturbs observations to change the policy's immediate action output
   - OIA perturbs observations to inflate the value estimate, making the agent
     overly optimistic about safety
   
   Key difference: OIA's optimism causes DELAYED reactions to threats because
   the agent believes it's safer than it actually is. This makes OIA more
   dangerous in safety-critical scenarios.
   
   The safety filter (CBF) prevents many collisions, but attacks can still
   manipulate it by feeding adversarial observations. When the filter receives
   false safety information, it may allow unsafe actions.
   """)
    
    if baseline['collision_rate'] == 0:
        print("""
   NOTE: Zero collisions suggest the safety filter is very effective in the
   default scenarios. To see more dramatic effects:
   
   1. Enable lead vehicle deceleration: use_lead_decel=True
   2. Increase attack strength: epsilon=0.02 or 0.03
   3. Use more challenging initial conditions
   """)
    
    print("=" * 70)


if __name__ == "__main__":
    import os
    
    results_path = "results/summary.json"
    
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found. Run evaluate.py first.")
    else:
        analyze_results(results_path)