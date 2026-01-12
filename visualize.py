import os
import numpy as np
import matplotlib.pyplot as plt


def plot_single_trajectory(traj_path, save_path=None):

    data = np.load(traj_path)
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    # Plot 1: Distance headway
    axes[0].plot(data["t"], data["dx"], 'b-', linewidth=1.5, label="Distance headway")
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=1, label="Collision threshold")
    axes[0].set_ylabel("dx (m)", fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)
    axes[0].set_title("Trajectory Analysis", fontsize=12, fontweight='bold')
    
    # Plot 2: Velocities
    axes[1].plot(data["t"], data["v"], 'b-', linewidth=1.5, label="Ego velocity")
    axes[1].plot(data["t"], data["lead_v"], 'g--', linewidth=1.5, label="Lead velocity")
    axes[1].set_ylabel("Velocity (m/s)", fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)
    
    # Plot 3: Actions
    axes[2].plot(data["t"], data["rl_action"], 'orange', linewidth=1, alpha=0.7, label="RL action")
    axes[2].plot(data["t"], data["applied_action"], 'b-', linewidth=1.5, label="Applied action (after safety filter)")
    axes[2].axhline(y=0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    axes[2].set_ylabel("Acceleration (m/s²)", fontsize=11)
    axes[2].set_xlabel("Time (s)", fontsize=11)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_comparison(results_dir="results", output_path="results/comparison.png"):

    # Load first episode from each condition
    conditions = ["baseline", "fgsm", "oia"]
    colors = {"baseline": "blue", "fgsm": "orange", "oia": "red"}
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    for condition in conditions:
        traj_path = os.path.join(results_dir, f"trajectory_{condition}_ep0.npz")
        
        if not os.path.exists(traj_path):
            print(f"Warning: {traj_path} not found")
            continue
        
        data = np.load(traj_path)
        color = colors[condition]
        label = condition.upper()
        
        # Plot distance headway
        axes[0].plot(data["t"], data["dx"], color=color, linewidth=1.5, 
                    label=label, alpha=0.8)
        
        # Plot ego velocity
        axes[1].plot(data["t"], data["v"], color=color, linewidth=1.5, 
                    label=label, alpha=0.8)
        
        # Plot applied action
        axes[2].plot(data["t"], data["applied_action"], color=color, 
                    linewidth=1.5, label=label, alpha=0.8)
    
    # Format plots
    axes[0].axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    axes[0].set_ylabel("Distance Headway (m)", fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10, loc='best')
    axes[0].set_title("Comparison: Baseline vs FGSM vs OIA", fontsize=14, fontweight='bold')
    
    axes[1].set_ylabel("Ego Velocity (m/s)", fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10, loc='best')
    
    axes[2].axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    axes[2].set_ylabel("Applied Acceleration (m/s²)", fontsize=12)
    axes[2].set_xlabel("Time (s)", fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=10, loc='best')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved comparison figure to {output_path}")
    plt.close()


def plot_summary_metrics(summary_path="results/summary.json", output_path="results/metrics.png"):

    import json
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    conditions = ["baseline", "fgsm", "oia"]
    collision_rates = [summary[c]["collision_rate"] for c in conditions]
    mean_returns = [summary[c]["mean_return"] for c in conditions]
    std_returns = [summary[c]["std_return"] for c in conditions]
    jerks = [summary[c]["mean_jerk"] for c in conditions]
    rmse_values = [summary[c].get("mean_rmse", 0.0) for c in conditions]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Collision rate
    axes[0, 0].bar(conditions, collision_rates, color=['blue', 'orange', 'red'], alpha=0.7)
    axes[0, 0].set_ylabel("Collision Rate", fontsize=12)
    axes[0, 0].set_title("Collision Rate by Condition", fontsize=12, fontweight='bold')
    axes[0, 0].set_ylim([0, max(collision_rates) * 1.2 if max(collision_rates) > 0 else 1.0])
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Mean return
    axes[0, 1].bar(conditions, mean_returns, yerr=std_returns, 
               color=['blue', 'orange', 'red'], alpha=0.7, capsize=5)
    axes[0, 1].set_ylabel("Episode Return", fontsize=12)
    axes[0, 1].set_title("Mean Episode Return", fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Jerk
    axes[1, 0].bar(conditions, jerks, color=['blue', 'orange', 'red'], alpha=0.7)
    axes[1, 0].set_ylabel("Mean Jerk (m/s³)", fontsize=12)
    axes[1, 0].set_title("Mean Jerk (Control Smoothness)", fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # RMSE (Stealth) - Lower is more stealthy
    axes[1, 1].bar(conditions, rmse_values, color=['blue', 'orange', 'red'], alpha=0.7)
    axes[1, 1].set_ylabel("Mean RMSE (Lower = Stealthier)", fontsize=12)
    axes[1, 1].set_title("Attack Stealth (RMSE)", fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Add values on top of bars for RMSE
    for i, v in enumerate(rmse_values):
        if v > 0:
            axes[1, 1].text(i, v + max(rmse_values) * 0.02, f'{v:.4f}', 
                           ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved metrics figure to {output_path}")
    plt.close()


def plot_stealth_comparison(summary_path="results/summary.json", output_path="results/stealth_comparison.png"):

    import json
    
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    
    # Extract RMSE values (excluding baseline which should be 0)
    fgsm_rmse = summary['fgsm'].get('mean_rmse', 0.0)
    oia_rmse = summary['oia'].get('mean_rmse', 0.0)
    fgsm_std = summary['fgsm'].get('std_rmse', 0.0)
    oia_std = summary['oia'].get('std_rmse', 0.0)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    attacks = ['FGSM', 'OIA']
    rmse_means = [fgsm_rmse, oia_rmse]
    rmse_stds = [fgsm_std, oia_std]
    colors = ['orange', 'red']
    
    bars = ax.bar(attacks, rmse_means, yerr=rmse_stds, 
                  color=colors, alpha=0.7, capsize=10, width=0.6)
    
    ax.set_ylabel("RMSE (Root Mean Square Error)", fontsize=14)
    ax.set_title("Attack Stealth Comparison\n(Lower RMSE = More Stealthy)", 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (v, std) in enumerate(zip(rmse_means, rmse_stds)):
        if v > 0:
            ax.text(i, v + std + max(rmse_means) * 0.02, 
                   f'{v:.4f}\n±{std:.4f}', 
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add interpretation text
    if oia_rmse < fgsm_rmse and oia_rmse > 0:
        winner = "OIA"
        ratio = fgsm_rmse / oia_rmse
        textstr = f'{winner} is {ratio:.2f}x more stealthy\n(Lower perturbation magnitude)'
    elif fgsm_rmse < oia_rmse and fgsm_rmse > 0:
        winner = "FGSM"
        ratio = oia_rmse / fgsm_rmse
        textstr = f'{winner} is {ratio:.2f}x more stealthy\n(Lower perturbation magnitude)'
    else:
        textstr = 'Similar stealth levels'
    
    ax.text(0.5, 0.95, textstr, transform=ax.transAxes,
           fontsize=11, verticalalignment='top', horizontalalignment='center',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved stealth comparison to {output_path}")
    plt.close()


if __name__ == "__main__":
    import sys
    
    results_dir = "results"
    
    # Generate all plots
    print("Generating visualization plots...")
    
    # Individual trajectories
    for condition in ["baseline", "fgsm", "oia"]:
        traj_path = os.path.join(results_dir, f"trajectory_{condition}_ep0.npz")
        if os.path.exists(traj_path):
            save_path = os.path.join(results_dir, f"plot_{condition}.png")
            plot_single_trajectory(traj_path, save_path)
    
    # Comparison plot
    plot_comparison(results_dir, os.path.join(results_dir, "comparison.png"))
    
    # Summary metrics (now includes RMSE)
    summary_path = os.path.join(results_dir, "summary.json")
    if os.path.exists(summary_path):
        plot_summary_metrics(summary_path, os.path.join(results_dir, "metrics.png"))
        plot_stealth_comparison(summary_path, os.path.join(results_dir, "stealth_comparison.png"))
    
    print("\nAll plots generated successfully!")