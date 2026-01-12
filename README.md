# PPO and Adversarial Attacks on ACC Environment

Clean implementation of the simplified 1D Adaptive Cruise Control (ACC) environment with PPO training and adversarial attacks (FGSM and OIA).

## Project Structure

```
.
├── acc_env.py          # 1D ACC environment with CBF safety filter
├── attacks.py          # FGSM and OIA attack implementations
├── train.py            # PPO training script
├── evaluate.py         # Evaluation script
├── visualize.py        # Trajectory and metrics visualization
├── analyze.py          # Results analysis
├── requirements.txt   
└── README.md          

# Generated during execution:
├── models/            # Trained models and normalization stats
│   ├── ppo_acc_final.zip
│   ├── vec_normalize.pkl
│   └── checkpoints...
├── logs/              # TensorBoard logs
└── results/           # Evaluation results and plots
    ├── summary.json
    ├── trajectory_*.npz
    ├── comparison.png
    └── metrics.png
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Train PPO Agent

Train the PPO agent on the ACC environment with observation normalization:

```bash
python train.py
```

This will:
- Create 8 parallel environments
- Train for 200,000 timesteps
- Save checkpoints every 50,000 steps
- Save final model to `models/ppo_acc_final.zip`
- Save normalization statistics to `models/vec_normalize.pkl`

Training takes approximately 5-10 minutes on a modern CPU.

### 2. Evaluate Agent

Evaluate the trained agent under baseline and attack conditions.

**Basic evaluation (normal driving):**
```bash
python evaluate.py
```

**Challenging scenario (lead vehicle braking):**
```bash
python evaluate.py --scenario challenging
```

**Gentle challenge:**
```bash
python evaluate.py --scenario gentle
```

**Custom epsilon:**
```bash
python evaluate.py --epsilon 0.015
```

**Multi-epsilon evaluation:**
```bash
python evaluate.py --multi-epsilon --scenario challenging
```

This will:
- Run 100 episodes for each condition (baseline, FGSM, OIA)
- Compute collision rates, episode returns, and jerk metrics
- Save results to `results/summary.json`
- Save sample trajectories to `results/trajectory_*.npz`

Evaluation takes approximately 2-3 minutes per scenario.

### 3. Visualize Results

Generate plots from evaluation results:

```bash
python visualize.py
```

This creates:
- `results/comparison.png` - Side-by-side trajectory comparison
- `results/metrics.png` - Bar charts of collision rate, return, and jerk
- `results/plot_baseline.png` - Individual baseline trajectory
- `results/plot_fgsm.png` - Individual FGSM trajectory
- `results/plot_oia.png` - Individual OIA trajectory

## Key Components

### ACC Environment (`acc_env.py`)

- **State**: `[dx, dv, v]` where dx is distance headway, dv is relative velocity, v is ego velocity
- **Action**: Scalar acceleration in `[-3.5, 2.0]` m/s²
- **Safety Filter**: CBF-based clamp ensuring `h(s) = dx - T_h * v >= 0`
- **Reward**: Penalizes speed error, unsafe distances, and aggressive actions

### Attacks (`attacks.py`)

**FGSM (Fast Gradient Sign Method)**
- Perturbs observation to maximize change in policy output
- `s' = s + ε · sign(∇_s μ_θ(s))`

**OIA (Optimism Induction Attack)**
- Perturbs observation to inflate value estimate
- `s' = s + ε · sign(∇_s V_φ(s))`
- Makes agent overly optimistic, delaying critical braking

### Training (`train.py`)

- Uses Stable-Baselines3 PPO implementation
- 8 parallel environments for faster training
- VecNormalize wrapper for observation normalization
- Hyperparameters follow assignment suggestions

### Evaluation (`evaluate.py`)

Runs three evaluation conditions:
1. **Baseline**: No attack
2. **FGSM**: Fast Gradient Sign Method with ε=0.01
3. **OIA**: Optimism Induction Attack with ε=0.01

Metrics computed:
- Collision rate (fraction of episodes with collision)
- Mean episode return (higher is better)
- Mean jerk (lower is smoother)

## Expected Results

Based on the assignment specifications:

1. **Baseline** should have very low collision rate (<5%) and good returns
2. **FGSM** should degrade performance moderately
3. **OIA** should show stronger degradation than FGSM:
   - Higher collision rate
   - Lower episode returns
   - More aggressive/delayed braking behavior

The key insight is that OIA is more effective because it makes the agent believe it's in a safer state, causing delayed reactions to critical situations.

## Customization

### Run Different Scenarios

```bash
# Normal driving (default - no lead braking)
python evaluate.py --scenario normal

# Challenging (lead brakes at -2.0 m/s²)
python evaluate.py --scenario challenging

# Gentle challenge (lead brakes at -1.5 m/s²)  
python evaluate.py --scenario gentle
```

### Modify Attack Strength

```bash
# Weaker attack
python evaluate.py --epsilon 0.005

# Stronger attack
python evaluate.py --epsilon 0.02

# Test multiple epsilons
python evaluate.py --multi-epsilon --epsilons 0.005 0.01 0.015 0.02
```

### Change Number of Episodes

```bash
python evaluate.py --n-episodes 200
```

### Specify Output Directory

```bash
python evaluate.py --output-dir my_results
```

### Analyze Existing Results

```bash
python analyze.py  # Analyzes results/summary.json
```

## Troubleshooting

**Issue**: Training is slow
- Reduce `n_envs` in `train.py` if memory is limited
- Training on GPU will be faster if PyTorch CUDA is available

**Issue**: Evaluation results show no difference between attacks
- Ensure epsilon is not too small (try 0.02)
- Verify VecNormalize stats are loaded correctly
- Check that safety filter is using adversarial observations

**Issue**: All episodes have collisions
- Reduce epsilon value
- Verify safety filter is functioning (check `applied_action` vs `rl_action`)
- Retrain with more timesteps
