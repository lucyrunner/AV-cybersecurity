import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from acc_env import ACCEnv


def make_env():
    """Create a single ACC environment instance"""
    return ACCEnv()


def train_ppo(
    total_timesteps=200000,
    n_envs=8,
    save_dir="models",
    save_freq=50000
):
    """
    Train PPO agent with vectorized environments and observation normalization.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    print("Creating vectorized environments...")
    # Create vectorized environment
    vec_env = DummyVecEnv([make_env for _ in range(n_envs)])

    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        gamma=0.99
    )
    
    print("Creating PPO model...")
    # PPO hyperparameters (from assignment suggestion)
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=128,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log="./logs"
    )
    
    # Setup checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq // n_envs,  # adjust for parallel envs
        save_path=save_dir,
        name_prefix="ppo_acc"
    )
    
    print(f"Training for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # Save final model and normalization stats
    model.save(os.path.join(save_dir, "ppo_acc_final"))
    vec_env.save(os.path.join(save_dir, "vec_normalize.pkl"))
    
    print(f"\nTraining complete. Model saved to {save_dir}")
    print(f"Observation mean: {vec_env.obs_rms.mean}")
    print(f"Observation var: {vec_env.obs_rms.var}")
    
    return model, vec_env


if __name__ == "__main__":
    # Train with default settings
    model, vec_env = train_ppo(
        total_timesteps=200000,
        n_envs=8,
        save_dir="models",
        save_freq=50000
    )
    
    print("\nTraining finished successfully!")