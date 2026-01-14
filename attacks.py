import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class AttackBase:

        self.model = model
        self.epsilon = epsilon
        self.device = next(model.policy.parameters()).device
    

    def __init__(self, model, epsilon=0.01): 
        self.model = model
        self.epsilon = epsilon
        self.device = next(model.policy.parameters()).device
    
    def perturb(self, obs):

        raise NotImplementedError


class FGSM(AttackBase):

    def perturb(self, obs):

        obs_np = np.array(obs, dtype=np.float32)
        is_batched = len(obs_np.shape) == 2
        
        if not is_batched:
            obs_np = obs_np.reshape(1, -1)
        
        # Convert to torch tensor
        obs_tensor = torch.tensor(obs_np, dtype=torch.float32, device=self.device)
        obs_tensor.requires_grad = True
        
        # Forward pass through policy to get mean action
        with torch.enable_grad():
            # Get policy features and mean action
            features = self.model.policy.extract_features(obs_tensor)
            latent_pi = self.model.policy.mlp_extractor.forward_actor(features)
            mean_actions = self.model.policy.action_net(latent_pi)

            loss = mean_actions.sum()
            loss.backward()
        
        # Get gradient and compute perturbation
        grad = obs_tensor.grad.cpu().numpy()
        perturbation = self.epsilon * np.sign(grad)
        
        # Apply perturbation and clip to valid range [-1, 1]
        obs_adv = obs_np + perturbation
        obs_adv = np.clip(obs_adv, -1.0, 1.0)
        
        # Return in original shape
        if not is_batched:
            obs_adv = obs_adv.squeeze(0)
        
        return obs_adv


class OIA(AttackBase):
    
    def perturb(self, obs):
        # Handle both single obs and batched obs
        obs_np = np.array(obs, dtype=np.float32)
        is_batched = len(obs_np.shape) == 2
        
        if not is_batched:
            obs_np = obs_np.reshape(1, -1)
        
        # Convert to torch tensor
        obs_tensor = torch.tensor(obs_np, dtype=torch.float32, device=self.device)
        obs_tensor.requires_grad = True
        
        # Forward pass through value network
        with torch.enable_grad():
            # Get value estimate
            features = self.model.policy.extract_features(obs_tensor)
            latent_vf = self.model.policy.mlp_extractor.forward_critic(features)
            value = self.model.policy.value_net(latent_vf)
            
            # Compute gradient of value w.r.t. observation
            value.sum().backward()
        
        # Get gradient and compute perturbation
        grad = obs_tensor.grad.cpu().numpy()
        perturbation = self.epsilon * np.sign(grad)
        
        # Apply perturbation and clip to valid range [-1, 1]
        obs_adv = obs_np + perturbation
        obs_adv = np.clip(obs_adv, -1.0, 1.0)
        
        # Return in original shape
        if not is_batched:
            obs_adv = obs_adv.squeeze(0)
        
        return obs_adv


def create_attack(attack_type, model, epsilon=0.01):

    if attack_type.lower() == 'fgsm':
        return FGSM(model, epsilon)
    elif attack_type.lower() == 'oia':
        return OIA(model, epsilon)
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")


class AttackWrapper(gym.Wrapper):

    def __init__(self, env, attack, apply_to_safety_filter=True):

        super().__init__(env)
        self.attack = attack
        self.apply_to_safety_filter = apply_to_safety_filter
        
        # Track original and perturbed observations for RMSE computation
        self.original_obs = None
        self.perturbed_obs = None
        self.step_rmse_values = []
        self.episode_rmse = 0.0
    
    def reset(self, **kwargs):

        obs, info = self.env.reset(**kwargs)
        
        # Store original observation
        self.original_obs = obs.copy()
        
        # Perturb observation
        self.perturbed_obs = self.attack.perturb(obs)
        
        # Compute step RMSE
        step_rmse = np.sqrt(np.mean((self.perturbed_obs - self.original_obs) ** 2))
        self.step_rmse_values = [step_rmse]
        self.episode_rmse = step_rmse
        
        # Apply to safety filter if enabled
        if self.apply_to_safety_filter and hasattr(self.env, 'set_safety_obs_for_filter'):
            self.env.set_safety_obs_for_filter(self.perturbed_obs)
        
        return self.perturbed_obs, info
    
    def step(self, action):

        obs, reward, terminated, truncated, info = self.env.step(action)

        self.original_obs = obs.copy()

        self.perturbed_obs = self.attack.perturb(obs)
        
        # Compute step RMSE
        step_rmse = np.sqrt(np.mean((self.perturbed_obs - self.original_obs) ** 2))
        self.step_rmse_values.append(step_rmse)
        
        # Apply to safety filter if enabled
        if self.apply_to_safety_filter and hasattr(self.env, 'set_safety_obs_for_filter'):
            self.env.set_safety_obs_for_filter(self.perturbed_obs)
        
        # Add RMSE to info
        info['step_rmse'] = step_rmse
        info['episode_rmse_mean'] = np.mean(self.step_rmse_values)
        
        return self.perturbed_obs, reward, terminated, truncated, info
    
    def get_episode_rmse(self):
        
        return np.mean(self.step_rmse_values) if self.step_rmse_values else 0.0


class FGSMAttackWrapper(AttackWrapper):

    def __init__(self, env, model, epsilon=0.01):
        attack = FGSM(model, epsilon)
        super().__init__(env, attack, apply_to_safety_filter=True)


class OIAAttackWrapper(AttackWrapper):

    def __init__(self, env, model, epsilon=0.01):
        attack = OIA(model, epsilon)
        super().__init__(env, attack, apply_to_safety_filter=True)