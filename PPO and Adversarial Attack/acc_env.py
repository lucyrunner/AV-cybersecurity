#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Cell 1: imports & constants
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import json
import os
from typing import Optional, Dict, Any


# Environment physical / control constants
DT = 0.1 # time step (s)
TH = 1.5 # desired time headway (s)
D0 = 5.0 # standstill distance (m)
V_REF = 15.0 # target lead/desired speed (m/s)
A_MIN = -3.5 # braking limit (m/s^2)
A_MAX = 2.0 # acceleration limit (m/s^2)


# Observation scaling ranges (used when normalize_obs=False; if using VecNormalize, wrapper will handle scaling)
OBS_LOW = np.array([0.0, -30.0, 0.0], dtype=np.float32) # [dx, dv, v]
OBS_HIGH = np.array([200.0, 30.0, 40.0], dtype=np.float32)


# Useful for deterministic reproducibility
DEFAULT_SEED = 2025


# In[5]:


# ===== ACCEnv (self-contained with _apply_safety) =====
from typing import Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ---- Constants ----
DT = 0.1
TH = 1.5
D0 = 5.0
V_REF = 15.0
A_MIN = -3.5
A_MAX =  2.0
OBS_LOW  = np.array([0.0,  -20.0,  0.0], dtype=np.float32)
OBS_HIGH = np.array([200.0,  20.0, 30.0], dtype=np.float32)
DEFAULT_SEED = 0

class ACCEnv(gym.Env):
    """
    1D Adaptive Cruise Control (ACC) with Control Barrier Function (CBF) safety filter.

    State: [dx, dv, v]
    Action: scalar acceleration a ∈ [A_MIN, A_MAX]

    Attack-aware hook:
      env.set_safety_obs_for_filter(obs_adv)
    makes the safety filter use attacked obs for one step.
    """

    metadata = {"render_modes": ["human"], "render_fps": int(1 / DT)}

    def __init__(self, normalize_obs: bool = False, seed: Optional[int] = None):
        super().__init__()
        self.normalize_obs = normalize_obs
        self.seed_val = seed if seed is not None else DEFAULT_SEED
        self.np_random, _ = gym.utils.seeding.np_random(self.seed_val)

        self.observation_space = spaces.Box(low=OBS_LOW, high=OBS_HIGH, dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([A_MIN], dtype=np.float32),
                                       high=np.array([A_MAX], dtype=np.float32), dtype=np.float32)

        # internal state
        self.x_ego, self.x_lead, self.v_ego, self.v_lead, self.a_ego = 0.0, 20.0, V_REF-0.5, V_REF, 0.0

        # attack-aware override
        self._safety_obs_override = None

        # logging
        self.current_step, self.max_steps, self.collision = 0, 400, False

        self.reset()

    # ---------- Helpers ----------
    def _obs_to_array(self):
        return np.array([self.x_lead - self.x_ego, self.v_lead - self.v_ego, self.v_ego], dtype=np.float32)

    def _normalize(self, obs: np.ndarray) -> np.ndarray:
        return (2.0 * (obs - OBS_LOW) / (OBS_HIGH - OBS_LOW + 1e-8) - 1.0).astype(np.float32)

    def _denormalize(self, obs_norm: np.ndarray) -> np.ndarray:
        return (((obs_norm + 1.0) / 2.0) * (OBS_HIGH - OBS_LOW) + OBS_LOW).astype(np.float32)

    def set_safety_obs_for_filter(self, obs):
        self._safety_obs_override = np.array(obs, dtype=np.float32)

    def clear_safety_obs_for_filter(self):
        self._safety_obs_override = None

    def _compute_amax_safe(self, obs_for_filter: np.ndarray) -> float:
        dx, dv, v = float(obs_for_filter[0]), float(obs_for_filter[1]), float(obs_for_filter[2])
        num = dx - TH * v + (self.v_lead - v) * DT
        denom = TH * DT
        if denom <= 0: return A_MIN
        return float(np.clip(num / (denom + 1e-8), A_MIN, A_MAX))

    def _apply_safety(self, a_rl: float, dx: float, dv: float, v: float) -> float:
        """
        Optional helper: compute the CBF clamp given a proposed action and a raw state (dx,dv,v).
        (Your step() already handles the override + clamp; this is just for completeness.)
        """
        a_safe_max = self._compute_amax_safe(np.array([dx, dv, v], dtype=np.float32))
        a_clamped = min(a_rl, a_safe_max)
        return float(np.clip(a_clamped, A_MIN, A_MAX))

    # ---------- Gym API ----------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if seed is not None:
            self.seed_val = int(seed)
            self.np_random, _ = gym.utils.seeding.np_random(self.seed_val)

        self.current_step, self.collision = 0, False
        self.x_ego, self.x_lead = 0.0, 30.0 + self.np_random.uniform(-1.0, 1.0)
        self.v_ego = V_REF - 0.5 + self.np_random.uniform(-0.2, 0.2)
        self.v_lead = V_REF + self.np_random.uniform(-0.5, 0.5)
        self.a_ego, self._safety_obs_override = 0.0, None

        obs = self._obs_to_array()
        return (self._normalize(obs) if self.normalize_obs else obs), {}

    def step(self, action: np.ndarray):
        a = float(np.array(action).reshape(-1)[0]) if isinstance(action, (list, tuple, np.ndarray)) else float(action)

        if self._safety_obs_override is not None:
            obs_for_filter = self._safety_obs_override
            if np.max(np.abs(obs_for_filter)) <= 1.1 and self.normalize_obs:
                obs_for_filter = self._denormalize(obs_for_filter)
            self._safety_obs_override = None
        else:
            obs_for_filter = self._obs_to_array()

        a_clamped = self._apply_safety(a, obs_for_filter[0], obs_for_filter[1], obs_for_filter[2])

        self.x_ego += self.v_ego * DT + 0.5 * a_clamped * DT * DT
        self.v_ego = float(np.clip(self.v_ego + a_clamped * DT, 0.0, 100.0))
        self.a_ego = a_clamped

        if hasattr(self, "lead_acc") and self.lead_acc is not None:
            self.v_lead = float(np.clip(self.v_lead + self.lead_acc * DT, 0.0, 100.0))
            self.x_lead += self.v_lead * DT + 0.5 * self.lead_acc * DT * DT
        else:
            self.x_lead += self.v_lead * DT

        self.current_step += 1

        obs = self._obs_to_array()
        collision = obs[0] <= 0.0
        if collision: self.collision = True

        info = {
            "collision": collision,
            "dx": float(obs[0]),
            "ego_v": float(obs[2]),
            "lead_v": float(self.v_lead),
            "applied_action": float(a_clamped),
        }

        speed_err = (self.v_ego - V_REF)
        dsafe = D0 + TH * self.v_ego
        safe_pen = max(0.0, dsafe - obs[0])
        action_pen = 0.5 * (a_clamped ** 2)
        reward = -(0.5 * speed_err**2 + 2.0 * safe_pen**2 + 0.01 * action_pen)

        done = self.collision or (self.current_step >= self.max_steps)
        if self.collision: info["terminal_reason"] = "collision"
        elif self.current_step >= self.max_steps: info["terminal_reason"] = "time_limit"

        obs_out = self._normalize(obs) if self.normalize_obs else obs
        return obs_out, float(reward), bool(done and self.collision), bool(done and not self.collision), info

    def render(self, mode="human"):
        print(f"step={self.current_step:03d}  dx={self.x_lead - self.x_ego:6.3f}  "
              f"v_e={self.v_ego:5.2f}  v_l={self.v_lead:5.2f}  a={self.a_ego:5.2f}")

    def close(self):
        pass


# In[6]:


# === Sanity check for ACCEnv with safety filter ===
env = ACCEnv(normalize_obs=True, seed=42)

obs, _ = env.reset()
print("Reset OK. Initial obs (normalized):", obs)

# 1. Step normally with an aggressive action
print("\n[Baseline step]")
action = np.array([2.0], dtype=np.float32)  # strong acceleration
obs, r, term, trunc, info = env.step(action)
print("Applied action=2.0 -> clamped:", info["applied_action"], 
      "dx:", info["dx"], "collision:", info["collision"])

# 2. Force override: fake small headway to trigger clamp
print("\n[Override with adversarial observation]")
obs_fake_close = np.array([-0.99, 0.0, 0.0], dtype=np.float32)  # tiny headway, normalized
env.set_safety_obs_for_filter(obs_fake_close)
obs, r, term, trunc, info = env.step(action)
print("Applied action=2.0 with override -> clamped:", info["applied_action"], 
      "dx:", info["dx"], "collision:", info["collision"])

# 3. Check that override is consumed (next step should use true obs again)
print("\n[Next step after override consumed]")
obs, r, term, trunc, info = env.step(action)
print("Applied action=2.0 (no override) -> clamped:", info["applied_action"], 
      "dx:", info["dx"], "collision:", info["collision"])

# 4. Directly test _apply_safety helper
print("\n[Direct _apply_safety check]")
print("Clamp from helper (dx=2.0, dv=0.0, v=15.0):",
      env._apply_safety(a_rl=2.0, dx=2.0, dv=0.0, v=15.0))


# In[ ]:


# Cell 3: quick sanity run
if __name__ == '__main__':
    env = ACCEnv(normalize_obs=False, seed=42)
    obs, _ = env.reset()
    print('init obs:', obs)
    done = False
    total_r = 0.0
    while not done:
        # simple policy: try to track target speed with a P-controller
        v = obs[2]
        a_cmd = 0.5 * (V_REF - v)
        a_cmd = float(np.clip(a_cmd, A_MIN, A_MAX))
        obs, r, done, trunc, info = env.step([a_cmd])
        total_r += r
        if env.current_step % 50 == 0:
            env.render()
    print('episode return:', total_r, 'collision:', info.get('collision', False))


# In[ ]:


# Logging helpers

def run_and_log_one_episode(env, policy_fn, attack_fn=None, eps=0.01, capture_trace=True, seed=None):
    """
    Runs one episode with given policy function (callable obs->action) and optional attack_fn(obs)->adv_obs.
    Returns: dict containing per-step traces and episode metadata.
    """
    if seed is not None:
        env.reset(seed=seed)
    else:
        env.reset()

    obs, _ = env.reset() if isinstance(env.reset(), tuple) else env.reset()
    traces = {'t': [], 'dx': [], 'dv': [], 'v': [], 'action': [], 'applied_action': [], 'lead_v': []}
    ep_return = 0.0
    ep_collided = False

    while True:
        obs_for_policy = obs
        if attack_fn is not None:
            adv_obs = attack_fn(obs, eps)
            if hasattr(env, 'set_safety_obs_for_filter'):
                env.set_safety_obs_for_filter(adv_obs)
            obs_for_policy = adv_obs
        else:
            if hasattr(env, 'clear_safety_obs_for_filter'):
                env.clear_safety_obs_for_filter()

        action = policy_fn(obs_for_policy)
        next_obs, r, done, trunc, info = env.step(action)
        ep_return += r
        ep_collided = ep_collided or bool(info.get('collision', False))

        if capture_trace:
            if env.normalize_obs:
                raw_obs = env._denormalize(obs if attack_fn is None else adv_obs)
            else:
                raw_obs = obs.copy()
            traces['t'].append(env.current_step * DT)
            traces['dx'].append(float(raw_obs[0]))
            traces['dv'].append(float(raw_obs[1]))
            traces['v'].append(float(raw_obs[2]))
            traces['action'].append(float(action[0] if isinstance(action, (list, tuple, np.ndarray)) else action))
            traces['applied_action'].append(float(info.get('applied_action', np.nan)))
            traces['lead_v'].append(float(info.get('lead_v', np.nan)))

        obs = next_obs
        if done:
            break

    return {
        'return': float(ep_return),
        'collision': bool(ep_collided),
        'traces': traces,
    }

# Example policy function for testing (simple PD/P-controller)
def simple_policy(obs):
    if isinstance(obs, np.ndarray) and obs.shape[-1] == 3 and (np.max(np.abs(obs)) <= 1.1):
        raw = env._denormalize(obs)
    else:
        raw = obs
    v = float(raw[2])
    a_cmd = 1.0 * (V_REF - v)
    return np.array([float(np.clip(a_cmd, A_MIN, A_MAX))], dtype=np.float32)


# In[ ]:


# Example wiring for evaluation with SB3 + VecNormalize

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3 import PPO

# Create base env factory
def make_env_fn(normalize_obs=False):
    def _f():
        return ACCEnv(normalize_obs=normalize_obs)
    return _f

# Create a vectorized env and wrap with VecNormalize
vec_env = DummyVecEnv([make_env_fn(normalize_obs=False)])
vec_norm = VecNormalize(vec_env, norm_obs=True, norm_reward=False, clip_obs=10.)

# Load or create a model (if you have a saved model, load it instead)
# model = PPO('MlpPolicy', vec_norm, verbose=1)  # for training
# model.learn(total_timesteps=200000)
# model.save('ppo_acc.zip')

# Example: load a trained model (adjust path as needed)
# model = PPO.load('ppo_acc.zip', env=vec_norm)

def eval_model_with_attacks(model, vec_norm, n_episodes=100, attack_fn=None, eps=0.01):
    vec_norm.training = False
    vec_norm.norm_reward = False

    results = {'episodes': n_episodes, 'collision_count': 0, 'returns': []}
    for ep in range(n_episodes):
        obs = vec_norm.reset()
        obs = obs[0]
        ep_coll = False
        ep_ret = 0.0
        done = False
        while not done:
            obs_for_policy = obs
            if attack_fn is not None:
                adv = attack_fn(obs, eps)
                inner_env = vec_norm.envs[0]
                if hasattr(inner_env, 'set_safety_obs_for_filter'):
                    inner_env.set_safety_obs_for_filter(adv)
                obs_for_policy = adv
            else:
                inner_env = vec_norm.envs[0]
                if hasattr(inner_env, 'clear_safety_obs_for_filter'):
                    inner_env.clear_safety_obs_for_filter()

            action, _states = model.predict(obs_for_policy, deterministic=True)
            obs, reward, terminated, truncated, info = vec_norm.step(action)
            obs = obs[0]
            ep_ret += float(np.asarray(reward).sum())
            ep_coll = ep_coll or bool(info[0].get('collision', False))
            done = bool(terminated[0]) or bool(truncated[0])

        results['returns'].append(ep_ret)
        results['collision_count'] += int(ep_coll)

    results['collision_rate'] = results['collision_count'] / max(1, n_episodes)
    results['return_mean'] = float(np.mean(results['returns']))
    return results

# Example usage:
# res_base = eval_model_with_attacks(model, vec_norm, n_episodes=100, attack_fn=None, eps=0.0)
# res_fgsm = eval_model_with_attacks(model, vec_norm, n_episodes=100, attack_fn=fgsm_fn, eps=0.01)
# res_oia  = eval_model_with_attacks(model, vec_norm, n_episodes=100, attack_fn=oia_fn, eps=0.01)


# In[ ]:


# Save/export helpers

import os
import json
import numpy as np
from typing import Dict, Any

def save_traces_and_metrics(out_dir: str, traces_list: list, metrics: Dict[str, Any]):
    """
    Save metrics to metrics.json and each episode's traces as compressed .npz files.

    Args:
        out_dir: directory path to save everything
        traces_list: list of dicts, each from run_and_log_one_episode (with 'traces' inside)
        metrics: dict of aggregate metrics (e.g., collision_rate, return_mean, etc.)
    """
    os.makedirs(out_dir, exist_ok=True)

    # Save metrics.json
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save per-episode traces
    for i, tr in enumerate(traces_list):
        np.savez_compressed(
            os.path.join(out_dir, f'trace_ep_{i:03d}.npz'),
            **tr['traces']
        )

    print(f"Saved metrics and {len(traces_list)} traces to {out_dir}")

