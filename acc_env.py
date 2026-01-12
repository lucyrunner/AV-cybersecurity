import gymnasium as gym
from gymnasium import spaces
import numpy as np


class ACCEnv(gym.Env):
    
    metadata = {"render_modes": ["human"]}
    
    # Physical constants
    DT = 0.1           # time step (s)
    TH = 1.5           # desired time headway (s)
    D0 = 5.0           # standstill distance (m)
    V_REF = 15.0       # target speed (m/s)
    A_MIN = -3.5       # braking limit (m/s^2)
    A_MAX = 2.0        # acceleration limit (m/s^2)
    
    # State space bounds (for Box space definition)
    OBS_LOW = np.array([0.0, -30.0, 0.0], dtype=np.float32)
    OBS_HIGH = np.array([200.0, 30.0, 40.0], dtype=np.float32)
    
    def __init__(self):
        super().__init__()
        
        self.observation_space = spaces.Box(
            low=self.OBS_LOW, 
            high=self.OBS_HIGH, 
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array([self.A_MIN], dtype=np.float32),
            high=np.array([self.A_MAX], dtype=np.float32),
            dtype=np.float32
        )
        
        # Episode settings
        self.max_steps = 400
        
        # State variables (will be initialized in reset)
        self.x_ego = 0.0
        self.x_lead = 0.0
        self.v_ego = 0.0
        self.v_lead = 0.0
        self.a_ego = 0.0
        self.current_step = 0
        
        # Lead vehicle deceleration profile (for testing scenarios)
        self.lead_decel_active = False
        self.lead_decel_start = None
        self.lead_decel_duration = None
        self.lead_decel_value = None
        
        # Attack interface: allows external override of observation for safety filter
        self._safety_obs_override = None
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize state with small random variations
        self.x_ego = 0.0
        self.x_lead = 30.0 + self.np_random.uniform(-1.0, 1.0)
        self.v_ego = self.V_REF - 0.5 + self.np_random.uniform(-0.2, 0.2)
        self.v_lead = self.V_REF + self.np_random.uniform(-0.5, 0.5)
        self.a_ego = 0.0
        self.current_step = 0
        
        # Reset lead vehicle deceleration profile
        self.lead_decel_active = False
        self.lead_decel_start = None
        self.lead_decel_duration = None
        self.lead_decel_value = None
        
        # Clear attack override
        self._safety_obs_override = None
        
        obs = self._get_obs()
        return obs, {}
    
    def _get_obs(self):
        """Return current observation [dx, dv, v]"""
        dx = self.x_lead - self.x_ego
        dv = self.v_lead - self.v_ego
        v = self.v_ego
        return np.array([dx, dv, v], dtype=np.float32)
    
    def set_safety_obs_for_filter(self, obs_adv):
        self._safety_obs_override = np.array(obs_adv, dtype=np.float32)
    
    def _compute_safe_action(self, obs_for_filter):
        dx, dv, v = obs_for_filter[0], obs_for_filter[1], obs_for_filter[2]
        
        numerator = dx - self.TH * v + (self.v_lead - v) * self.DT
        denominator = self.TH * self.DT
        
        if denominator <= 0:
            return self.A_MIN
        
        a_max_safe = numerator / denominator
        return float(np.clip(a_max_safe, self.A_MIN, self.A_MAX))
    
    def _apply_safety_filter(self, action_rl):
        # Determine which observation to use for safety filter
        if self._safety_obs_override is not None:
            obs_for_filter = self._safety_obs_override
            self._safety_obs_override = None  # consume override
        else:
            obs_for_filter = self._get_obs()
        
        # Compute maximum safe action and clamp
        a_max_safe = self._compute_safe_action(obs_for_filter)
        a_safe = min(action_rl, a_max_safe)
        a_safe = np.clip(a_safe, self.A_MIN, self.A_MAX)
        
        return float(a_safe)
    
    def step(self, action):
        # Extract scalar action
        a_rl = float(action[0]) if isinstance(action, np.ndarray) else float(action)
        
        # Apply safety filter
        a_safe = self._apply_safety_filter(a_rl)
        
        # Update ego vehicle (forward Euler integration)
        self.x_ego += self.v_ego * self.DT + 0.5 * a_safe * self.DT ** 2
        self.v_ego = np.clip(self.v_ego + a_safe * self.DT, 0.0, 100.0)
        self.a_ego = a_safe
        
        # Update lead vehicle
        if self.lead_decel_active:
            t = self.current_step * self.DT
            if self.lead_decel_start <= t < self.lead_decel_start + self.lead_decel_duration:
                # Apply deceleration
                self.v_lead = max(0.0, self.v_lead + self.lead_decel_value * self.DT)
        
        self.x_lead += self.v_lead * self.DT
        
        # Get new observation
        obs = self._get_obs()
        
        # Check collision
        collision = obs[0] <= 0.0
        
        # Compute reward
        speed_error = self.v_ego - self.V_REF
        d_safe = self.D0 + self.TH * self.v_ego
        safe_violation = max(0.0, d_safe - obs[0])
        
        reward = -(
            0.5 * speed_error ** 2 +
            2.0 * safe_violation ** 2 +
            0.01 * a_safe ** 2
        )
        
        # Update step counter
        self.current_step += 1
        
        # Check termination
        terminated = collision
        truncated = self.current_step >= self.max_steps
        
        info = {
            "collision": collision,
            "dx": float(obs[0]),
            "ego_v": float(obs[2]),
            "lead_v": float(self.v_lead),
            "applied_action": float(a_safe),
            "rl_action": float(a_rl),
        }
        
        return obs, float(reward), terminated, truncated, info
    
    def render(self):
        dx = self.x_lead - self.x_ego
        print(f"Step {self.current_step:3d} | dx={dx:6.2f}m | "
              f"v_ego={self.v_ego:5.2f}m/s | v_lead={self.v_lead:5.2f}m/s | "
              f"a={self.a_ego:5.2f}m/s²")
    
    def activate_lead_deceleration(self, start_time=5.0, duration=3.0, decel=-2.0):

        self.lead_decel_active = True
        self.lead_decel_start = start_time
        self.lead_decel_duration = duration
        self.lead_decel_value = decel