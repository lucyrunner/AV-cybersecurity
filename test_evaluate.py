import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

from acc_env import ACCEnv
from evaluate import evaluate_single_condition


class DummyModel:
    def predict(self, obs, deterministic=True):
        return np.array([[0.0]], dtype=np.float32), None


def test_evaluate_single_condition_handles_vec_env_step_output():
    vec_env = DummyVecEnv([lambda: ACCEnv()])
    base_env = vec_env.envs[0]
    model = DummyModel()

    result = evaluate_single_condition(
        vec_env=vec_env,
        base_env=base_env,
        model=model,
        attack=None,
        n_episodes=1,
        scenario_config={"use_lead_decel": False},
        verbose=False,
    )

    assert isinstance(result["collision_rate"], float)
    assert len(result["episodes"]) == 1
