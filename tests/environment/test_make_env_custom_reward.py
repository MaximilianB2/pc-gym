import pytest
import numpy as np
import sys
sys.path.append("..\src\pcgym") # Add local pc-gym files to path.

from pcgym import make_env

def custom_reward_function(env, state, action, constraint_violated):
    return -np.sum(np.square(state))

def test_make_env_custom_reward():
    env_params = {
        "model": "cstr",
        "a_space": {"low": np.array([-1]), "high": np.array([1])},
        "o_space": {"low": np.array([-1, -1]), "high": np.array([1, 1])},
        "SP": {"T": [350] * 100},
        "N": 100,
        "tsim": 10,
        "x0": np.array([0.5, 350]),
        "custom_reward": custom_reward_function
    }
    env = make_env(env_params)
    assert env.custom_reward
    assert env.custom_reward_f == custom_reward_function
    
    env.reset()
    action = env.action_space.sample()
    _, reward, _, _, _ = env.step(action)
    assert isinstance(reward, float)
    assert reward <= 0  # Given our custom reward function