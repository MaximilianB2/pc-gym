import pytest
import numpy as np
from pcgym import make_env

@pytest.fixture
def env_params():
    return {
        'model': 'cstr',
        'N': 120,
        'tsim': 26,
        'SP': {'Ca': [0.85] * 40 + [0.9] * 40 + [0.87] * 40},
        'a_space': {'low': np.array([295]), 'high': np.array([302])},
        'o_space': {'low': np.array([0.7, 300, 0.8]), 'high': np.array([1, 350, 0.9])},
        'x0': np.array([0.8, 330, 0.8]),
        'r_scale': {'Ca': 1e3},
        'normalise_a': True,
        'normalise_o': True,
        'noise': True,
        'integration_method': 'casadi',
        'noise_percentage': 0.001
    }

def test_make_env_initialization(env_params):
    env = make_env(env_params)
    assert env.model.__class__.__name__ == "cstr"
    assert env.N == 120
    assert env.tsim == 26
    assert env.normalise_a == True
    assert env.normalise_o == True

def test_make_env_reset(env_params):
    env = make_env(env_params)
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (3,)
    if env.normalise_o:
        assert all(-1.001 <= o <= 1.0001 for o in obs)
    else:
        assert all(env.observation_space.low <= o <= env.observation_space.high for o in obs)
    assert isinstance(info, dict)

def test_make_env_step(env_params):
    env = make_env(env_params)
    env.reset()
    action = env.action_space.sample()
    assert action.shape == (1,)

    assert -1 <= action[0] <= 1

    obs, reward, done, truncated, info = env.step(action)
    assert isinstance(obs, np.ndarray)
    assert obs.shape == (3,)
    assert all(-1.0001 <= o <= 1.0001 for o in obs)  # Check if observations are normalized
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

def test_env_action_space(env_params):
    env = make_env(env_params)
    assert env.action_space.shape == (1,)
    assert env.action_space.low[0] == -1
    assert env.action_space.high[0] == 1

def test_env_observation_space(env_params):
    env = make_env(env_params)
    assert env.observation_space.shape == (3,)
    np.testing.assert_array_almost_equal(env.observation_space.low, np.array([-1, -1, -1]))
    np.testing.assert_array_almost_equal(env.observation_space.high, np.array([1, 1, 1]))