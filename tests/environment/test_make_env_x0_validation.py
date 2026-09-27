import numpy as np
import pytest

from pcgym import make_env


@pytest.fixture
def env_params():
    return {
        "model": "cstr",
        "N": 20,
        "tsim": 5,
        "SP": {"Ca": [0.85] * 20},
        "a_space": {"low": np.array([295]), "high": np.array([302])},
        "o_space": {"low": np.array([0.7, 300, 0.8]), "high": np.array([1, 350, 0.9])},
        "x0": np.array([0.8, 330, 0.8]),
        "integration_method": "casadi",
    }


def test_correct_x0_is_accepted(env_params):
    make_env(env_params)


def test_x0_missing_setpoint_entry_raises(env_params):
    env_params["x0"] = np.array([0.8, 330])
    with pytest.raises(ValueError, match=r"x0 has 2 entries but 3 are expected.*'Ca', 'T', 'Ca_SP'"):
        make_env(env_params)


def test_x0_too_long_raises(env_params):
    env_params["x0"] = np.array([0.8, 330, 0.8, 1.0])
    with pytest.raises(ValueError, match="x0 has 4 entries but 3 are expected"):
        make_env(env_params)


def test_x0_without_setpoints_is_model_states_only():
    params = {
        "model": "cstr",
        "N": 20,
        "tsim": 5,
        "a_space": {"low": np.array([295]), "high": np.array([302])},
        "o_space": {"low": np.array([0.7, 300]), "high": np.array([1, 350])},
        "x0": np.array([0.8, 330]),
        "reward_states": ["Ca"],
        "maximise_reward": True,
        "integration_method": "casadi",
    }
    make_env(params)
    params["x0"] = np.array([0.8, 330, 0.8])
    with pytest.raises(ValueError, match="x0 has 3 entries but 2 are expected"):
        make_env(params)


def test_o_space_missing_setpoint_entry_raises(env_params):
    env_params["o_space"] = {"low": np.array([0.7, 300]), "high": np.array([1, 350])}
    with pytest.raises(ValueError, match="o_space has 2 entries but 3 are expected"):
        make_env(env_params)
