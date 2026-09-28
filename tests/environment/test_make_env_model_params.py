"""Tests for overriding model parameters via ``env_params["model_params"]``."""

import numpy as np
import pytest

from pcgym import make_env

BASE = {
    "model": "cstr",
    "N": 10,
    "tsim": 10,
    "integration_method": "casadi",
    "a_space": {"low": np.array([295.0]), "high": np.array([302.0])},
    "o_space": {"low": np.array([0.7, 300, 0.8]), "high": np.array([1.0, 350, 0.9])},
    "SP": {"Ca": [0.85] * 10},
    "x0": np.array([0.8, 330, 0.8]),
}


def _params(**overrides):
    params = {k: (v.copy() if isinstance(v, dict) else v) for k, v in BASE.items()}
    params.update(overrides)
    return params


def test_model_params_override_defaults():
    default_env = make_env(_params())
    custom_env = make_env(_params(model_params={"k0": 1.0e10, "V": 120}))

    assert custom_env.model.k0 == 1.0e10
    assert custom_env.model.V == 120
    # The default instance is unaffected.
    assert default_env.model.k0 != 1.0e10


def test_model_params_changes_dynamics():
    """An overridden parameter should actually change the simulated dynamics."""
    fast = make_env(_params(model_params={"k0": 1.0e11}))
    slow = make_env(_params(model_params={"k0": 1.0e9}))

    fast.reset()
    slow.reset()
    action = np.array([0.0])  # identical (normalised) action for both
    fast_obs, *_ = fast.step(action)
    slow_obs, *_ = slow.step(action)

    assert not np.allclose(fast_obs, slow_obs)


def test_unknown_model_param_raises():
    with pytest.raises(ValueError, match="Unknown model_params"):
        make_env(_params(model_params={"not_a_real_param": 1.0}))


def test_model_params_must_be_dict():
    with pytest.raises(ValueError, match="model_params must be a dictionary"):
        make_env(_params(model_params=[("k0", 1.0)]))


def test_model_params_on_custom_model():
    """Custom-model instances accept overrides too."""
    from pcgym.model_classes import cstr

    env = make_env(_params(custom_model=cstr(), model_params={"k0": 5.0e10}))
    assert env.model.k0 == 5.0e10


if __name__ == "__main__":
    pytest.main([__file__])
