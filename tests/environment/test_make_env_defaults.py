"""Tests for falling back to a model's default a_space / o_space / x0."""

import numpy as np
import pytest

from pcgym import make_env
from pcgym.model_classes import cstr
from pcgym.model_defaults import get_default, has_defaults


def test_minimal_config_uses_all_defaults():
    """A model with registered defaults runs given only model/N/tsim/SP."""
    env = make_env(
        {
            "model": "four_tank",
            "N": 10,
            "tsim": 1000,
            "integration_method": "casadi",
            "SP": {"h3": [0.5] * 10, "h4": [0.2] * 10},
        }
    )
    obs, _ = env.reset()
    obs2, _, _, _, _ = env.step(env.action_space.sample())

    # Defaults were injected into env_params.
    assert np.array_equal(env.env_params["a_space"]["high"], get_default("four_tank", "a_space")["high"])
    assert np.array_equal(env.env_params["x0"], get_default("four_tank", "x0"))
    assert obs.shape == obs2.shape


def test_partial_override_keeps_user_values():
    """User-supplied keys are respected; only the omitted ones default."""
    custom_a = {"low": np.array([0.0, 0.0]), "high": np.array([5.0, 5.0])}
    env = make_env(
        {
            "model": "four_tank",
            "N": 10,
            "tsim": 1000,
            "integration_method": "casadi",
            "SP": {"h3": [0.5] * 10, "h4": [0.2] * 10},
            "a_space": custom_a,
        }
    )
    # a_space kept, x0/o_space defaulted.
    assert np.array_equal(env.env_params["a_space"]["high"], custom_a["high"])
    assert np.array_equal(env.env_params["x0"], get_default("four_tank", "x0"))


def test_custom_model_requires_explicit_spaces():
    with pytest.raises(ValueError, match="custom_model"):
        make_env(
            {
                "custom_model": cstr(),
                "N": 10,
                "tsim": 10,
                "integration_method": "casadi",
                "SP": {"Ca": [0.85] * 10},
            }
        )


def test_model_without_defaults_raises_helpfully():
    with pytest.raises(ValueError, match="has no registered"):
        make_env(
            {
                "model": "batch",
                "N": 10,
                "tsim": 10,
                "integration_method": "casadi",
                "reward_states": ["Cc"],
                "maximise_reward": True,
            }
        )


def test_defaults_are_fresh_copies():
    """Mutating a returned default must not corrupt the shared table."""
    x0_a = get_default("cstr", "x0")
    x0_a[0] = 999.0
    x0_b = get_default("cstr", "x0")
    assert x0_b[0] != 999.0


def test_has_defaults():
    assert has_defaults("distillation_column")
    assert not has_defaults("batch")


if __name__ == "__main__":
    pytest.main([__file__])
