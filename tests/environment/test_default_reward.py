import numpy as np
import pytest

from pcgym import make_env
from pcgym.models import MODEL_REGISTRY, Batch, Regulation

N = 10
REGULATION_MODELS = sorted(name for name, spec in MODEL_REGISTRY.items() if isinstance(spec.task, Regulation))


def test_regulation_models_have_defaults_for_a_minimal_env():
    for name in REGULATION_MODELS:
        assert MODEL_REGISTRY[name].defaults is not None, name


@pytest.mark.parametrize("model", REGULATION_MODELS)
def test_minimal_env_uses_default_task(model):
    env = make_env({"model": model, "N": N, "tsim": 1})
    task = MODEL_REGISTRY[model].task
    states = env.model.info()["states"]
    low, high = env.observation_space_base.low, env.observation_space_base.high

    assert set(env.SP) == set(task.setpoint)
    for k, v in task.setpoint.items():
        i = states.index(k)
        assert env.SP[k] == [v] * N  # constant setpoint
        assert low[i] <= v <= high[i]  # reachable within the observation bounds
        assert env.x0[i] != v  # the default task is not trivially satisfied by x0
        assert env.env_params["r_scale"][k] == pytest.approx(1 / (high[i] - low[i]) ** 2)

    obs, _ = env.reset(seed=0)
    n_x = len(states)
    np.testing.assert_allclose(env.state[n_x : n_x + len(task.setpoint)], list(task.setpoint.values()))


def test_cstr_default_reward_is_range_normalised():
    env = make_env({"model": "cstr", "N": N, "tsim": 2, "normalise_o": False})
    env.reset()
    _, r, *_ = env.step(np.array([0.0]))
    assert r == pytest.approx(-((env.state[0] - 0.9) ** 2) / 0.3**2)


def test_explicit_setpoint_keeps_existing_behaviour():
    env = make_env({"model": "cstr", "N": N, "tsim": 2, "SP": {"Ca": [0.85] * N}})
    assert env.SP == {"Ca": [0.85] * N}
    assert "r_scale" not in env.env_params
    np.testing.assert_allclose(env.x0, [0.8, 330, 0.8])  # user/default x0 untouched


def test_user_r_scale_is_respected():
    env = make_env({"model": "cstr", "N": N, "tsim": 2, "r_scale": {"Ca": 5.0}})
    assert env.env_params["r_scale"] == {"Ca": 5.0}
    assert env.SP == {"Ca": [0.9] * N}


def test_model_state_only_x0_and_o_space_are_extended():
    env = make_env(
        {
            "model": "cstr",
            "N": N,
            "tsim": 2,
            "x0": np.array([0.8, 330.0]),
            "o_space": {"low": np.array([0.7, 300]), "high": np.array([1.0, 350])},
        }
    )
    np.testing.assert_allclose(env.x0, [0.8, 330, 0.9])
    np.testing.assert_allclose(env.observation_space_base.low, [0.7, 300, 0.7])
    np.testing.assert_allclose(env.observation_space_base.high, [1.0, 350, 1.0])


def test_batch_model_default_reward_is_end_of_episode_yield():
    assert MODEL_REGISTRY["batch"].task == Batch(reward_states=("Cb",))
    env = make_env(
        {
            "model": "batch",
            "N": N,
            "tsim": 2,
            "x0": np.array([1.0, 0.0, 0.0, 300.0]),
            "a_space": {"low": np.array([290.0]), "high": np.array([310.0])},
            "o_space": {"low": np.array([0, 0, 0, 250.0]), "high": np.array([2, 2, 2, 400.0])},
        }
    )
    env.reset()
    rewards = [env.step(np.array([0.0]))[1] for _ in range(N)]
    assert all(r == 0 for r in rewards[:-1])
    assert rewards[-1] == pytest.approx(env.state[1]) and rewards[-1] > 0


def test_model_without_task_needs_a_reward():
    params = {
        "model": "hydraulic_tank",
        "N": N,
        "tsim": 2,
        "x0": np.array([1.0, 0.5]),
        "a_space": {"low": np.array([-0.5]), "high": np.array([0.5])},
        "o_space": {"low": np.array([0.0, 0.0]), "high": np.array([2.0, 2.0])},
    }
    with pytest.raises(ValueError, match="No reward is configured"):
        make_env(params)
