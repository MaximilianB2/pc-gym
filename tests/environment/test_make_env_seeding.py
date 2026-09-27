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
        "normalise_o": False,
        "noise": True,
        "noise_percentage": 0.01,
        "uncertainty_percentages": {"k0": 0.2},
        "distribution": "normal",
        "uncertainty_bounds": {"low": np.array([1e9]), "high": np.array([1e11])},
        "integration_method": "casadi",
    }


def _trajectory(env, seed):
    obs, _ = env.reset(seed=seed)
    traj = [obs.copy()]
    for _ in range(5):
        obs, *_ = env.step(np.array([0.0]))
        traj.append(obs.copy())
    return np.array(traj)


def test_same_seed_gives_identical_trajectories(env_params):
    a = _trajectory(make_env(env_params), seed=123)
    b = _trajectory(make_env(env_params), seed=123)
    np.testing.assert_array_equal(a, b)


def test_different_seeds_give_different_trajectories(env_params):
    a = _trajectory(make_env(env_params), seed=1)
    b = _trajectory(make_env(env_params), seed=2)
    assert not np.allclose(a, b)


def test_reset_without_seed_does_not_repeat_episodes(env_params):
    env = make_env(env_params)
    a = _trajectory(env, seed=7)
    b = _trajectory(env, seed=None)
    assert not np.allclose(a, b)


def test_seeding_does_not_touch_global_numpy_rng(env_params):
    np.random.seed(0)
    expected = np.random.rand()
    np.random.seed(0)
    _trajectory(make_env(env_params), seed=5)
    assert np.random.rand() == expected


def test_empirical_distribution_is_seeded(env_params):
    del env_params["uncertainty_percentages"], env_params["distribution"]
    env_params["empirical_distribution"] = {"k0": np.linspace(5e10, 9e10, 50)}
    a = make_env(env_params)
    b = make_env(env_params)
    a.reset(seed=11)
    b.reset(seed=11)
    assert a.model.k0 == b.model.k0
