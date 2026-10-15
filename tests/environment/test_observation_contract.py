"""Observation regressions and bounded episodes for the first four repair candidates."""

import copy
import itertools

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from pcgym import make_env


@pytest.fixture(params=[True, False], ids=["normalised", "physical"])
def normalise_o(request):
    return request.param


@pytest.fixture(
    params=[("first_order_system", 10), ("cstr", 26), ("nonsmooth_control", 5), ("multistage_extraction", 60)],
    ids=lambda case: case[0],
)
def candidate(request, normalise_o):
    model, horizon = request.param
    # State, action and task values come from the model's registered defaults.
    env = make_env(
        {"model": model, "N": 60, "tsim": horizon, "integration_method": "casadi", "normalise_o": normalise_o}
    )
    yield env
    env.close()


def _assert_observation(env, obs):
    assert obs.dtype == env.observation_space.dtype == np.dtype(np.float32)
    assert np.isfinite(obs).all()
    assert env.observation_space.contains(obs), (obs, env.observation_space)


def test_default_task_passes_gymnasium_checker(candidate):
    candidate.action_space.seed(0)
    check_env(candidate, skip_render_check=True)


@pytest.mark.parametrize("profile", ["midpoint", "corners", "alternating"] + [f"random_{i}" for i in range(10)])
def test_default_task_full_episode_observations(candidate, profile):
    env = candidate
    n_inputs = env.action_space.shape[0]
    shape = (env.N, n_inputs)
    if profile == "corners":
        sequences = [np.tile(corner, (env.N, 1)) for corner in itertools.product((-1.0, 1.0), repeat=n_inputs)]
    elif profile == "midpoint":
        sequences = [np.zeros(shape)]
    elif profile == "alternating":
        sequences = [np.tile(np.where(np.arange(env.N) % 2, 1.0, -1.0)[:, None], (1, n_inputs))]
    else:
        seed = int(profile.removeprefix("random_"))
        sequences = [np.random.default_rng(seed).uniform(-1, 1, shape)]

    for actions in sequences:
        obs, _ = env.reset(seed=0)
        _assert_observation(env, obs)
        for step, action in enumerate(actions.astype(np.float32), 1):
            assert env.action_space.contains(action)
            obs, reward, terminated, truncated, _ = env.step(action)
            _assert_observation(env, obs)
            assert np.isfinite(env.state).all()
            assert np.isfinite(reward)
            assert terminated is False
            assert truncated is (step == env.N)


@pytest.mark.parametrize("target", [0.8, 0.9])
def test_decimal_bound_endpoints(target, normalise_o):
    env = make_env(
        {
            "model": "cstr",
            "N": 4,
            "tsim": 1,
            "SP": {"Ca": [target] * 4},
            "x0": np.array([0.8, 330, target]),
            "normalise_o": normalise_o,
        }
    )
    try:
        assert env.observation_space_base.low[-1] == 0.8
        assert env.observation_space_base.high[-1] == 0.9
        expected = (-1.0 if target == 0.8 else 1.0) if normalise_o else np.float32(target)
        obs, _ = env.reset(seed=0)
        _assert_observation(env, obs)
        assert obs[-1] == expected
        built = env.build_obs(states=[0.8, 330], setpoints=[target], normalise=normalise_o)
        assert built.dtype == np.float64
        np.testing.assert_array_equal(built.astype(np.float32), obs)
        obs, *_ = env.step(np.zeros(1, dtype=np.float32))
        _assert_observation(env, obs)
        assert obs[-1] == expected
    finally:
        env.close()


@pytest.mark.parametrize("features", ["disturbance", "uncertainty", "both"])
def test_appended_bounds_keep_physical_precision(features, normalise_o):
    params = {"model": "cstr", "N": 4, "tsim": 1, "normalise_o": normalise_o}
    low, high = [0.7, 300, 0.8], [1.0, 350, 0.9]
    if features in ("disturbance", "both"):
        params.update(
            disturbances={"Caf": np.full(4, 1.1)},
            disturbance_bounds={"low": np.array([0.9]), "high": np.array([1.1])},
        )
        low.append(0.9)
        high.append(1.1)
    if features in ("uncertainty", "both"):
        params.update(
            uncertainty_percentages={"k0": 0.0},
            uncertainty_bounds={"low": np.array([6.0e10]), "high": np.array([7.2e10])},
        )
        low.append(6.0e10)
        high.append(7.2e10)
    env = make_env(params)
    try:
        np.testing.assert_array_equal(env.observation_space_base.low, low)
        np.testing.assert_array_equal(env.observation_space_base.high, high)
        obs, _ = env.reset(seed=0)
        for step in range(env.N + 1):
            _assert_observation(env, obs)
            expected = np.ones(len(high) - 2) if normalise_o else np.asarray(high[2:], dtype=np.float32)
            np.testing.assert_array_equal(obs[2:], expected)
            if step < env.N:
                obs, *_ = env.step(np.zeros(1, dtype=np.float32))
    finally:
        env.close()


@pytest.mark.parametrize("initial", [np.array([0, 0]), [0, 0]], ids=["integer_array", "integer_list"])
def test_integer_initial_state_follows_analytic_solution(initial, normalise_o):
    env = make_env(
        {
            "model": "first_order_system",
            "N": 4,
            "tsim": 1,
            "x0": initial,
            "SP": {"x": [0.0] * 4},
            "normalise_a": False,
            "normalise_o": normalise_o,
        }
    )
    try:
        env.reset(seed=0)
        for step in range(1, env.N + 1):
            obs, *_ = env.step(np.array([0.5], dtype=np.float32))
            expected = env.model.K * 0.5 * (1 - np.exp(-step * env.dt / env.model.tau))
            assert env.state[0] == pytest.approx(expected, abs=2e-6)
            assert env.state.dtype == np.float64
            _assert_observation(env, obs)
        np.testing.assert_array_equal(initial, [0, 0])
        env.reset(seed=0)
        np.testing.assert_array_equal(env.state, [0.0, 0.0])
    finally:
        env.close()


def test_out_of_bounds_state_is_not_clipped(normalise_o):
    env = make_env(
        {
            "model": "first_order_system",
            "N": 4,
            "tsim": 1,
            "model_params": {"K": 10.0},
            "normalise_o": normalise_o,
        }
    )
    try:
        env.reset(seed=0)
        obs, *_ = env.step(np.ones(1, dtype=np.float32))
        assert env.state[0] > env.observation_space_base.high[0]
        assert obs[0] > env.observation_space.high[0]
        assert not env.observation_space.contains(obs)
    finally:
        env.close()


def test_observations_and_info_are_independent_snapshots(normalise_o):
    env = make_env(
        {
            "model": "cstr",
            "N": 4,
            "tsim": 1,
            "normalise_o": normalise_o,
            "partial_observation": ["Ca"],
            "constraints": {"T": [300, 350]},
            "done_on_cons_vio": False,
            "r_penalty": False,
        }
    )
    try:
        obs, info = env.reset(seed=0)
        assert obs.dtype == np.float32
        assert obs[1] == 0.0
        assert info["obs"][1] != 0.0  # evaluation retains the full measurement
        assert env.obs[1] == env.state[1] == 330.0
        assert info["obs"].dtype == np.float64
        obs_before, info_before = obs.copy(), copy.deepcopy(info)
        new_obs, *_, new_info = env.step(np.zeros(1, dtype=np.float32))
        np.testing.assert_array_equal(obs, obs_before)
        assert info.keys() == info_before.keys()
        for key in info:
            np.testing.assert_array_equal(info[key], info_before[key])

        state_before, measured_before = env.state.copy(), env.obs.copy()
        new_obs[:] = 123
        new_info["obs"][:] = 123
        new_info["cons_info"][:] = 123
        np.testing.assert_array_equal(env.state, state_before)
        np.testing.assert_array_equal(env.obs, measured_before)
        assert not np.any(env.info["obs"] == 123)
        assert not np.any(env.info["cons_info"] == 123)
        env.reset(seed=0)
        for key in info:
            np.testing.assert_array_equal(info[key], info_before[key])
    finally:
        env.close()
