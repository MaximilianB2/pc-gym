"""Episode, stochastic observation and invalid-input regressions for training consumers."""

import copy
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
from casadi import DM

from pcgym import make_env


def params(**extra):
    config = {"model": "first_order_system", "N": 4, "tsim": 1}
    config.update(extra)
    return config


def batch_params(**extra):
    return params(reward_states=["x"], x0=[0.2], o_space={"low": [0.0], "high": [1.0]}, **extra)


@pytest.mark.parametrize("normalise_o", [True, False])
def test_batch_time_and_terminal_objective(normalise_o):
    env = make_env(batch_params(normalise_o=normalise_o))
    obs, _ = env.reset(seed=42)
    assert env.observation_info() == [("x", "state"), ("time_remaining", "time")]
    assert env.Nx == len(obs) == 2
    np.testing.assert_allclose(env.build_obs(states=[0.2], normalise=normalise_o), obs)
    for t in range(1, env.N + 1):
        obs, reward, term, trunc, _ = env.step([0.0])
        remaining = 1 - t / env.N
        assert env.state[-1] == pytest.approx(remaining)
        assert obs[-1] == pytest.approx(2 * remaining - 1 if normalise_o else remaining)
        assert env.observation_space.contains(obs)
        assert term is (t == env.N)
        assert trunc is False
        assert reward == (pytest.approx(env.state[0]) if term else 0)
        np.testing.assert_allclose(
            env.build_obs(states=env.state[:1], time_remaining=remaining, normalise=normalise_o), obs, atol=1e-7
        )


def test_custom_reward_can_declare_finite_horizon():
    env = make_env(params(custom_reward=lambda *args: 0.0, SP={"x": [0.7]}, finite_horizon=True))
    env.reset()
    for _ in range(env.N):
        _, _, term, trunc, _ = env.step([0.0])
    assert term and not trunc
    assert env.state[-1] == 0


@pytest.mark.parametrize("normalise_o", [True, False])
def test_batch_passes_gymnasium_checker(normalise_o):
    from gymnasium.utils.env_checker import check_env

    check_env(make_env(batch_params(normalise_o=normalise_o)), skip_render_check=True)


@pytest.mark.parametrize("horizon", [1, 4])
def test_batch_constraint_termination_takes_precedence_over_horizon(horizon):
    env = make_env(
        batch_params(
            N=horizon, constraints={"x": [0.21]}, cons_type={"x": ["<="]}, done_on_cons_vio=True, r_penalty=False
        )
    )
    env.reset()
    _, reward, term, trunc, _ = env.step([1.0])
    assert term and not trunc
    assert reward == (pytest.approx(env.state[0]) if horizon == 1 else 0.0)


def test_empty_reward_simulation_keeps_continuing_episode_layout():
    config = batch_params()
    config["reward_states"] = []
    env = make_env(config)
    assert env.observation_info() == [("x", "state")]
    env.reset()
    for _ in range(env.N):
        _, reward, term, trunc, _ = env.step([0.0])
    assert trunc and not term and reward == 0


def test_numpy_horizon_returns_python_boolean_flags():
    env = make_env(params(N=np.int64(2), tsim=np.float64(1)))
    env.reset()
    for _ in range(env.N):
        _, _, term, trunc, _ = env.step([0.0])
        assert type(term) is bool and type(trunc) is bool


@pytest.mark.parametrize("batch", [True, False])
def test_vectorized_autoreset_preserves_terminal_observation_and_bootstrap_flag(batch):
    vec = pytest.importorskip("stable_baselines3.common.vec_env").DummyVecEnv(
        [lambda: make_env(batch_params() if batch else params())]
    )
    try:
        initial = vec.reset().copy()
        for _ in range(4):
            obs, rewards, done, infos = vec.step(np.zeros((1, 1), dtype=np.float32))
        assert done[0]
        assert infos[0]["TimeLimit.truncated"] is (not batch)
        terminal = infos[0]["terminal_observation"]
        assert vec.observation_space.contains(terminal)
        if batch:
            assert terminal[-1] == -1 and obs[0, -1] == 1
            assert rewards[0] > 0
        np.testing.assert_array_equal(obs, initial)
    finally:
        vec.close()


def test_step_requires_an_active_episode_and_reset_clears_previous_action():
    env = make_env(params(N=1))
    with pytest.raises(gym.error.ResetNeeded):
        env.step([0.0])
    env.reset()
    env.step([0.0])
    with pytest.raises(gym.error.ResetNeeded):
        env.step([0.0])
    _, info = env.reset()
    assert "u" not in info
    env.step([0.0])


@pytest.mark.parametrize("normalise_o", [True, False])
@pytest.mark.parametrize("percentage", [1, np.float32(1), {"x": 1.0}])
def test_noise_is_seeded_at_reset_and_step_with_truthful_space(normalise_o, percentage):
    env = make_env(params(normalise_o=normalise_o, noise=True, noise_percentage=percentage))
    assert np.isneginf(env.observation_space.low[0])
    assert np.isposinf(env.observation_space.high[0])
    assert np.isfinite(env.observation_space.low[1])
    assert np.isfinite(env.observation_space_base.low).all()
    runs = []
    for seed in (123, 123, 124):
        obs, _ = env.reset(seed=seed)
        assert env.obs[0] != env.state[0]
        states, measurements = [env.state.copy()], [obs]
        for _ in range(env.N):
            obs, *_ = env.step([0.0])
            assert env.observation_space.contains(obs)
            states.append(env.state.copy())
            measurements.append(obs)
        runs.append((states, measurements))
    np.testing.assert_array_equal(runs[0][0], runs[2][0])
    np.testing.assert_array_equal(runs[0][1], runs[1][1])
    assert not np.array_equal(runs[0][1], runs[2][1])


def test_partial_observation_space_contains_masked_zero_in_physical_units():
    env = make_env({"model": "cstr", "N": 4, "tsim": 1, "normalise_o": False, "partial_observation": ["Ca"]})
    obs, info = env.reset()
    assert obs[1] == env.observation_space.low[1] == env.observation_space.high[1] == 0
    assert info["obs"][1] == 330
    assert env.observation_space.contains(obs)


@pytest.mark.parametrize("distribution", ["uniform", "normal"])
def test_x0_dictionary_uses_nonzero_index_and_seed(distribution):
    env = make_env(
        {
            "model": "cstr",
            "N": 4,
            "tsim": 1,
            "uncertainty_percentages": {"x0": {1: 0.01}},
            "distribution": distribution,
        }
    )
    rng = np.random.default_rng(7)
    expected = rng.normal(330, 3.3) if distribution == "normal" else 330 * (1 + rng.uniform(-0.01, 0.01))
    env.reset(seed=7)
    assert env.state[0] == 0.8
    assert env.state[1] == expected
    assert env.NUn == 0
    first = env.state.copy()
    env.reset(seed=7)
    np.testing.assert_array_equal(env.state, first)


def test_negative_nominal_normal_uncertainty_and_parameter_space():
    env = make_env(
        params(
            model_params={"K": -1.0},
            uncertainty_percentages={"K": 0.2},
            distribution="normal",
            uncertainty_bounds={"low": [-1.01], "high": [-0.99]},
        )
    )
    obs, _ = env.reset(seed=3)
    assert env.model.K == np.random.default_rng(3).normal(-1, 0.2)
    assert np.isneginf(env.observation_space.low[-1])
    assert np.isposinf(env.observation_space.high[-1])
    assert env.NUn == 1 and env.Nx == len(obs) == 3
    assert obs[-1] > 1  # normalization does not clip a sample outside the finite scaling bounds


@pytest.mark.parametrize("normalise_o", [True, False])
def test_seeded_noise_uncertainty_disturbance_and_time_coexist(normalise_o):
    config = {
        "model": "cstr",
        "N": 4,
        "tsim": 0.2,
        "normalise_o": normalise_o,
        "x0": [0.8, 330],
        "o_space": {"low": [0.5, 300], "high": [1.2, 360]},
        "reward_states": ["Ca"],
        "noise": True,
        "noise_percentage": {"T": 0.01},
        "uncertainty_percentages": {"x0": {0: 0.01}, "k0": 0.01},
        "uncertainty_bounds": {"low": [7e10], "high": [7.4e10]},
        "disturbances": {"Ti": [350, 349, 348, 347, 346]},
        "disturbance_bounds": {"low": [345], "high": [355]},
    }
    env = make_env(config)
    trajectories = []
    for seed in (7, 7, 8):
        obs, info = env.reset(seed=seed)
        snapshots = [obs.copy()]
        for t in range(1, env.N + 1):
            obs, *_, info = env.step([0.0])
            assert env.Nx == len(obs) == 5
            assert env.observation_space.contains(obs)
            assert env.state[2] == config["disturbances"]["Ti"][t]
            assert env.state[3] == env.model.k0
            assert env.state[4] == 1 - t / env.N
            snapshots.append(obs.copy())
        trajectories.append(snapshots)
    np.testing.assert_array_equal(trajectories[0], trajectories[1])
    assert not np.array_equal(trajectories[0], trajectories[2])
    np.testing.assert_array_equal(np.asarray(trajectories[0])[:, [2, 4]], np.asarray(trajectories[2])[:, [2, 4]])


def test_x0_percentage_sequence_and_zero_noise_are_supported():
    env = make_env(params(uncertainty_percentages={"x0": [0.2]}, noise=True, noise_percentage=0))
    obs, _ = env.reset(seed=7)
    assert env.state[0] == 0.1 * (1 + np.random.default_rng(7).uniform(-0.2, 0.2))
    np.testing.assert_array_equal(env.obs, env.state)
    assert np.isfinite(env.observation_space.low).all()
    assert env.observation_space.contains(obs)


def test_empirical_x0_and_parameters_are_seeded_and_in_layout():
    env = make_env(
        params(
            empirical_distribution={"x0": {0: [0.2, 0.3]}, "K": [0.8, 1.0]},
            uncertainty_bounds={"low": [0.8], "high": [1.0]},
        )
    )
    obs, _ = env.reset(seed=5)
    rng = np.random.default_rng(5)
    assert env.state[0] == rng.choice([0.2, 0.3])
    assert env.state[-1] == env.model.K == rng.choice([0.8, 1.0])
    assert env.observation_info()[-1] == ("K", "uncertainty")
    assert env.NUn == 1 and env.Nx == len(obs)
    repeated, _ = env.reset(seed=5)
    np.testing.assert_array_equal(obs, repeated)


@pytest.mark.parametrize(
    "extra, message",
    [
        ({"N": 0}, "N"),
        ({"N": -2}, "N"),
        ({"N": 2.5}, "N"),
        ({"N": True}, "N"),
        ({"tsim": 0}, "tsim"),
        ({"tsim": np.inf}, "tsim"),
        ({"tsim": np.nan}, "tsim"),
        ({"x0": [np.nan, 0.2]}, "x0"),
        ({"integration_method": "typo"}, "integration_method"),
        ({"a_space": {"low": [0, 0], "high": [1, 1]}}, "inputs"),
        ({"a_space": {"low": [1], "high": [0]}}, "a_space"),
        ({"a_space": {"low": [0], "high": [np.inf]}}, "a_space"),
        ({"o_space": {"low": [0, 0], "high": [0, 1]}}, "o_space"),
        ({"o_space": {"low": [0, 0], "high": [np.inf, 1]}}, "o_space"),
        ({"o_space": {"low": [[0, 0]], "high": [[1, 1]]}}, "o_space"),
        ({"SP": {"typo": [0.2]}}, "SP"),
        ({"SP": 42}, "SP"),
        ({"o_space": {"low": [], "high": []}}, "o_space"),
        ({"SP": {"x": []}}, "SP"),
        ({"SP": {"x": [np.nan]}}, "SP"),
        ({"SP": {"x": [[0.2]]}}, "SP"),
        ({"partial_observation": ["typo"]}, "partial_observation"),
        ({"r_scale": {"typo": 1.0}}, "r_scale"),
        ({"noise": True}, "noise_percentage"),
        ({"noise": True, "noise_percentage": -0.1}, "noise_percentage"),
        ({"noise": True, "noise_percentage": {"typo": 0.1}}, "noise_percentage"),
        ({"noise": True, "noise_percentage": np.nan}, "noise_percentage"),
        ({"uncertainty_percentages": {"typo": 0.1}}, "uncertainty"),
        ({"uncertainty_percentages": {"x0": {1: 0.1}}}, "x0"),
        ({"uncertainty_percentages": {"x0": {0: -0.1}}}, "uncertainty"),
        ({"uncertainty_percentages": {"x0": {0: 0.1}}, "distribution": "typo"}, "distribution"),
        ({"uncertainty_percentages": {"K": 0.1}, "empirical_distribution": {"K": [1]}}, "one"),
        ({"uncertainty_percentages": {"K": 0.1}, "uncertainty_bounds": {"low": [0.9], "high": [1.0]}}, "support"),
        ({"empirical_distribution": {"K": []}}, "empirical_distribution"),
        ({"empirical_distribution": {"K": [np.nan]}}, "empirical_distribution"),
        ({"empirical_distribution": {"x0": {0: [[0.2]]}}}, "empirical_distribution"),
        ({"disturbances": {"typo": [1]}}, "disturbances"),
        ({"custom_reward": 42}, "custom_reward"),
        ({"finite_horizon": "yes"}, "finite_horizon"),
        ({"normalise_a": "yes"}, "normalise_a"),
        ({"uncertainty_percentages": {"x0": {0: 20.0}}}, "support"),
        ({"uncertainty_percentages": {"x0": {-1: 0.1}}}, "x0"),
        ({"uncertainty_percentages": {"x0": {0.5: 0.1}}}, "x0"),
        ({"uncertainty_percentages": {"K": np.inf}}, "uncertainty"),
        ({"uncertainty_percentages": {"K": 0.1}, "uncertainty_bounds": {"low": [], "high": []}}, "uncertainty_bounds"),
        ({"constraints": {"typo": [1.0]}}, "constraints"),
        ({"constraints": {"x": [1.0]}, "cons_type": {"x": ["=>"]}}, "cons_type"),
    ],
)
def test_invalid_configuration_fails_at_construction(extra, message):
    with pytest.raises(ValueError, match=message):
        make_env(params(**extra))


@pytest.mark.parametrize("action", [[], [0, 0], [[0]], [np.nan], [np.inf], [1.1], 0.0])
def test_invalid_action_does_not_advance_episode(action):
    env = make_env(params())
    env.reset()
    initial = env.state.copy()
    with pytest.raises(ValueError, match="action"):
        env.step(action)
    np.testing.assert_array_equal(env.state, initial)
    assert env.t == 0 and "u" not in env.info


@pytest.mark.parametrize("failure", ["nonfinite", "wrong_shape", "exception"])
@pytest.mark.parametrize("backend", ["casadi", "jax"])
def test_solver_failure_preserves_episode_and_delta_input(failure, backend):
    env = make_env(
        params(
            a_delta=True, a_0=[0.4], a_space={"low": [-0.1], "high": [0.1]}, a_space_act={"low": [0.0], "high": [1.0]}
        )
    )
    env.reset(seed=0)
    state, obs, info, action = env.state.copy(), env.obs.copy(), copy.deepcopy(env.info), env.a_save.copy()

    def bad_step(x, u):
        x[:] = 123  # even a faulty integration adapter must not mutate the live state
        if failure == "exception":
            raise RuntimeError("test solver failure")
        values = np.array([np.nan] if failure == "nonfinite" else [1, 2])
        return {"xf": DM(values)} if backend == "casadi" else values

    # Exercise the JAX adapter contract without requiring the optional solver dependency.
    env.integration_method = backend
    env.int_eng = SimpleNamespace(casadi_step=bad_step, jax_step=bad_step)
    with pytest.raises(RuntimeError, match="integration.*step 0") as exc:
        env.step([1.0])
    assert exc.value.__cause__ is not None
    np.testing.assert_array_equal(env.state, state)
    np.testing.assert_array_equal(env.obs, obs)
    np.testing.assert_array_equal(env.a_save, action)
    np.testing.assert_array_equal(env.info["obs"], info["obs"])
    assert env.t == 0 and not env.done and "u" not in env.info


@pytest.mark.parametrize("endpoint", [0.1, 0.3])
def test_float32_physical_action_endpoints_are_accepted(endpoint):
    env = make_env(params(normalise_a=False, a_space={"low": [0.1], "high": [0.3]}))
    env.reset()
    _, _, _, _, info = env.step(np.array([endpoint], dtype=np.float32))
    assert info["u"][0] == pytest.approx(endpoint, rel=0, abs=2e-9)


@pytest.mark.parametrize("normalise_o", [True, False])
def test_disturbance_bounds_have_one_entry_per_active_schedule(normalise_o):
    with pytest.raises(ValueError, match="disturbance_bounds"):
        make_env(
            {
                "model": "cstr",
                "N": 4,
                "tsim": 1,
                "normalise_o": normalise_o,
                "disturbances": {"Ti": [350]},
                "disturbance_bounds": {"low": [], "high": []},
            }
        )
