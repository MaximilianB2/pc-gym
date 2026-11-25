"""Schedules use the observation's state index, with disturbances held over each interval."""

from dataclasses import dataclass

import numpy as np
import pytest

from pcgym import make_env


@dataclass
class DrivenIntegrator:
    """Constant-rate dynamics give an independent check of the applied disturbance index."""

    int_method: str = "casadi"
    d1: float = 0.1
    d2: float = 0.2

    def __call__(self, x, u):
        return np.array([u[0] + u[1], 2 * u[2]])

    def info(self):
        return {
            "states": ["x", "y"],
            "inputs": ["u"],
            "disturbances": ["d1", "d2"],
            "parameters": {"d1": self.d1, "d2": self.d2},
        }


def _physical(env, obs):
    if env.normalise_o:
        low, high = env.observation_space_base.low, env.observation_space_base.high
        return (obs + 1) * (high - low) / 2 + low
    return obs


@pytest.mark.parametrize("normalise_o", [True, False])
@pytest.mark.parametrize("length", [1, 4, 5], ids=["constant", "N_values", "N_plus_1_values"])
def test_setpoint_observation_and_reward_use_returned_state_time(normalise_o, length):
    schedule = [0.2, 0.4, 0.6, 0.3, 0.7][:length]
    params = {
        "model": "first_order_system",
        "N": 4,
        "tsim": 1,
        "x0": np.array([0.1, 0.0]),  # the schedule overrides this setpoint placeholder
        "SP": {"x": schedule},
        "r_scale": {"x": 3.0},
        "normalise_o": normalise_o,
    }
    env = make_env(params)
    try:
        for _ in range(2):
            obs, info = env.reset(seed=0)
            assert env.t == 0
            assert env.state[-1] == schedule[0]
            assert _physical(env, obs)[-1] == pytest.approx(schedule[0], abs=1e-7)
            assert _physical(env, info["obs"])[-1] == pytest.approx(schedule[0])
            for t in range(1, env.N + 1):
                obs, reward, terminated, truncated, info = env.step(np.zeros(1, dtype=np.float32))
                target = schedule[min(t, length - 1)]
                assert env.t == t
                assert env.state[-1] == target
                assert _physical(env, obs)[-1] == pytest.approx(target, abs=1e-7)
                assert _physical(env, info["obs"])[-1] == pytest.approx(target)
                assert reward == pytest.approx(-3 * (env.state[0] - target) ** 2)
                assert terminated is False
                assert truncated is (t == env.N)
        np.testing.assert_array_equal(params["x0"], [0.1, 0.0])
        np.testing.assert_array_equal(env.env_params["x0"], params["x0"])
    finally:
        env.close()


def _driven_params(length=5, setpoints=True, partial_disturbances=False, **extra):
    # Deliberately reverse dictionary order relative to the model's states/disturbances.
    disturbances = {"d2": [0.3, 0.6, -0.1, 0.0, 0.4][:length]}
    if not partial_disturbances:
        disturbances["d1"] = [0.1, 0.2, 0.4, -0.2, 1.0][:length]
    n_obs = 4 if setpoints else 2
    params = {
        "custom_model": DrivenIntegrator(),
        "N": 4,
        "tsim": 1,
        "x0": np.array([0.2, 0.4, -1.0, -2.0][:n_obs]),
        "a_space": {"low": np.array([-1.0]), "high": np.array([1.0])},
        "o_space": {"low": np.full(n_obs, -5.0), "high": np.full(n_obs, 5.0)},
        "normalise_a": False,
        "disturbances": disturbances,
        "disturbance_bounds": {"low": np.full(len(disturbances), -1.0), "high": np.ones(len(disturbances))},
        "r_scale": {"x": 2.0, "y": 3.0},
    }
    if setpoints:
        params["SP"] = {"y": [0.3, 0.2, 0.1, 0.4, 0.5][:length], "x": [0.1, 0.4, 0.2, 0.3, 0.6][:length]}
    else:
        params["reward_states"] = ["x"]
    params.update(extra)
    return params


@pytest.mark.parametrize("normalise_o", [True, False])
@pytest.mark.parametrize("length", [1, 4, 5])
@pytest.mark.parametrize("setpoints", [True, False])
@pytest.mark.parametrize("partial_disturbances", [True, False])
def test_disturbance_drives_current_interval_and_observation_shows_next(
    normalise_o, length, setpoints, partial_disturbances
):
    params = _driven_params(length, setpoints, partial_disturbances, normalise_o=normalise_o)
    env = make_env(params)
    try:
        env.reset(seed=0)
        expected_state = np.array([0.2, 0.4])
        for t, u in enumerate([0.2, -0.4, 0.0, 0.6]):
            current = min(t, length - 1)
            following = min(t + 1, length - 1)
            d1 = env.model.d1 if partial_disturbances else params["disturbances"]["d1"][current]
            d2 = params["disturbances"]["d2"][current]
            expected_state += env.dt * np.array([u + d1, 2 * d2])
            obs, reward, terminated, truncated, info = env.step(np.array([u]))
            np.testing.assert_allclose(env.state[:2], expected_state, atol=1e-10)
            np.testing.assert_allclose(_physical(env, obs)[:2], expected_state, atol=5e-7)
            assert env.observation_space.contains(obs)
            assert terminated is (not setpoints and t + 1 == env.N)
            assert truncated is (setpoints and t + 1 == env.N)

            # State entries follow the documented layout; inactive disturbances use model defaults.
            expected_slots = []
            if setpoints:
                expected_slots.extend(params["SP"][name][following] for name in ("y", "x"))
                expected_reward = sum(
                    -params["r_scale"][name] * (expected_state[i] - params["SP"][name][following]) ** 2
                    for i, name in enumerate(("x", "y"))
                )
                assert reward == pytest.approx(expected_reward)
            expected_slots.extend(
                params["disturbances"][name][following] for name in ("d1", "d2") if name in params["disturbances"]
            )
            if not setpoints:
                expected_slots.append(1 - (t + 1) / env.N)
            np.testing.assert_array_equal(env.state[2:], expected_slots)
            np.testing.assert_allclose(_physical(env, info["obs"]), env.state, atol=1e-12)
    finally:
        env.close()


def test_custom_reward_sees_next_schedule_and_the_disturbance_actually_applied():
    def reward(env, obs, uk, violated):
        env.info["reward_inputs"] = {"t": env.t, "obs": obs.copy(), "uk": uk.copy()}
        return float(obs[0] - obs[3] - uk[1])

    env = make_env(_driven_params(custom_reward=reward, normalise_o=False))
    try:
        env.reset(seed=0)
        obs, value, *_, info = env.step(np.array([0.0]))
        inputs = info["reward_inputs"]
        assert inputs["t"] == 1
        np.testing.assert_allclose(inputs["uk"], [0.0, 0.1, 0.3])
        np.testing.assert_allclose(inputs["obs"][2:], [0.2, 0.4, 0.2, 0.6])
        np.testing.assert_allclose(obs, inputs["obs"], atol=1e-7)
        assert value == pytest.approx(0.225 - 0.4 - 0.1)
    finally:
        env.close()
