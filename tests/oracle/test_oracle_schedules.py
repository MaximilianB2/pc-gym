"""Check horizon forecasts and oracle trajectories against the environment's state-time convention."""

from dataclasses import dataclass

import numpy as np
import pytest

from pcgym import make_env
from pcgym.oracle import oracle

pytest.importorskip("do_mpc", reason="the oracle needs pcgym[oracle]")
pytestmark = pytest.mark.slow


def _params(length=5, disturbance_mode="both"):
    params = {
        "model": "cstr",
        "N": 4,
        "tsim": 1,
        # Reversed relative to model state order, so forecasts must respect SP insertion order.
        "SP": {"T": [330, 331, 332, 329, 328][:length], "Ca": [0.85, 0.86, 0.87, 0.84, 0.83][:length]},
        "x0": np.array([0.85, 330.0, 330.0, 0.85]),
        "o_space": {"low": np.array([0.7, 300, 300, 0.7]), "high": np.array([1.0, 350, 350, 1.0])},
        "normalise_a": False,
        "normalise_o": False,
    }
    if disturbance_mode != "none":
        disturbances = {"Caf": [1.0, 1.05, 0.95, 1.02, 0.98][:length]}
        if disturbance_mode == "both":
            disturbances["Ti"] = [350, 345, 355, 348, 352][:length]
        params.update(
            disturbances=disturbances,
            disturbance_bounds={
                "low": np.array([320.0, 0.9] if disturbance_mode == "both" else [0.9]),
                "high": np.array([360.0, 1.1] if disturbance_mode == "both" else [1.1]),
            },
        )
    return params


@pytest.mark.parametrize("length", [1, 4, 5])
@pytest.mark.parametrize("disturbance_mode", ["none", "partial", "both"])
def test_forecasts_vary_by_stage_and_hold_the_final_value(length, disturbance_mode):
    env = make_env(_params(length, disturbance_mode))
    try:
        controller = oracle(env, env.env_params, MPC_params={"N": 3})
        mpc, simulator = controller.setup_mpc()
        for t in (0, 1, env.N - 1, env.N):
            # The small roundoff represents do-mpc's accumulation of floating simulation times.
            t_now = t * env.dt + 1e-14
            forecast = mpc.tvp_fun(t_now)
            for stage in range(controller.N + 1):
                idx = min(t + stage, length - 1)
                expected_sp = [env.SP[name][idx] for name in ("T", "Ca")]
                np.testing.assert_allclose(np.asarray(forecast["_tvp", stage, "SP"]).ravel(), expected_sp)
                if disturbance_mode != "none":
                    expected_d = [
                        env.disturbances[name][idx] if name in env.disturbances else getattr(env.model, name)
                        for name in ("Ti", "Caf")
                    ]
                    np.testing.assert_allclose(np.asarray(forecast["_tvp", stage, "d"]).ravel(), expected_d)
            current = simulator.tvp_fun(t_now)
            np.testing.assert_array_equal(current["SP"], forecast["_tvp", 0, "SP"])
            if disturbance_mode != "none":
                np.testing.assert_array_equal(current["d"], forecast["_tvp", 0, "d"])
    finally:
        env.close()


@dataclass
class ScalarIntegrator:
    int_method: str = "casadi"
    d: float = 0.0

    def __call__(self, x, u):
        return np.array([u[0] + u[1]])

    def info(self):
        return {"states": ["x"], "inputs": ["u"], "disturbances": ["d"], "parameters": {"d": self.d}}


def test_mpc_tracks_changing_targets_at_each_predicted_state():
    targets = np.array([0.0, 0.25, 0.0, -0.25, 0.5])
    disturbances = np.array([0.2, -0.2, 0.3, 0.1, 0.0])
    env = make_env(
        {
            "custom_model": ScalarIntegrator(),
            "N": 4,
            "tsim": 1,
            "SP": {"x": targets},
            "x0": np.array([0.0, 0.0]),
            "a_space": {"low": np.array([-5.0]), "high": np.array([5.0])},
            "o_space": {"low": np.array([-2.0, -2.0]), "high": np.array([2.0, 2.0])},
            "disturbances": {"d": disturbances},
            "disturbance_bounds": {"low": np.array([-1.0]), "high": np.array([1.0])},
            "normalise_a": False,
            "normalise_o": False,
        }
    )
    try:
        controller = oracle(env, env.env_params, MPC_params={"N": 2})
        mpc, simulator = controller.setup_mpc()
        mpc.x0 = np.array([0.0])
        mpc.set_initial_guess()
        mpc.make_step(np.array([0.0]))
        # With no input penalty and feasible targets, each predicted state should hit its own target.
        predicted_states = mpc.data.prediction(("_x", "x"))[0, :, 0]
        np.testing.assert_allclose(predicted_states, targets[:3], atol=2e-5)

        controller._mpc, controller._simulator = mpc, simulator
        states, inputs = controller.mpc()
        np.testing.assert_allclose(states[0], targets, atol=2e-5)
        np.testing.assert_allclose(inputs[0], np.diff(targets) / env.dt - disturbances[:-1], atol=1e-4)
        np.testing.assert_array_equal(inputs[1], disturbances[:-1])
    finally:
        env.close()


def test_oracle_trajectory_replays_in_environment_with_changing_disturbances():
    env = make_env(_params())
    try:
        controller = oracle(env, env.env_params, MPC_params={"N": 2})
        states, inputs = controller.mpc()
        env.reset(seed=0)
        np.testing.assert_allclose(env.state[:2], states[:, 0])
        np.testing.assert_array_equal(inputs[1], env.disturbances["Ti"][: env.N])
        np.testing.assert_array_equal(inputs[2], env.disturbances["Caf"][: env.N])
        assert np.all(inputs[0] >= env.env_params["a_space"]["low"][0])
        assert np.all(inputs[0] <= env.env_params["a_space"]["high"][0])
        for t in range(env.N):
            env.step(inputs[:1, t])
            # The two CVODES instances use different tolerances. Allow 2e-5 concentration units
            # and 0.002 K over this one-minute replay, checked against an independent SciPy solve.
            assert env.state[0] == pytest.approx(states[0, t + 1], rel=0, abs=2e-5)
            assert env.state[1] == pytest.approx(states[1, t + 1], rel=0, abs=2e-3)
    finally:
        env.close()
