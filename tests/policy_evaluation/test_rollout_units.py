import numpy as np
import pytest

from pcgym import make_env
from pcgym.policy_evaluation import policy_eval

N = 8


class ConstantPolicy:
    def __init__(self, a):
        self.a = np.array(a, dtype=float)

    def predict(self, obs, deterministic=True):
        return self.a, None


def _params(normalise_o, normalise_a, **extra):
    params = {
        "model": "cstr",
        "N": N,
        "tsim": 2,
        "SP": {"Ca": [0.85] * N},
        "a_space": {"low": np.array([295]), "high": np.array([302])},
        "o_space": {"low": np.array([0.7, 300, 0.8]), "high": np.array([1, 350, 0.9])},
        "x0": np.array([0.8, 330, 0.8]),
        "normalise_o": normalise_o,
        "normalise_a": normalise_a,
        "integration_method": "casadi",
    }
    params.update(extra)
    return params


def _raw_trajectory(params, action):
    env = make_env(params)
    env.reset()
    states = [env.state.copy()]
    for _ in range(N):
        env.step(action)
        states.append(env.state.copy())
    return np.array(states).T


@pytest.mark.parametrize("normalise_o", [True, False])
@pytest.mark.parametrize("normalise_a", [True, False])
def test_rollout_states_and_actions_are_in_physical_units(normalise_o, normalise_a):
    params = _params(normalise_o, normalise_a)
    action = np.array([0.0]) if normalise_a else np.array([298.5])
    pe = policy_eval(make_env, {"pi": ConstantPolicy(action)}, 1, params)

    _, states, actions, _ = pe.rollout(pe.policies["pi"])

    np.testing.assert_allclose(states, _raw_trajectory(params, action), rtol=1e-6)
    np.testing.assert_allclose(actions, 298.5)


def test_rollout_records_full_state_under_partial_observation():
    params = _params(True, True, partial_observation=["Ca"])
    pe = policy_eval(make_env, {"pi": ConstantPolicy([0.0])}, 1, params)
    _, states, _, _ = pe.rollout(pe.policies["pi"])
    np.testing.assert_allclose(states, _raw_trajectory(params, np.array([0.0])), rtol=1e-6)


def test_delta_actions_are_denormalised_once_and_clipped():
    params = _params(
        True,
        True,
        a_space={"low": np.array([-2]), "high": np.array([2])},
        a_delta=True,
        a_0=np.array([300.0]),
        a_space_act={"low": np.array([295]), "high": np.array([302])},
    )
    env = make_env(params)
    env.reset()
    env.step(np.array([0.5]))  # delta = +1 K
    np.testing.assert_allclose(env.info["u"], 301.0)
    env.step(np.array([1.0]))  # delta = +2 K, clipped to the 302 K limit
    np.testing.assert_allclose(env.info["u"], 302.0)
