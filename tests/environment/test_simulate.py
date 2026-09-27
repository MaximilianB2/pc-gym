import numpy as np
import pytest

from pcgym import make_env

N = 8


def _params(**extra):
    params = {
        "model": "cstr",
        "N": N,
        "tsim": 2,
        "SP": {"Ca": [0.85] * N},
        "a_space": {"low": np.array([295]), "high": np.array([302])},
        "o_space": {"low": np.array([0.7, 300, 0.8]), "high": np.array([1, 350, 0.9])},
        "x0": np.array([0.8, 330, 0.8]),
        "normalise_a": False,
        "integration_method": "casadi",
    }
    params.update(extra)
    return params


U = np.linspace(296, 301, N).reshape(-1, 1)


def _stepped(env):
    env.reset()
    traj = [env.state[:2].copy()]
    for u in U:
        env.step(u)
        traj.append(env.state[:2].copy())
    return np.array(traj)


def test_simulate_matches_step():
    env = make_env(_params())
    expected = _stepped(env)
    traj = make_env(_params()).simulate([0.8, 330], U)
    assert traj.shape == (N + 1, 2)
    np.testing.assert_allclose(traj, expected, rtol=1e-8)


def test_simulate_leaves_episode_untouched():
    env = make_env(_params())
    env.reset()
    env.step(U[0])
    t, state, info_obs = env.t, env.state.copy(), env.info["obs"].copy()
    env.simulate(env.state, U)
    assert env.t == t
    np.testing.assert_array_equal(env.state, state)
    np.testing.assert_array_equal(env.info["obs"], info_obs)


def test_simulate_param_override_is_temporary():
    env = make_env(_params())
    k0 = env.model.k0
    nominal = env.simulate([0.8, 330], U)
    perturbed = env.simulate([0.8, 330], U, params={"k0": k0 * 2})
    assert not np.allclose(nominal, perturbed)
    assert env.model.k0 == k0
    np.testing.assert_allclose(env.simulate([0.8, 330], U), nominal)


def test_simulate_rejects_unknown_params_and_bad_inputs():
    env = make_env(_params())
    with pytest.raises(ValueError, match="Unknown model parameter"):
        env.simulate([0.8, 330], U, params={"not_a_param": 1})
    with pytest.raises(ValueError, match="columns"):
        env.simulate([0.8, 330], np.ones((N, 2)))


def test_simulate_fills_nominal_disturbances():
    params = _params(
        disturbances={"Ti": np.full(N, 350.0)},
        disturbance_bounds={"low": np.array([320]), "high": np.array([360])},
    )
    env = make_env(params)
    controls_only = env.simulate([0.8, 330], U)
    full = env.simulate([0.8, 330], np.hstack([U, np.tile([350.0, 1.0], (N, 1))]))
    np.testing.assert_allclose(controls_only, full)


@pytest.mark.slow
def test_simulate_jax_matches_casadi():
    pytest.importorskip("diffrax")
    casadi = make_env(_params()).simulate([0.8, 330], U)
    jax = make_env(_params(integration_method="jax")).simulate([0.8, 330], U)
    np.testing.assert_allclose(jax, casadi, rtol=1e-5)
