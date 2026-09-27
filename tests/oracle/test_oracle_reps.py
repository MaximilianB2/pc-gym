import numpy as np
import pytest

from pcgym import make_env
from pcgym.oracle import oracle

N = 10


def _params(**extra):
    params = {
        "model": "cstr",
        "N": N,
        "tsim": 2.5,
        "SP": {"Ca": [0.85] * 5 + [0.88] * 5},
        "a_space": {"low": np.array([295]), "high": np.array([305])},
        "o_space": {"low": np.array([0.7, 300, 0.8]), "high": np.array([1, 350, 0.9])},
        "x0": np.array([0.85, 330, 0.8]),
        "integration_method": "casadi",
    }
    params.update(extra)
    return params


class ZeroPolicy:
    def predict(self, obs, deterministic=True):
        return np.array([0.0]), None


@pytest.mark.slow
@pytest.mark.parametrize(
    "extra",
    [
        {},
        {
            "disturbances": {"Ti": np.repeat([350, 345], [5, 5])},
            "disturbance_bounds": {"low": np.array([320]), "high": np.array([350])},
        },
    ],
    ids=["plain", "disturbances"],
)
def test_oracle_mpc_can_run_repeatedly(extra):
    params = _params(**extra)
    o = oracle(make_env(params), params, MPC_params={"N": 2})
    x1, u1 = o.mpc()
    x2, u2 = o.mpc()
    np.testing.assert_allclose(x1, x2)
    np.testing.assert_allclose(u1, u2)


@pytest.mark.slow
def test_get_rollouts_with_oracle_and_multiple_reps():
    env = make_env(_params())
    _, data = env.get_rollouts({"zero": ZeroPolicy()}, reps=5, oracle=True, MPC_params={"N": 2})
    assert data["oracle"]["x"].shape == (env.Nx_oracle, N + 1, 5)
    assert np.all(data["oracle"]["x"] == data["oracle"]["x"][:, :, :1])


def test_oracle_does_not_mutate_callers_env_params():
    params = _params(integration_method="jax")
    o = oracle(make_env(_params()), params)
    assert params["integration_method"] == "jax"
    assert o.env_params["integration_method"] == "casadi"
