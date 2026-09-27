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
        "normalise_a": True,
        "normalise_o": False,
        "integration_method": "casadi",
    }


def _rollout(env, n=5):
    env.reset()
    for _ in range(n):
        env.step(np.array([0.0]))
    return env.state[: env.Nx_oracle].copy()


def test_integrator_uses_owning_env_model(env_params):
    env = make_env(env_params)
    assert _rollout(env) is not None
    env.reset()
    assert env.int_eng.env is env


def test_model_param_change_after_make_env_affects_dynamics(env_params):
    nominal = _rollout(make_env(env_params))

    env = make_env(env_params)
    env.model.k0 = env.model.k0 * 10
    assert not np.allclose(_rollout(env), nominal)


def test_model_param_change_between_episodes_affects_dynamics(env_params):
    env = make_env(env_params)
    nominal = _rollout(env)
    env.model.k0 = env.model.k0 * 10
    assert not np.allclose(_rollout(env), nominal)


def test_discretised_plant_is_reused_across_steps(env_params):
    env = make_env(env_params)
    env.reset()
    plant = env.int_eng.discretised_plant
    for _ in range(3):
        env.step(np.array([0.0]))
    assert env.int_eng.discretised_plant is plant


def test_parametric_uncertainty_affects_dynamics(env_params):
    env_params["uncertainty_percentages"] = {"k0": 0.5}
    env_params["distribution"] = "uniform"
    env_params["uncertainty_bounds"] = {"low": np.array([1e9]), "high": np.array([1e11])}
    env = make_env(env_params)

    finals = []
    for _ in range(3):
        env.reset()
        for _ in range(5):
            env.step(np.array([0.0]))
        finals.append(env.state[0])
    assert np.ptp(finals) > 0


@pytest.mark.parametrize(
    "model_name, u",
    [
        ("cstr", np.array([300.0, 355.0, 1.1])),
        ("complex_cstr", np.array([300.0, 355.0, 1.1])),
        ("multistage_extraction", np.array([5.0, 500.0, 0.9, 0.1])),
    ],
)
def test_model_call_with_disturbances_does_not_mutate_model(model_name, u):
    from pcgym.model_classes import complex_cstr, cstr, multistage_extraction

    cls = {"cstr": cstr, "complex_cstr": complex_cstr, "multistage_extraction": multistage_extraction}[model_name]
    model = cls(int_method="casadi")
    before = dict(model.info()["parameters"])
    model(np.ones(len(model.info()["states"])), u)
    assert model.info()["parameters"] == before
