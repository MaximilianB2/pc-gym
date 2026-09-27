import numpy as np
import pytest

from pcgym import make_env

N = 10


def _params(**extra):
    params = {
        "model": "cstr",
        "N": N,
        "tsim": 2,
        "SP": {"Ca": [0.85] * N},
        "a_space": {"low": np.array([295]), "high": np.array([302])},
        "o_space": {"low": np.array([0.7, 300, 0.8]), "high": np.array([1, 350, 0.9])},
        "x0": np.array([0.8, 330, 0.8]),
        "normalise_o": False,
        "integration_method": "casadi",
    }
    params.update(extra)
    return params


DIST = {
    "disturbances": {"Ti": np.linspace(350, 340, N)},
    "disturbance_bounds": {"low": np.array([320]), "high": np.array([360])},
}
UNC = {
    "uncertainty_percentages": {"k0": 0.1},
    "distribution": "uniform",
    "uncertainty_bounds": {"low": np.array([1e10]), "high": np.array([1e11])},
}


def test_observation_info_basic():
    env = make_env(_params())
    assert env.observation_info() == [("Ca", "state"), ("T", "state"), ("Ca_SP", "setpoint")]


def test_observation_info_matches_obs_length_with_all_parts():
    env = make_env(_params(**DIST, **UNC))
    info = env.observation_info()
    assert info == [
        ("Ca", "state"),
        ("T", "state"),
        ("Ca_SP", "setpoint"),
        ("Ti", "disturbance"),
        ("k0", "uncertainty"),
    ]
    obs, _ = env.reset(seed=0)
    assert obs.shape == (len(info),) == env.observation_space.shape


def test_disturbance_and_uncertainty_slots_are_not_overwritten():
    env = make_env(_params(**DIST, **UNC))
    obs, _ = env.reset(seed=0)
    k0 = obs[4]
    assert k0 == env.model.k0
    for t in range(3):
        obs, *_ = env.step(np.array([298.5]))
        assert obs[4] == k0  # uncertain parameter stays fixed during the episode
        assert obs[3] == pytest.approx(env.disturbances["Ti"][t + 1])  # disturbance slot updated


def test_disturbances_without_setpoints():
    params = _params(**DIST, reward_states=["Ca"], maximise_reward=True)
    del params["SP"]
    params["x0"] = np.array([0.8, 330])
    params["o_space"] = {"low": np.array([0.7, 300]), "high": np.array([1, 350])}
    env = make_env(params)
    env.reset()
    obs, *_ = env.step(np.array([298.5]))
    assert obs[2] == pytest.approx(env.disturbances["Ti"][1])


def test_build_obs_from_named_parts_matches_reset():
    env = make_env(_params(**DIST))
    obs, _ = env.reset()
    built = env.build_obs(states={"T": 330, "Ca": 0.8}, setpoints={"Ca": 0.8}, disturbances={"Ti": 350})
    np.testing.assert_allclose(built, obs)


def test_build_obs_defaults_and_x0_prefix():
    env = make_env(_params())
    built = env.build_obs(states=[0.8, 330])
    np.testing.assert_allclose(built, [0.8, 330, 0.85])  # setpoint defaults to SP[0]


def test_build_obs_normalise_matches_env():
    params = _params(normalise_o=True)
    env = make_env(params)
    obs, _ = env.reset()
    np.testing.assert_allclose(env.build_obs(states=[0.8, 330], setpoints=[0.8], normalise=True), obs)


def test_build_obs_rejects_missing_or_unknown_names():
    env = make_env(_params())
    with pytest.raises(ValueError, match="missing \['T'\]"):
        env.build_obs(states={"Ca": 0.8})
    with pytest.raises(ValueError, match="unknown \['Cb'\]"):
        env.build_obs(states={"Ca": 0.8, "T": 330, "Cb": 1})
    with pytest.raises(ValueError, match="expected 2 state values"):
        env.build_obs(states=[0.8])
