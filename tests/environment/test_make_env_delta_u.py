import numpy as np

from pcgym import make_env


def test_make_env_delta_u():
    env_params = {
        "model": "first_order_system",
        "a_space": {"low": np.array([-1]), "high": np.array([1])},
        "o_space": {"low": np.array([-1, -1]), "high": np.array([1, 1])},
        "SP": {"x": [0.5] * 100},
        "N": 100,
        "tsim": 10,
        "x0": np.array([0.2, 0.5]),
        "a_delta": True,
        "a_0": np.array([0]),
        "a_space_act": {"low": np.array([-1]), "high": np.array([1])},
    }
    env = make_env(env_params)
    assert env.a_delta
    assert np.all(env.a_0 == np.array([0]))

    env.reset()
    action1 = np.array([0.5])
    obs1, _, _, _, _ = env.step(action1)
    action2 = np.array([-0.3])
    obs2, _, _, _, _ = env.step(action2)

    # Check if the action is cumulative
    assert np.isclose(env.a_save, np.array([0.2]))

    # Check if the action is clipped to a_space_act
    action3 = np.array([1.0])
    obs3, _, _, _, _ = env.step(action3)
    np.testing.assert_allclose(env.a_save, [1.0])
    assert np.all(env.a_save <= env.env_params["a_space_act"]["high"])
    assert np.all(env.a_save >= env.env_params["a_space_act"]["low"])
