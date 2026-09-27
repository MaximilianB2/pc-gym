import numpy as np

from pcgym import make_env


def test_make_env_constraints():
    env_params = {
        "model": "cstr",
        "a_space": {"low": np.array([295]), "high": np.array([302])},
        "o_space": {"low": np.array([0.7, 300, 0.8]), "high": np.array([1, 350, 0.9])},
        "SP": {"Ca": [0.85] * 100},
        "N": 100,
        "tsim": 10,
        "x0": np.array([0.8, 330, 0.8]),
        "constraints": lambda x, u: np.array([329 - x[1], x[1] - 331]).reshape(-1),
        "done_on_cons_vio": True,
        "r_penalty": True,
    }
    env = make_env(env_params)
    assert env.constraint_active
    assert env.done_on_constraint
    assert env.r_penalty

    env.reset()
    action = np.array([-1.0])  # minimum coolant temperature cools T from 330 K to ~328.4 K, below 329 K
    _, reward, done, _, info = env.step(action)
    assert done
    assert reward < 0
    assert "cons_info" in info


def _violation_flags(normalise_o, normalise_a):
    env = make_env(
        {
            "model": "cstr",
            "a_space": {"low": np.array([295]), "high": np.array([302])},
            "o_space": {"low": np.array([0.7, 300, 0.8]), "high": np.array([1, 350, 0.9])},
            "SP": {"Ca": [0.85] * 10},
            "N": 10,
            "tsim": 1,
            "x0": np.array([0.8, 330, 0.8]),
            "constraints": {"T": [329, 331]},
            "cons_type": {"T": [">=", "<="]},
            "done_on_cons_vio": False,
            "r_penalty": False,
            "normalise_o": normalise_o,
            "normalise_a": normalise_a,
        }
    )
    env.reset(seed=0)
    a = np.array([-1.0]) if normalise_a else np.array([295.0])
    flags = []
    for _ in range(3):
        env.step(a)
        flags.append(bool(np.any(env.info["cons_info"][:, env.t, 0] > 0)))
    return flags, env.info["cons_info"][:, 1, 0].copy()


def test_constraint_check_uses_physical_units_regardless_of_normalisation():
    results = [_violation_flags(o, a) for o in (True, False) for a in (True, False)]
    for flags, g in results[1:]:
        assert flags == results[0][0]
        np.testing.assert_allclose(g, results[0][1])
    # T starts inside [329, 331] and is cooled below 329 K by the minimum coolant temperature.
    assert results[0][0][-1] is True
