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
        "constraints": lambda x, u: np.array([319 - x[1], x[1] - 331]).reshape(-1),
        "done_on_cons_vio": True,
        "r_penalty": True,
    }
    env = make_env(env_params)
    assert env.constraint_active
    assert env.done_on_constraint
    assert env.r_penalty

    env.reset()
    action = np.array([1.0])  # max-normalised action drives T past 331
    _, reward, done, _, info = env.step(action)
    assert done
    assert reward < 0
    assert "cons_info" in info
