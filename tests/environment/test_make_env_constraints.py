import pytest
import numpy as np
import sys
sys.path.append("..\src\pcgym") # Add local pc-gym files to path.

from pcgym import make_env

def test_make_env_constraints():
    env_params = {
        "model": "cstr",
        "a_space": {"low": np.array([-1]), "high": np.array([1])},
        "o_space": {"low": np.array([-1, -1]), "high": np.array([1, 1])},
        "SP": {"T": [350] * 100},
        "N": 100,
        "tsim": 10,
        "x0": np.array([0.5, 350]),
        "constraints": {"Ca": [0, 1], "T": [300, 400]},
        "done_on_cons_vio": True,
        "r_penalty": True,
        "cons_type": {"Ca": [">=", "<="], "T": [">=", "<="]}
    }
    env = make_env(env_params)
    assert env.constraint_active
    assert env.done_on_constraint
    assert env.r_penalty
    
    env.reset()
    action = np.array([500])  # Action that should violate temperature constraint
    _, reward, done, _, info = env.step(action)
    assert done  # Episode should end due to constraint violation
    assert reward < 0  # Should receive a penalty
    assert "cons_info" in info