import numpy as np
import pytest

from pcgym import make_env

N = 10


@pytest.fixture
def env_params():
    return {
        "model": "cstr",
        "N": N,
        "tsim": 5,
        "SP": {"Ca": [0.85] * 5 + [0.9] * 5},
        "a_space": {"low": np.array([295]), "high": np.array([302])},
        "o_space": {"low": np.array([0.7, 300, 0.8]), "high": np.array([1, 350, 0.9])},
        "x0": np.array([0.8, 330, 0.8]),
        "integration_method": "casadi",
    }


def _run_episode(env):
    env.reset(seed=0)
    flags = []
    while True:
        _, _, terminated, truncated, _ = env.step(np.array([0.0]))
        flags.append((terminated, truncated))
        if terminated or truncated:
            return flags


def test_episode_runs_exactly_n_steps(env_params):
    flags = _run_episode(make_env(env_params))
    assert len(flags) == N


def test_time_limit_is_truncation_not_termination(env_params):
    flags = _run_episode(make_env(env_params))
    assert flags[-1] == (False, True)
    assert all(f == (False, False) for f in flags[:-1])


def test_step_flags_are_python_bools(env_params):
    env = make_env(env_params)
    env.reset()
    _, _, terminated, truncated, _ = env.step(np.array([0.0]))
    assert type(terminated) is bool and type(truncated) is bool


def test_constraint_violation_is_termination(env_params):
    env_params.update(
        {
            "constraints": {"T": [327, 1000]},
            "cons_type": {"T": [">=", "<="]},
            "done_on_cons_vio": True,
            "r_penalty": False,
            "normalise_o": False,
        }
    )
    env = make_env(env_params)
    env.reset()
    _, _, terminated, truncated, _ = env.step(np.array([-1.0]))
    assert terminated and not truncated


def test_batch_reward_is_paid_on_final_step():
    params = {
        "model": "cstr",
        "N": N,
        "tsim": 5,
        "a_space": {"low": np.array([295]), "high": np.array([302])},
        "o_space": {"low": np.array([0.7, 300]), "high": np.array([1, 350])},
        "x0": np.array([0.8, 330]),
        "reward_states": ["Ca"],
        "maximise_reward": True,
        "integration_method": "casadi",
    }
    env = make_env(params)
    env.reset()
    rewards = []
    for _ in range(N):
        _, r, *_ = env.step(np.array([0.0]))
        rewards.append(r)
    assert all(r == 0 for r in rewards[:-1])
    assert rewards[-1] == pytest.approx(env.state[0])
