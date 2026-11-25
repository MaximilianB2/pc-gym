"""Evaluation must describe completed transitions, including short episodes."""

from unittest.mock import patch

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from pcgym import make_env
from pcgym.evaluation_metrics import median_absolute_deviation, reproducibility_metric
from pcgym.policy_evaluation import policy_eval


class ConstantPolicy:
    def __init__(self, action=(0.0,)):
        self.action = np.array(action)

    def predict(self, obs, deterministic=True):
        return self.action.copy(), None


def params(**extra):
    return {"model": "first_order_system", "N": 4, "tsim": 1, **extra}


def test_short_uncertain_episode_has_terminal_state_and_explicit_masks():
    config = params(
        uncertainty_percentages={"K": 0.1},
        uncertainty_bounds={"low": [0.9], "high": [1.1]},
        constraints={"x": [0.2]},
        cons_type={"x": ["<="]},
        done_on_cons_vio=True,
        r_penalty=False,
    )
    evaluator = policy_eval(make_env, {"pi": ConstantPolicy([1.0])}, 2, config, seeds=[10, 11])
    data = evaluator.get_rollouts()["pi"]
    np.testing.assert_array_equal(data["lengths"], [1, 1])
    np.testing.assert_array_equal(data["terminated"], [True, True])
    assert not data["truncated"].any()
    assert data["x"].shape == (3, 5, 2)
    np.testing.assert_array_equal(data["state_mask"][:, 0], [True, True, False, False, False])
    np.testing.assert_array_equal(data["transition_mask"][:, 0], [True, False, False, False])
    assert np.isfinite(data["x"][:, :2]).all()
    for component in ("r", "x", "g"):
        assert np.isnan(data[component][:, 2:]).all()
    assert np.isnan(data["u"][:, 1:]).all()
    assert np.all(data["g"][:, 1] > 0)


def test_policies_receive_matched_reproducible_scenarios():
    config = params(noise=True, noise_percentage=0.02, uncertainty_percentages={"x0": {0: 0.5}})
    evaluator = policy_eval(make_env, {"a": ConstantPolicy(), "b": ConstantPolicy()}, 3, config, seed=81)
    first = evaluator.get_rollouts()
    second = evaluator.get_rollouts()
    for key in ("x", "r", "u", "seeds"):
        np.testing.assert_array_equal(first["a"][key], first["b"][key])
        np.testing.assert_array_equal(first["a"][key], second["a"][key])
    assert not np.array_equal(first["a"]["x"][:, :, 0], first["a"]["x"][:, :, 1])


@pytest.mark.parametrize("custom", [False, True])
def test_batch_and_custom_rewards_match_direct_steps(custom):
    config = params(reward_states=["x"], x0=[0.2], o_space={"low": [0], "high": [1]})
    if custom:
        config.update(custom_reward=lambda env, x, u, violation: float(x[0] + env.t), finite_horizon=True)
    evaluator = policy_eval(make_env, {"pi": ConstantPolicy()}, 1, config, seeds=[5])
    result = evaluator.get_rollouts()["pi"]
    env = make_env(config)
    env.reset(seed=5)
    for t in range(1, 5):
        _, reward, term, trunc, _ = env.step([0.0])
        np.testing.assert_allclose(result["x"][:, t, 0], env.state)
        assert result["r"][0, t, 0] == reward
    assert term and not trunc
    assert result["terminated"][0] and result["lengths"][0] == 4


def test_recurrent_policy_state_restarts_each_episode():
    class Recurrent:
        def predict(self, obs, state=None, episode_start=None, deterministic=True):
            if episode_start[0]:
                assert state is None
                state = 0
            else:
                assert state > 0
            return np.array([0.0]), state + 1

    policy_eval(make_env, {"pi": Recurrent()}, 2, params()).get_rollouts()


@pytest.mark.parametrize(
    "extra",
    [
        {"noise": True, "noise_percentage": 0.01},
        {"uncertainty_percentages": {"x0": {0: 0.1}}},
        {"custom_reward": lambda *args: 0.0, "SP": {"x": [0.7]}},
        {"reward_states": ["x"], "x0": [0.2], "o_space": {"low": [0], "high": [1]}},
        {"done_on_cons_vio": True, "constraints": {"x": [0.8]}, "cons_type": {"x": ["<="]}, "r_penalty": False},
    ],
)
def test_oracle_rejects_unmatched_or_unsupported_comparisons(extra):
    with pytest.raises(ValueError, match="oracle"):
        policy_eval(make_env, {}, 1, params(**extra), oracle=True)


def test_oracle_reward_does_not_change_live_episode():
    evaluator = policy_eval(make_env, {}, 1, params(SP={"x": [0.1, 0.2, 0.3, 0.4, 0.5]}))
    evaluator.env.reset(seed=4)
    evaluator.env.step([0.0])
    original = evaluator.env.state.copy()
    rewards = evaluator.oracle_reward_fn(np.array([[0.1, 0.2, 0.5]]), np.array([[0.5, 0.5]]))
    np.testing.assert_allclose(rewards, [0, 0, -0.04])
    assert evaluator.env.t == 1
    np.testing.assert_array_equal(evaluator.env.state, original)


def test_metrics_use_episode_returns_and_ignore_padding_and_metadata():
    # The first episode ends after one transition. Even finite padding is excluded by its mask.
    data = {
        "pi": {
            "r": np.array([[[0, 0], [2, 1], [100, 3]]]),
            "x": np.array([[[1, 3], [2, 4], [100, 6]]]),
            "u": np.array([[[1, 3], [100, 5]]]),
            "g": np.array([[[[-9, -4]], [[2, -1]], [[100, -3]]]]),
            "state_mask": np.array([[True, True], [True, True], [False, True]]),
            "transition_mask": np.array([[True, True], [False, True]]),
            "lengths": np.array([1, 2]),
        }
    }
    metric = reproducibility_metric("std", "mean", 0)
    mean = metric.policy_performance_metric(data, None)["pi"]
    np.testing.assert_allclose(mean["r"], [3])
    np.testing.assert_allclose(mean["x"], [[2, 3, 6]])
    np.testing.assert_allclose(mean["u"], [[2, 5]])
    np.testing.assert_allclose(mean["g"].ravel(), [0, 1, 0])
    assert set(mean) == {"r", "x", "u", "g"}
    np.testing.assert_allclose(metric.policy_dispersion_metric(data, "r")["pi"]["r"], [1])
    with pytest.raises(ValueError, match="component"):
        metric.determine_op("lengths")


def test_mad_broadcasts_over_repetitions():
    values = np.array([[[1, 2, 4], [5, 7, 12]], [[2, 4, 7], [1, 8, 9]]])
    np.testing.assert_array_equal(median_absolute_deviation(values).get_value(), [[1, 2], [2, 1]])


@pytest.mark.parametrize("batch", [False, True])
def test_plot_handles_short_schedules_batch_and_episode_return_histogram(batch):
    config = params(SP={"x": [0.4, 0.5]})
    if batch:
        config = params(reward_states=["x"], x0=[0.2], o_space={"low": [0], "high": [1]})
    evaluator = policy_eval(make_env, {"pi": ConstantPolicy()}, 1, config)
    data = evaluator.get_rollouts()
    with (
        patch("matplotlib.pyplot.show"),
        patch("matplotlib.axes.Axes.hist", autospec=True, side_effect=Axes.hist) as hist,
    ):
        evaluator.plot_data(data, reward_dist=True)
    np.testing.assert_allclose(hist.call_args.args[1], data["pi"]["r"][:, 1:].sum(axis=1).ravel())
    plt.close("all")


def test_seed_arguments_validate_and_convenience_api_propagates():
    for kwargs in ({"reps": 0}, {"reps": 2, "seeds": [1]}, {"reps": 1, "seeds": [-1]}):
        with pytest.raises(ValueError):
            policy_eval(make_env, {}, env_params=params(), **kwargs)
    _, data = make_env(params()).get_rollouts({"pi": ConstantPolicy()}, 1, seeds=[41])
    np.testing.assert_array_equal(data["pi"]["seeds"], [41])


def test_plot_multiple_disturbances_and_positive_violations_after_early_end():
    config = {
        "model": "cstr",
        "N": 4,
        "tsim": 1,
        "disturbances": {"Ti": [350, 345], "Caf": [1.0]},
        "disturbance_bounds": {"low": [340, 0.9], "high": [355, 1.1]},
        "constraints": {"Ca": [0.81]},
        "cons_type": {"Ca": ["<="]},
        "done_on_cons_vio": True,
        "r_penalty": False,
    }
    evaluator = policy_eval(make_env, {f"pi{i}": ConstantPolicy() for i in range(6)}, 2, config, cons_viol=True)
    data = evaluator.get_rollouts()
    assert np.all(data["pi0"]["lengths"] < 4)
    with patch("matplotlib.pyplot.show"):
        evaluator.plot_data(data)
    trajectory, violation = (plt.figure(i) for i in plt.get_fignums())
    assert [ax.get_ylabel() for ax in trajectory.axes] == ["Ca", "T", "Tc", "Ti", "Caf"]
    np.testing.assert_array_equal(trajectory.axes[-2].lines[0].get_ydata(), [350, 345, 345, 345, 345])
    assert np.nanmin(violation.axes[0].lines[0].get_ydata()) >= 0
    plt.close("all")


def test_oracle_constraint_penalty_uses_initial_and_next_state():
    evaluator = policy_eval(
        make_env,
        {},
        1,
        params(
            SP={"x": [0.3]},
            constraints={"x": [0.2]},
            cons_type={"x": ["<="]},
            done_on_cons_vio=False,
            r_penalty=True,
        ),
    )
    rewards = evaluator.oracle_reward_fn(np.array([[0.3, 0.1, 0.1]]), np.zeros((1, 2)))
    np.testing.assert_allclose(rewards, [0, -1000.04, -0.04])


@pytest.mark.parametrize("reward", [np.nan, np.inf, -np.inf])
def test_invalid_custom_rewards_are_failures_not_missing_samples(reward):
    evaluator = policy_eval(
        make_env, {"pi": ConstantPolicy()}, 1, params(SP={"x": [0.7]}, custom_reward=lambda *args: reward)
    )
    with pytest.raises(ValueError, match="Nonfinite.*transition 1"):
        evaluator.get_rollouts()
