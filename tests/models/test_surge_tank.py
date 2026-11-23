"""Independent checks of the surge tank: geometry, mass balance, domain invariance and default task."""

import gymnasium as gym
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from pcgym import make_env
from pcgym.models import get_model_spec
from pcgym.models.surge_tank import BAND, inflow_schedule, surge_tank_reward

SHIFT = {"model": "surge_tank", "N": 48, "tsim": 480}  # one 8-h shift, a decision every 10 min


def segment_volume(level, V_tot=28.0):
    """Liquid volume of a horizontal cylinder filled to ``level`` % of its diameter [m3]."""
    theta = np.arccos(1.0 - 2.0 * np.asarray(level) / 100.0)
    return V_tot * (theta - np.sin(theta) * np.cos(theta)) / np.pi


def valve(op, model):
    return model.Q_ref * np.exp(model.k_valve * (op - model.OP_ref))


def run(env, actions):
    obs, _ = env.reset(seed=0)
    total, levels = 0.0, [env.state[0]]
    for action in actions:
        obs, reward, terminated, truncated, _ = env.step(np.asarray(action, dtype=np.float32))
        total += reward
        levels.append(env.state[0])
    return total, np.array(levels)


@pytest.mark.parametrize("level", [15.0, 30.0, 50.0, 70.0, 85.0])
def test_level_rate_matches_horizontal_cylinder_geometry(level):
    model = get_model_spec("surge_tank").cls(int_method="casadi")
    imbalance = 3.0  # m3/h
    rate = model(np.array([level, 0.0, 50.0]), np.array([0.0, imbalance]))[0]
    h = 1e-4
    dVdL = (segment_volume(level + h) - segment_volume(level - h)) / (2 * h)
    assert rate == pytest.approx(imbalance / 60.0 / dVdL, rel=1e-6)


def test_default_x0_is_a_steady_valve_below_the_inflow():
    model = get_model_spec("surge_tank").cls(int_method="casadi")
    x0 = get_model_spec("surge_tank").defaults()["x0"]
    derivative = model(x0, np.array([0.0]))
    assert derivative[1] == pytest.approx(0.0, abs=1e-12)
    assert derivative[2] == 0.0
    assert derivative[0] > 0  # the nominal 20 m3/h inflow exceeds the valve's 18.46 m3/h
    np.testing.assert_allclose(model(np.array([50.0, 20.0, 52.0]), np.array([0.0])), 0.0, atol=1e-12)


def test_holding_the_valve_conserves_volume():
    env = make_env(SHIFT)
    times = np.arange(25) * env.dt
    expected = (20.0 - env.x0[1]) * times / 60.0
    exact = solve_ivp(
        lambda t, x: env.model(x, np.array([0.0])), (0, times[-1]), env.x0, t_eval=times, rtol=1e-11, atol=1e-11
    )
    np.testing.assert_allclose(segment_volume(exact.y[0]) - segment_volume(50.0), expected, rtol=1e-7, atol=1e-9)
    # The env's CVODES step reproduces it to its default tolerances.
    states = env.simulate(env.x0, np.zeros((24, 1)))
    np.testing.assert_allclose(segment_volume(states[:, 0]) - segment_volume(50.0), expected, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize("level,op", [(5.0, 70.0), (95.0, 30.0)])
def test_level_stays_inside_the_drum_when_it_empties_or_floods(level, op):
    env = make_env({**SHIFT, "N": 60, "tsim": 600})
    model = env.model
    q = valve(op, model)
    states = env.simulate([level, q, op], np.zeros((60, 1)))
    assert np.all((states[:, 0] > 0.0) & (states[:, 0] < 100.0))
    # Interior equilibria: the pump loses suction near empty, the surplus overflows near full.
    if level < 50:
        expected = -model.delta * np.log(1.0 - model.Q_in / q)
    else:
        expected = 100.0 + model.delta * np.log(1.0 - q / model.Q_in)
    assert states[-1, 0] == pytest.approx(expected, abs=1e-4)


def test_rejects_horizons_where_the_valve_rate_can_leave_its_range():
    with pytest.raises(ValueError, match="drive OP outside"):
        make_env({**SHIFT, "N": 300, "tsim": 3000})


def test_rejects_nonphysical_initial_states_and_parameters():
    with pytest.raises(ValueError, match="requires 0 < L < 100"):
        make_env({**SHIFT, "x0": np.array([100.0, 18.0, 50.0])})
    with pytest.raises(ValueError, match="requires positive"):
        make_env({**SHIFT, "model_params": {"tau": 0.0}})


def test_default_task_is_not_met_by_holding_and_is_feasible():
    hold, hold_levels = run(make_env(SHIFT), np.zeros((48, 1)))
    # Open at the full rate for 100 min to the balancing opening (52 %), then hold.
    actions = np.zeros((48, 1))
    actions[:10] = 1.0
    balanced, levels = run(make_env(SHIFT), actions)
    assert hold_levels.max() > 80 and hold < -1000
    assert levels.max() < BAND[1] and levels.min() >= 50.0
    assert balanced == pytest.approx(-2.0, rel=1e-6)  # 2 % of travel at lam = 1, never outside the band
    assert abs(levels[-1] - levels[-2]) < 1e-3


def test_band_reward_charges_excursion_minutes_and_valve_travel():
    env = make_env({**SHIFT, "x0": np.array([60.0, 20.0, 52.0]), "custom_reward": surge_tank_reward(lam=3.0)})
    env.reset(seed=0)
    _, reward, *_ = env.step(np.array([-0.5], dtype=np.float32))
    assert env.info["u"][0] == pytest.approx(-0.01)
    assert reward == pytest.approx(-env.dt * ((env.state[0] - 58.0) + 3.0 * 0.01), rel=1e-6)
    with pytest.raises(ValueError, match="band low < high"):
        surge_tank_reward(band=(58.0, 43.0))


def test_band_constraint_records_excursions_without_ending_the_episode():
    env = make_env(
        {
            **SHIFT,
            "constraints": {"L": list(BAND)},
            "cons_type": {"L": [">=", "<="]},
            "done_on_cons_vio": False,
            "r_penalty": False,
        }
    )
    env.reset(seed=0)
    for _ in range(48):
        *_, terminated, truncated, info = env.step(np.array([0.0], dtype=np.float32))
    assert not terminated and truncated
    above = info["cons_info"][1, 1:, 0]
    assert above.max() > 20 and np.all(info["cons_info"][0, 1:, 0] < 0)


def test_inflow_schedule_drives_and_is_observed_by_the_env():
    q = inflow_schedule(48, 480, rng=7)
    np.testing.assert_array_equal(q, inflow_schedule(48, 480, rng=7))
    assert q.shape == (48,) and abs(q.mean() - 20.0) < 1.5
    env = make_env({**SHIFT, "disturbances": {"Q_in": q}, "disturbance_bounds": {"low": [10.0], "high": [30.0]}})
    env.reset(seed=0)
    assert env.observation_info()[-1] == ("Q_in", "disturbance")
    level0, flow0 = env.state[0], env.state[1]
    env.step(np.array([0.0], dtype=np.float32))
    gained = segment_volume(env.state[0]) - segment_volume(level0)
    assert gained == pytest.approx((q[0] - flow0) * env.dt / 60.0, rel=1e-3)  # CVODES default tolerance
    with pytest.raises(ValueError, match="divisible by N"):
        inflow_schedule(48, 470)


class ResampledInflow(gym.Wrapper):
    """The docs' recipe: a fresh inflow schedule for every episode."""

    def __init__(self, env, seed=None):
        super().__init__(env)
        self.rng = np.random.default_rng(seed)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        base = self.env.unwrapped
        base.disturbances["Q_in"] = inflow_schedule(base.N, base.tsim, rng=self.rng)
        return self.env.reset(seed=seed, options=options)


def test_resampled_inflow_recipe_gives_a_new_schedule_per_episode():
    env = ResampledInflow(
        make_env(
            {
                **SHIFT,
                "disturbances": {"Q_in": np.full(48, 20.0)},
                "disturbance_bounds": {"low": [10.0], "high": [30.0]},
            }
        ),
        seed=0,
    )
    first, _ = env.reset()
    second, _ = env.reset()
    assert first[-1] != second[-1]
    again, _ = env.reset(seed=3)
    np.testing.assert_array_equal(again, env.reset(seed=3)[0])
