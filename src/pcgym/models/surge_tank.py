from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from pcgym.models._base import BaseModel, jnp
from pcgym.models._registry import CustomReward, register_model

BAND = (43.0, 58.0)  # level band of the default task [%]


def _defaults():
    """Canonical a_space / o_space / x0 used when they are omitted from env_params.

    x0 is the hand-over after an upstream rate rise: the drum at mid-level and the valve steady at
    50 % (18.46 m3/h) while the nominal inflow is 20 m3/h. Holding the valve fills the drum.
    """
    return {
        # +-0.02 %/min is the actuation contract of +-0.2 % per 10-min decision.
        "a_space": {"low": np.array([-0.02]), "high": np.array([0.02])},
        "o_space": {"low": np.array([0.0, 0.0, 0.0]), "high": np.array([100.0, 40.0, 100.0])},
        "x0": np.array([50.0, 20.0 * np.exp(0.04 * (50.0 - 52.0)), 50.0]),  # [L, Q_out, OP]
    }


def surge_tank_reward(lam: float = 1.0, band: tuple = BAND):
    """Averaging level control: move the valve as little as possible, keep the level in the band.

    r_t = -(dt * (%-points of L_(t+1) outside the band) + lam * |dOP| * dt)

    One minute one %-point outside costs 1; one % of valve travel costs lam. The outflow is the
    valve's static flow behind a short lag, so pricing travel prices outflow variation; lam trades
    outflow smoothness against band keeping.
    """
    lo, hi = float(band[0]), float(band[1])
    lam = float(lam)
    if not lo < hi or lam < 0:
        raise ValueError("surge_tank_reward needs band low < high and lam >= 0")

    def reward(env, state, action, constraint_violated):
        level = state[0]
        excess = max(lo - level, 0.0) + max(level - hi, 0.0)
        return -env.dt * (excess + lam * abs(action[0]))

    return reward


def inflow_schedule(N, tsim, rng=None, q_nom=20.0, sigma=0.3, theta=0.165, event_size=(0.6, 1.2)):
    """A synthetic within-shift inflow schedule [m3/h] for ``env_params["disturbances"]["Q_in"]``.

    Ornstein-Uhlenbeck jitter around ``q_nom`` (stationary spread ``sigma`` m3/h, mean reversion
    ``1/theta`` min) on a one-minute grid, plus at most one event starting between 10 % and 60 % of
    the horizon: none, a step or a ramp of ``event_size`` m3/h (either sign), or a 30-90 min
    oscillation of half that amplitude. Returns the mean over each of the ``N`` intervals.
    """
    rng = np.random.default_rng(rng)
    n_min = int(round(tsim))
    if abs(tsim - n_min) > 1e-9 or n_min % N:
        raise ValueError("inflow_schedule needs tsim in whole minutes and divisible by N")
    a = np.exp(-theta)
    noise = sigma * np.sqrt(1.0 - a * a) * rng.standard_normal(n_min)
    q = np.empty(n_min)
    x = 0.0
    for i in range(n_min):
        x = a * x + noise[i]
        q[i] = x
    t = np.arange(n_min)
    t0 = int(rng.uniform(0.1, 0.6) * n_min)
    size = rng.uniform(*event_size) * rng.choice([-1.0, 1.0])
    kind = rng.choice(["none", "step", "ramp", "oscillation"])
    if kind == "step":
        q += size * (t >= t0)
    elif kind == "ramp":
        q += size * np.clip((t - t0) / (n_min - t0), 0.0, 1.0)
    elif kind == "oscillation":
        q += 0.5 * abs(size) * np.sin(2 * np.pi * (t - t0) / rng.uniform(30.0, 90.0)) * (t >= t0)
    return q_nom + q.reshape(N, -1).mean(axis=1)


class SurgeTankShifts(gym.Wrapper):
    """The stochastic-shift benchmark: a fresh inflow schedule every episode, unmetered by default.

    Wraps a ``surge_tank`` env built with a ``Q_in`` disturbance schedule. Each reset draws a new
    schedule with :func:`inflow_schedule`. With ``metered=False`` the inflow is removed from the
    observation, as on a drum without an inlet flow meter: the imbalance must then be inferred from
    how the level moves, which needs memory (frame stacking or a recurrent policy).
    """

    def __init__(self, env, seed=None, metered=False):
        super().__init__(env)
        base = env.unwrapped
        if not base.disturbance_active or "Q_in" not in base.disturbances:
            raise ValueError("SurgeTankShifts needs an env built with a 'Q_in' disturbance schedule")
        self.rng = np.random.default_rng(seed)
        layout = base.observation_info()
        self._keep = np.array([kind != "disturbance" for _, kind in layout]) if not metered else None
        if self._keep is not None:
            space = env.observation_space
            self.observation_space = gym.spaces.Box(space.low[self._keep], space.high[self._keep], dtype=space.dtype)

    def _observe(self, obs, info):
        if self._keep is None:
            return obs, info
        info["obs"] = info["obs"][self._keep]  # pc-gym's full-precision copy would leak the inflow
        return obs[self._keep], info

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        base = self.env.unwrapped
        base.disturbances["Q_in"] = inflow_schedule(base.N, base.tsim, rng=self.rng)
        return self._observe(*self.env.reset(seed=seed, options=options))

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs, info = self._observe(obs, info)
        return obs, reward, terminated, truncated, info


@register_model("surge_tank", defaults=_defaults, task=CustomReward(surge_tank_reward()))
@dataclass(frozen=False, kw_only=True)
class surge_tank(BaseModel):
    """Horizontal cylindrical surge drum with one outlet valve (averaging level control).

    The drum buffers a fluctuating inflow ahead of a downstream unit that wants a steady feed. The
    level is a band, not a setpoint: it moves so that the outflow does not have to. Time is in
    minutes, flows in m3/h, level and valve opening in %.

    States: L (level, % of diameter), Q_out (outlet flow, m3/h), OP (valve opening, %).
    Input: dOP (valve rate, %/min). Disturbance: Q_in (inflow, m3/h).

    Near empty the pump loses suction and near full the surplus overflows: both fade in over
    ``delta`` %-points, so the level stays inside (0, 100) for any input. They change nothing
    measurable above 15 % or below 85 %.
    """

    V_tot: float = 28.0  # drum volume [m3]
    Q_ref: float = 20.0  # valve static flow at OP_ref [m3/h]
    OP_ref: float = 52.0  # reference valve opening [%]
    k_valve: float = 0.04  # equal-percentage slope, rangeability exp(100*k_valve) ~ 55 [1/%]
    tau: float = 3.0  # outlet-flow lag behind the valve's static flow [min]
    Q_in: float = 20.0  # inflow when no disturbance schedule is given [m3/h]
    eps: float = 1e-4  # keeps the capacity finite at empty and full [-]
    delta: float = 0.5  # width of the suction-loss and overflow regions [%]
    int_method: str = "jax"
    states: list = None
    inputs: list = None
    disturbances: list = None
    uncertainties: dict = None

    def __post_init__(self):
        self.states = ["L", "Q_out", "OP"]
        self.inputs = ["dOP"]
        self.disturbances = ["Q_in"]

    def validate_initial_state(self, state):
        if not 0.0 < state[0] < 100.0 or state[1] < 0.0 or not 0.0 <= state[2] <= 100.0:
            raise ValueError("surge_tank requires 0 < L < 100 %, Q_out >= 0 and 0 <= OP <= 100 %")

    def validate_input_bounds(self, low, high):
        positive = [self.V_tot, self.Q_ref, self.k_valve, self.tau, self.eps, self.delta]
        if not np.isfinite(positive + [self.OP_ref, self.Q_in]).all() or min(positive) <= 0 or self.Q_in < 0:
            raise ValueError("surge_tank requires positive V_tot, Q_ref, k_valve, tau, eps, delta and Q_in >= 0")

    def validate_episode(self, state, low, high, duration):
        if state[2] + low[0] * duration < 0.0 or state[2] + high[0] * duration > 100.0:
            raise ValueError("surge_tank: the valve-rate bounds can drive OP outside [0, 100] % within the horizon")

    def __call__(self, x, u):
        xp = jnp if self.int_method == "jax" else np
        L, Q_out, OP = x[0], x[1], x[2]
        dOP = u[0]
        Q_in = u[1] if u.shape[0] > 1 else self.Q_in

        s = L / 100.0 * (1.0 - L / 100.0)
        # Horizontal cylinder: dV/dL = V_tot/100 * (8/pi) * sqrt(l (1 - l)), l = L/100 [m3/%].
        capacity = self.V_tot / 100.0 * (8.0 / np.pi) * (s * s + self.eps * self.eps) ** 0.25
        suction = 1.0 - xp.exp(-L / self.delta)
        room = 1.0 - xp.exp(-(100.0 - L) / self.delta)
        q_static = self.Q_ref * xp.exp(self.k_valve * (OP - self.OP_ref))

        dLdt = (Q_in * room - Q_out * suction) / 60.0 / capacity
        dQdt = (q_static - Q_out) / self.tau
        ret = [dLdt, dQdt, dOP]
        return jnp.array(ret) if self.int_method == "jax" else np.array(ret)
