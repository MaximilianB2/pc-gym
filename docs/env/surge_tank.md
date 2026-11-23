# Surge tank: averaging level control

```python
from pcgym import make_env

env = make_env({"model": "surge_tank", "N": 48, "tsim": 480})  # one 8-h shift, a decision every 10 min
```

A horizontal cylindrical drum buffers a fluctuating inflow ahead of a downstream unit that runs
best on a steady feed, such as a thermally integrated evaporator. One outlet valve is the only
handle. The level is a **band, not a setpoint**: moving level is how the drum absorbs inflow
variation, and holding it at 50 % would pass every inflow change straight to the outflow. The task
is to move the valve as little as possible while keeping the level inside [43, 58] %. This is
the classical averaging level control problem [1, 2].

The structure and magnitudes follow an industrial surge drum, but the parameters are rounded
and generic: this is a synthetic benchmark, not a calibration of a particular plant.

## Model

Time is in minutes, flows in m³/h, level and valve opening in %. The
[model](../../src/pcgym/models/surge_tank.py) has three states, one input and one disturbance.

| Symbol | Meaning | Unit |
|---|---|---|
| `L` | level, as % of the drum diameter | % |
| `Q_out` | outlet flow | m³/h |
| `OP` | valve opening | % |
| `dOP` (input) | valve rate | %/min |
| `Q_in` (disturbance) | inflow | m³/h |

```text
dL/dt     = (Q_in * room(L) - Q_out * suction(L)) / 60 / C(L)
dQ_out/dt = (Q_ref * exp(k_valve * (OP - OP_ref)) - Q_out) / tau
dOP/dt    = dOP

C(L)       = V_tot/100 * (8/pi) * ((l (1 - l))^2 + eps^2)^(1/4),   l = L/100
suction(L) = 1 - exp(-L / delta)
room(L)    = 1 - exp(-(100 - L) / delta)
```

- **Capacity.** `C(L)` is the volume per %-point of a horizontal cylinder, widest at mid-level:
  0.357 m³/% at 50 %, 0.255 at 15 % and 85 %. The level runs faster near the limits, exactly where
  a mistake costs most. It integrates to `V_tot` over 0–100 %; `eps` only keeps it finite at the ends.
- **Valve.** An equal-percentage characteristic (rangeability about 55) with a first-order lag.
  Around 20 m³/h, one % of opening moves the flow by about 0.8 m³/h.
- **Valve rate as the input.** The opening is a state and the agent sets its rate. The default
  bound of ±0.02 %/min is an actuation contract of ±0.2 % per 10-minute decision. This keeps the
  rate limit inside the action space, makes the current opening observable, and lets the reward
  see each move.
- **Ends of the drum.** Near empty the pump loses suction and near full the surplus overflows,
  both fading in over `delta` %-points. The level therefore stays inside (0, 100) for any input
  and horizon; above 15 % and below 85 % the gates differ from 1 by less than 1e-13. `Q_out`
  remains the flow the valve would pass.

| Parameter | Default | Unit | Meaning |
|---|---:|---|---|
| `V_tot` | 28 | m³ | drum volume |
| `Q_ref` | 20 | m³/h | valve static flow at `OP_ref` |
| `OP_ref` | 52 | % | reference opening |
| `k_valve` | 0.04 | 1/% | equal-percentage slope |
| `tau` | 3 | min | outlet-flow lag |
| `Q_in` | 20 | m³/h | inflow used when no disturbance schedule is given |
| `eps` | 1e-4 | – | capacity regularisation at the ends |
| `delta` | 0.5 | % | width of the suction-loss and overflow regions |

At 20 m³/h, half the drum holds about 42 minutes of flow, and a 1 m³/h imbalance moves the level
about 2.8 %-points per hour near mid-level.

## Default task

- **Initial state:** `[L, Q_out, OP] = [50, 18.46, 50]`. The valve is steady at 50 % while the
  inflow is 20 m³/h: the hand-over after an 8 % rise upstream. Holding the valve fills the drum.
- **Actions:** `dOP` in [-0.02, 0.02] %/min, normalised to [-1, 1] by default.
- **Observations:** `L` in [0, 100], `Q_out` in [0, 40], `OP` in [0, 100]. The inflow is not
  observed unless a disturbance schedule is given.
- **Reward:** one objective and one soft constraint.

  ```text
  r_t = -dt * ( %-points of L_(t+1) outside [43, 58]  +  lam * |dOP| )      lam = 1
  ```

  A minute spent one %-point outside costs 1; one % of valve travel costs `lam`. The outflow is
  the valve's flow behind a short lag, so pricing travel prices outflow variation. `lam` trades
  outflow smoothness against band keeping. Use `surge_tank_reward(lam=..., band=...)` from
  `pcgym.models.surge_tank` as `custom_reward` to change either.
- **Horizon:** continuing; episodes are truncated after `N` steps. The validated configuration is
  `N=48, tsim=480`. Construction rejects horizons over which the rate bound could drive `OP`
  outside [0, 100] %.

On the default task, an 8-hour episode scores:

| Controller | Return | %-pt·min outside | Valve travel % |
|---|---:|---:|---:|
| hold the valve | −5477 | 5477 | 0 |
| proportional on the level, best of five gains (`dOP = 0.008 (L - 50)`) | −8.6 | 0 | 8.6 |
| open at the full rate to 52 %, then hold | −2.0 | 0 | 2.0 |

The last row moves the valve exactly to the opening that balances the inflow. A finite-horizon
optimum can do slightly better by leaving a small residual imbalance.

## Stochastic shifts

`inflow_schedule(N, tsim, rng)` draws a within-shift inflow: mean-reverting jitter of 0.3 m³/h
(12-minute correlation time) around 20 m³/h, plus at most one event per shift. The event is either
a step or a ramp of 0.6–1.2 m³/h, or a 30–90 minute oscillation. Pass it as a
[disturbance](../guides/disturbances.md). pc-gym appends disturbances to the observation, so the
agent then sees the inflow acting over the next interval. On a real drum the inflow is often
unmetered and must be inferred from the level, so this variant is easier than that one.

A schedule is fixed when the environment is built. This wrapper draws a new one for every episode:

```python
import gymnasium as gym
import numpy as np

from pcgym import make_env
from pcgym.models.surge_tank import inflow_schedule


class ResampledInflow(gym.Wrapper):
    def __init__(self, env, seed=None):
        super().__init__(env)
        self.rng = np.random.default_rng(seed)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        base = self.env.unwrapped
        base.disturbances["Q_in"] = inflow_schedule(base.N, base.tsim, rng=self.rng)
        return self.env.reset(seed=seed, options=options)


env = ResampledInflow(
    make_env(
        {
            "model": "surge_tank",
            "N": 48,
            "tsim": 480,
            "x0": np.array([50.0, 20.0, 52.0]),  # balanced hand-over
            "uncertainty_percentages": {"x0": {2: 0.5 / 52}},  # opening within +-0.5 %
            "disturbances": {"Q_in": inflow_schedule(48, 480, rng=0)},
            "disturbance_bounds": {"low": np.array([10.0]), "high": np.array([30.0])},
        }
    )
)
```

Over 200 shifts of this recipe (seeds 0–199), with returns at two prices of valve travel:

| Controller | %-pt·min outside | Valve travel % | Worst-hour swing of `Q_out`, m³/h | Return, `lam=1` | Return, `lam=10` |
|---|---:|---:|---:|---:|---:|
| hold the valve | 262.8 | 0 | 0.20 | −262.8 | −262.8 |
| proportional, `dOP = 0.004 (L - 50)` | 0.0 | 2.20 | 0.47 | −2.2 | −22.0 |
| proportional, `dOP = 0.015 (L - 50)` | 0.7 | 5.09 | 0.89 | −5.8 | −51.6 |
| deadband: hold inside 46–54 %, gain 0.015 outside | 4.0 | 1.22 | 0.57 | −5.2 | −16.2 |
| feedforward on the observed inflow, with a level trim | 0.0 | 7.52 | 0.67 | −7.5 | −75.2 |

Rates are clipped to the action bounds. At `lam=1` a slow proportional controller is close to
the best of these; once travel is priced higher, holding inside a deadband wins. Chasing the
observed inflow keeps the level in band but spends the most travel.

## Constraints and randomisation

To report the band separately from the objective, for constrained RL or
[policy evaluation](../guides/Policy%20Evaluation.md), declare it as a constraint that does not
end the episode:

```python
{"constraints": {"L": [43.0, 58.0]}, "cons_type": {"L": [">=", "<="]},
 "done_on_cons_vio": False, "r_penalty": False}
```

`info["cons_info"]` then records `43 - L` and `L - 58` at every step. `done_on_cons_vio` applies to
all constraints at once, so a terminating safety limit cannot be combined with this soft band.

Useful randomisation axes are `k_valve` (the valve's gain drifts between operating campaigns),
`V_tot` (level-transmitter span and capacity) and `tau`. Use `uncertainty_percentages` with
`uncertainty_bounds`; pc-gym appends the sampled values to the observation.

## Validation

- Independent [tests](../../tests/models/test_surge_tank.py) check the capacity against the
  closed-form horizontal-cylinder volume at 15–85 %, and volume conservation to 1e-7 with a tight
  independent integrator. They also check the interior equilibria when the drum empties or floods,
  the domain rejections, the reward, the band constraint, the inflow schedule's wiring and the
  per-episode wrapper.
- The reference trajectory (30 steps of 10 min) agrees between CasADi and JAX to 8e-4 %-points.
- The strict [readiness audit](../../scripts/audit_env_readiness.py) passes all 7 default action
  scenarios with both backends, including the Gymnasium checker.
- PPO and SAC compatibility checks (2,048 transitions, seed 0) pass. Learning quality is not
  qualified.

## References

1. K. A. McDonald, T. J. McAvoy and A. Tits, "Optimal averaging level control," *AIChE Journal*
   32(1), 75–86, 1986. [doi:10.1002/aic.690320109](https://doi.org/10.1002/aic.690320109)
2. P. J. Campo and M. Morari, "Model predictive optimal averaging level control," *AIChE Journal*
   35(4), 579–591, 1989. [doi:10.1002/aic.690350408](https://doi.org/10.1002/aic.690350408)
