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

## Stochastic shifts: the benchmark

`inflow_schedule(N, tsim, rng)` draws a within-shift inflow: mean-reverting jitter of 0.3 m³/h
(12-minute correlation time) around 20 m³/h, plus at most one event per shift. The event is either
a step or a ramp of 0.6–1.2 m³/h, or a 30–90 minute oscillation.

`SurgeTankShifts` turns this into the benchmark. It draws a new schedule at every reset (pc-gym
fixes a [disturbance](../guides/disturbances.md) schedule when the environment is built). It also
removes the inflow from the observation, as on a drum without an inlet flow meter. The agent then
sees only `[L, Q_out, OP]` and must infer the imbalance from how the level moves, so it needs
memory: stack frames or use a recurrent policy.

```python
import gymnasium as gym
import numpy as np

from pcgym import make_env
from pcgym.models.surge_tank import SurgeTankShifts, inflow_schedule

env = SurgeTankShifts(
    make_env(
        {
            "model": "surge_tank",
            "N": 48,
            "tsim": 480,
            "x0": np.array([50.0, 20.0, 52.0]),  # balanced hand-over
            "uncertainty_percentages": {"x0": {2: 0.5 / 52}},  # opening within +-0.5 %
            "disturbances": {"Q_in": inflow_schedule(48, 480, rng=0)},  # replaced at every reset
            "disturbance_bounds": {"low": np.array([10.0]), "high": np.array([30.0])},
        }
    ),
    seed=0,
)
env = gym.wrappers.FrameStackObservation(env, 2)  # the current and the previous reading
```

`SurgeTankShifts(..., metered=True)` keeps the inflow in the observation: an easier variant in
which the imbalance is visible.

Over 200 shifts (seeds 0–199), with returns at two prices of valve travel:

| Controller | Needs | %-pt·min outside | Valve travel % | Writes | Worst-hour swing of `Q_out`, m³/h | Return, `lam=1` | Return, `lam=10` |
|---|---|---:|---:|---:|---:|---:|---:|
| hold the valve | – | 262.8 | 0 | 0 | 0.20 | −262.8 | −262.8 |
| proportional, `dOP = 0.004 (L - 50)` | current reading | 0.0 | 2.20 | 47.0 | 0.47 | −2.2 | −22.0 |
| deadband: hold inside 47–53 %, gain 0.015 outside | current reading | 0.9 | 1.38 | 12.1 | 0.58 | −2.3 | −14.8 |
| trend rule: full rate when the level 30 min ahead leaves 46–54 % | one past reading | 0.0 | 1.02 | 5.1 | 0.55 | −1.1 | −10.3 |
| trend rule: full rate when the level 60 min ahead leaves 44–56 % | one past reading | 1.0 | 0.79 | 4.0 | 0.45 | −1.8 | −8.9 |
| feedforward toward the opening that passes the inflow, ±0.3 % hysteresis | metered inflow | 0.0 | 3.94 | 19.7 | 0.56 | −3.9 | −39.4 |

Rates are clipped to the action bounds; a write is any step with a nonzero rate. Feedback on the
current level alone does not anticipate: the best memoryless controllers stay near −2.2 at
`lam=1` and −14.8 at `lam=10`. A two-parameter rule that extrapolates the level from one past
reading halves the cost at `lam=1` and cuts it by 40 % at `lam=10`, mostly by moving the valve
four or five times a shift instead of 12 to 47. A learned policy should at least match it.
Chasing even a metered inflow spends the most travel, because it follows the jitter as well as the
upsets.

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
  the domain rejections, the reward, the band constraint and the inflow schedule's wiring. The
  benchmark wrapper is checked to hide the inflow everywhere, and the trend rule to stay below
  0.6 of the proportional controller's cost (0.41 on 40 shifts).
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
