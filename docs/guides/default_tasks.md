# Default tasks

Every built-in model with default spaces also has a **default task**, so its spaces, initial state and reward can be supplied automatically:

```python
from pcgym import make_env

env = make_env({"model": "cstr", "N": 60, "tsim": 26})
```

The default task is used only when **no reward is configured**, meaning none of `SP`, `custom_reward` or `reward_states` is given. An explicitly configured reward is never modified.

A registered task is not a readiness certificate. The [current validation record](../plans/training_readiness_plan.md#surge-tank-averaging-level-control)
lists tested horizons and remaining learning/backend gaps. All 19 controlled defaults now have
nominal CasADi runtime and PPO/SAC compatibility evidence; multi-seed learning and experimental
calibration are separate claims.

## Regulation

The tracked state(s) are held at a constant setpoint for the whole episode. The reward at each step is

```
r_t = - sum_k (x_k,t - SP_k)^2 / (o_space_high_k - o_space_low_k)^2
```

Each squared tracking error is divided by the squared width of that state's `o_space` range.
This makes the error dimensionless; returns still depend on the task, horizon and selected ranges,
so they do not establish comparable control performance across models. Passing your own `r_scale` replaces this scaling.

When the default task is used:

- `SP` is set to the constant setpoint.
- `x0`'s setpoint entries are set to the setpoint values. If `x0` has only the model states, the setpoint entries are appended.
- If `o_space` has only the model states, the tracked states' bounds are reused for the setpoint entries.

## Batch

The reward is zero until the final step, which pays the value of the reward states (maximised).
Batch tasks append time remaining and terminate at their horizon. A `Regulation` task can also
declare `finite_horizon=True`, as the reactor–separator task does.

## Model-specific objectives

A registered `CustomReward` task supplies the existing `custom_reward(env, state, action, violation)`
callback and can declare a finite horizon. The disease default uses this to minimize infection burden
plus vaccination effort; the surge-tank default charges level excursions outside a band plus
valve travel. The callback receives physical measurements and applied physical controls.
An explicitly configured reward takes precedence over the registered task, including its horizon flag.

## Default task per model

| Model | Task | Setpoint | Normalising range |
|---|---|---|---|
| [batch](../env/batch.md) | Batch | maximise `Cb` at the final step | – |
| [biofilm_reactor](../env/biofilm.md) | Regulation | `S2_A` = 0.25 mg N/L | `S2_A`: 2.1 |
| `crystallization` | Regulation | `CV` = 1, `Ln` = 15 | `CV`: 3, `Ln`: 50 |
| `cstr` | Regulation | `Ca` = 0.9 | `Ca`: 0.3 |
| [complex_cstr](../env/complex_cstr.md) | Regulation | `Cb` = 0.25 | `Cb`: 2 |
| `cstr_series_recycle` | Regulation | `C2` = 97 | `C2`: 100 |
| [disease](../env/disease.md) | Finite horizon, custom reward | minimize `sum dt*(I + 20*u²)` | physical population fractions and daily rates |
| `distillation_column` | Regulation | `X0` = 0.5 | `X0`: 1 |
| `first_order_system` | Regulation | `x` = 0.7 | `x`: 1 |
| `four_tank` | Regulation | `h3` = 0.1, `h4` = 0.3 | `h3`: 0.6, `h4`: 0.6 |
| `heat_exchanger` | Regulation | `Tt8` = 320 | `Tt8`: 120 |
| [hydraulic_tank](../env/hydraulic_tank.md) | Regulation | `q1` = 0.9 | `q1`: 1.5 |
| `multistage_extraction` | Regulation | `X5` = 0.4 | `X5`: 1 |
| `multistage_extraction_reactive` | Regulation | `XA5` = 1.95 | `XA5`: 2 |
| `nonsmooth_control` | Regulation | `X1` = 0 | `X1`: 2 |
| `photobioreactor` | Regulation | `c_q` = 0.04 | `c_q`: 0.11 |
| `polymerisation_reactor` | Regulation | `M` = 2 | `M`: 10 |
| [reactor_separator_recycle](../env/reactor_separator_recycle.md) | Finite-horizon regulation | `H_R` = `H_M` = `H_B` = 10 | each height: 4 |
| [surge_tank](../env/surge_tank.md) | Custom reward | minimize `sum dt*(%-pt of L outside [43, 58] + abs(dOP))` | physical %, m³/h and %/min |

The distillation default uses `F >= D = 100 kmol/hr`, nonnegative internal flows and target
`X0=0.5`. Its old target `0.9` was infeasible: at steady state, nonnegative bottoms require
`D*X0 <= F*X_feed <= 300*0.2 = 60`, so `X0 <= 0.6`. A steady solution at reflux ratio 5 and
feed approximately 250.07 kmol/hr achieves the revised target with every mole fraction in `[0,1]`.
User-supplied input limits and feed qualities are checked against these flow balances, including
the corrected vapour relation `V_below = V_above - (1-q)*F`.

Crystallization's default quality targets have a nominal near-target 30-minute trajectory;
see its [growth-only assumptions, input units and validated domain](../env/crystallisation.md).

The series-CSTR and reactive-extraction targets change without changing their process equations.
For series CSTR, `C2=97` has a steady solution at `[F, L, Tc1, Tc2] ≈
[0.00222673, 0.005, 305, 305]`, with state `[C1, T1, C2, T2] ≈
[97.0721, 298.6812, 97, 298.8379]`. For reactive extraction, `XA5=1.95` has a steady solution
at `[L, G] ≈ [9.84944, 100]`. Both use the existing input units and bounds; the earlier targets
of 60 and 0.5 were infeasible. Their checked 60-step episode durations are 10 and 60 model time
units, respectively. Steady-state feasibility alone does not establish successful learned tracking.

Photoproduction uses `c_q=0.04` in the equations' native concentration units and a 200-hour
episode. The old target of 120 exceeds the nutrient-derived production bound by orders of
magnitude. The [photoproduction page](../env/photoproduction.md) gives the corrected scales,
inventory bound and a constant-input trajectory reaching the target.

The heat-exchanger target remains `Tt8=320 K`. Its tube and annular volumes now use the documented
diameters: `π L Dt²/4`, `π L (Dm²-Dt²)/4`, and `π L (Ds²-Dm²)/4` per stage. Surface areas are
unchanged. Correcting the missing factor of one quarter makes all temperature derivatives four
times larger at the same state/input/parameters. Inputs `[Ft, Fs, Tt0, Ts9]` are mass flows in
kg/s and inlet temperatures in K. The nominal audit uses 60 steps over 10 s; geometry, thermal
capacities and nonnegative mass-flow limits are validated. These nominal parameters are not an
experimental calibration.

The five new defaults use 60-step horizons of 15 for batch/complex CSTR/hydraulic transfer, 60 for
disease, and 30 for reactor–separator control, in each model's documented time units. Hydraulic
transfer checks input limits against the conserved total volume. Reactor–separator checks its
horizon and action box against positive vessel/volatile-inventory bounds; a longer horizon can
be rejected even when a narrower, specifically analyzed trajectory might be feasible.

The biofilm default uses a 100-hour horizon with 60 transitions. Its oxygen partition coefficient
is derived from Henry solubility at 298.15 K, the fresh feed is deoxygenated by default, and its
state bounds follow nitrogen inventories and oxygen supply. This is a synthetic benchmark;
changes to these assumptions require their own bounds and validation.

All 19 controlled models now have registered spaces, initial states and tasks.
The autonomous invariant-batch and coupled-oscillator models have no default training task.

The setpoints are the final targets of the benchmark configurations. None of them is already satisfied by the default `x0`.

## Autonomous simulation examples

These two models are explicitly registered with `simulation_only=True`, available as
`get_model_spec(name).simulation_only`. Both now integrate with CasADi using zero-length action
vectors. They cannot define an action-dependent learning task without adding a justified actuator.
The audit reports this classification and its `--strict` training gate rejects these examples.

| Model | Dynamics and checks |
|---|---|
| [coupled_oscillator](../../src/pcgym/models/coupled_oscillators.py) | Periodic spring-mass chain; positions followed by momenta. Total momentum and Hamiltonian energy are conserved; trajectories match a matrix exponential and the stored JAX reference. |
| [invariant_batch](../../src/pcgym/models/invariant_batch.py) | Reversible reactions `A+B <-> C`, `A+C <-> D`; nonnegative concentrations conserve `A+C+2D` and `B+C+D`. |

For example, simulate the invariant batch without introducing a control objective:

```python
import numpy as np
from pcgym import make_env

initial = np.array([1.0, 0.8, 0.0, 0.0])
env = make_env({
    "model": "invariant_batch", "N": 60, "tsim": 3,
    "x0": initial,
    "a_space": {"low": np.empty(0), "high": np.empty(0)},
    "o_space": {"low": np.zeros(4), "high": np.full(4, np.inf)},
    "normalise_a": False, "normalise_o": False,
    "reward_states": [],
})
trajectory = env.simulate(initial, np.empty((60, 0)))
env.close()
```

Fresh JAX execution remains unvalidated in the selected Python 3.14 environment; comparison with
the stored oscillator trajectory is historical cross-backend evidence.
