# Observation layout

pc-gym uses the same vector layout for observations and its internal physical state. Its entries always come in this order:

| Block | Kind | Present when | Named |
|---|---|---|---|
| Model states | `state` | always | the model's state names, e.g. `Ca`, `T` |
| Setpoints | `setpoint` | `SP` is given | `<state>_SP`, one per entry in `SP` |
| Disturbances | `disturbance` | `disturbances` is given | the disturbance name, e.g. `Ti` |
| Uncertain parameters | `uncertainty` | `uncertainty_percentages` or `empirical_distribution` is given | the parameter name, e.g. `k0` |
| Time remaining | `time` | the task has a finite objective horizon | `time_remaining` |

## Precision and returned data

`reset()` and `step()` return independent **float32** observation arrays, matching
`env.observation_space.dtype`, whether `normalise_o` is enabled or disabled. The simulation
state (`env.state`) and physical measurement (`env.obs`) use **float64**. Integer initial
states are converted to float64 so integration can retain fractional changes.

`env.observation_space_base` stores physical bounds in float64, including appended disturbance
and uncertainty bounds. Normalization uses those bounds before casting the observation to
float32. Supply physical bounds at their intended precision; converting an already rounded
float32 input to float64 cannot restore lost digits. Observations are not clipped to their
space: a physical excursion can still fall outside the declared physical-state bounds. Gaussian
noise and normally distributed uncertain entries have unbounded **public** bounds; their finite
scaling bounds stay in `observation_space_base`. Normalized noisy values may therefore exceed
`[-1, 1]` without violating the public space.

`info["obs"]` retains the full measurement in float64, after any measurement noise and before
partial-observation masking. It is normalized only when `normalise_o=True`. This allows
evaluation to recover physical units without losing the hidden state entries. The returned
observation and information dictionary are snapshots: later steps or resets do not change
them, and editing them does not change the environment.

This corrects earlier versions that declared float32 spaces but usually returned float64
observations. Code needing physical simulation precision should use `env.state`, or use
`info["obs"]` with the configured normalization taken into account.

## What goes in `x0` and `o_space`

`x0` and `o_space` cover **only the states and setpoints**. The disturbance and uncertain-parameter entries are appended automatically, and their bounds come from `disturbance_bounds` and `uncertainty_bounds`. Finite-horizon tasks append time remaining automatically, with physical bounds `[0, 1]`. `env.Nx` counts every observation entry, and `env.NUn` counts the uncertain model parameters; initial-state uncertainty adds no entries.

For a CSTR tracking `Ca`:

```python
env_params = {
    "model": "cstr",
    "SP": {"Ca": [0.85] * 100},
    "x0": np.array([0.8, 330, 0.8]),   # [Ca, T, Ca_SP]
    "o_space": {"low": np.array([0.7, 300, 0.8]), "high": np.array([1, 350, 0.9])},
    ...
}
```

If `x0` or `o_space` has the wrong length, `make_env` raises a `ValueError` that lists the expected order.

## Schedule timing

Setpoints and disturbances are indexed by state time. Let `dt = tsim / N`; reset returns
`x_0` at time zero, and each step advances the state by `dt`.

| Event | Disturbance applied during integration | Returned observation | Tracking reward target |
|---|---|---|---|
| `reset()` | None | `x_0`, `SP[0]`, `d[0]` | Initial reward is zero |
| `step(u_t)` | `d[t]`, held throughout the interval | `x_(t+1)`, `SP[t+1]`, `d[t+1]` | `SP[t+1]` |

The setpoint entries of `x0` still reserve positions in the layout, but reset takes their
active values from `SP[0]`. For example, the configuration above returns `Ca_SP = 0.85` at
reset, even though the placeholder in `x0` is `0.8`. The supplied configuration is preserved.

Provide `N + 1` schedule values to specify every observed state, including the final `x_N`.
An `N`-value schedule holds its last value at `x_N`; any shorter nonempty schedule likewise
holds its final value. A one-element sequence is a constant schedule. The final disturbance
`d[N]` is observable but is not applied within an episode of `N` transitions.

Setpoint slots follow `SP` dictionary order. Active disturbance slots follow the model's
disturbance order; omitted disturbances retain their model values. Custom rewards receive
the measurement at `t+1` and the physical input vector actually applied over interval `t`,
including `d[t]`.

The MPC oracle uses the same convention: forecast stage `k` at time `t` pairs its predicted
state with `SP[t+k]` and its outgoing interval with `d[t+k]`. Each stage has its own forecast,
including the terminal target, and values beyond the supplied schedule hold the last value.

This changes earlier behavior: disturbances were applied one interval early, reset could show
a stale target from `x0`, and step observations showed a target one index behind the reward.
Policies and reference trajectories using changing schedules should be reevaluated under
the corrected timing.

## Episode endings and time remaining

Regulation tasks run for exactly `N` transitions unless a configured physical constraint ends
the episode earlier. Their time limit returns `terminated=False, truncated=True`, allowing RL
algorithms to bootstrap from the final observation.

A built-in batch objective (nonempty `reward_states`, without a setpoint or custom reward)
has a finite horizon. Its final transition pays the yield reward and returns
`terminated=True, truncated=False`. Its last observation entry is `time_remaining = 1 - t / N`:
one at reset and zero at the final state. With `normalise_o=True`, these endpoints become
one and minus one. The time entry is unaffected by measurement noise or partial observations.

Set `finite_horizon=True` for a custom reward whose objective ends at `N`. Other custom
rewards and explicit empty-reward simulation configurations keep continuing-task semantics
by default. A built-in terminal batch reward cannot use `finite_horizon=False`. A configured
constraint termination takes precedence over truncation, including at the final step; early
constraint termination does not pay a batch yield that is defined only at `N`.

Call `reset()` before the first step and after either terminal flag is true. Otherwise `step()`
raises Gymnasium's `ResetNeeded`. Reset clears the previous episode's action and custom info.
Vectorized wrappers may reset automatically; use their `terminal_observation` for the completed
episode. This changes the observation shape and bootstrapping behavior of older batch policies.

## Measurement noise and uncertainty

With `noise=True`, supply `noise_percentage` as a finite nonnegative number or a dictionary
of model-state names to such numbers. `0.01` means a standard deviation of 1% of the absolute
state value. Noise is applied at both reset and step, to the measurement only. Missing entries
in a noise dictionary get zero noise. Setpoints, disturbances, parameters and time remain
noise-free. Built-in rewards use the true process state; custom rewards receive the measurement.

Gaussian measurement noise has unbounded support, so the affected public observation entries
use `(-inf, inf)`. Their physical normalization bounds remain finite. For partial observations,
hidden model-state entries are always zero and have public bounds `[0, 0]`, including when
physical units would normally exclude zero. An empty `partial_observation` list hides every
model state. Full measurements remain available in `info["obs"]`.

Initial-state uncertainty uses model-state indices, excluding setpoint placeholders:

```python
env_params.update({
    "uncertainty_percentages": {"x0": {1: 0.01}, "k0": 0.1},
    "distribution": "uniform",
    "uncertainty_bounds": {"low": np.array([6.48e10]), "high": np.array([7.92e10])},
})
```

For the default CSTR, this varies the initial temperature by ±1% and `k0` by ±10%. A sequence
under `x0` is also accepted, starting at model-state index zero. Uniform sampling is the default;
normal sampling uses standard deviation `percentage * abs(nominal)`, including negative nominal
values. Normally distributed entries use unbounded public bounds and are never clipped to the
supplied normalization range. Model-specific physical restrictions still need a suitable
distribution and operating-region validation.

Alternatively, `empirical_distribution` maps parameter names to nonempty finite sample vectors.
Its optional `x0` entry maps state indices to sample vectors. Each quantity is sampled independently.
Choose percentage or empirical uncertainty, not both. Parameter bounds must contain the complete
uniform or empirical support and have one entry per parameter, in configuration order. `o_space`
must contain bounded initial-state uncertainty. Initial-state-only uncertainty needs no
`uncertainty_bounds`. Normalized configurations require finite, positive scaling widths.

`reset(seed=...)` seeds initial-state uncertainty, parameter sampling and measurement noise.
Repeating a seed and action sequence reproduces the trajectory. Calling `reset()` without a
seed continues the generator. Adding noise at reset changes random-number consumption compared
with older versions, so old seeded noisy trajectories need regeneration.

## Inspecting the layout

`env.observation_info()` returns one `(name, kind)` pair per index:

```python
env = make_env(env_params)
env.observation_info()
# [('Ca', 'state'), ('T', 'state'), ('Ca_SP', 'setpoint')]
```

## Building vectors by name

`env.build_obs()` assembles a correctly ordered vector from named parts, so you never have to count indices by hand:

```python
obs = env.build_obs(
    states={"Ca": 0.8, "T": 330},
    setpoints={"Ca": 0.85},        # keyed by the tracked state
    disturbances={"Ti": 350},      # only if disturbances are active
)
```

- Each part can be a dict keyed by name, or a sequence in layout order.
- Setpoints, disturbances and uncertain parameters default to the first setpoint value, the first disturbance value and the nominal parameter value.
- For finite-horizon tasks, `time_remaining` defaults to `1.0`; supply the fraction explicitly when querying a policy at a later stage.
- `normalise=True` scales the vector the same way the environment does when `normalise_o` is set, which is useful for querying a trained policy at a chosen state.
- The helper returns float64 and does not apply partial-observation masking. For a fully observed policy, cast its result to `env.observation_space.dtype`; use the unnormalized result when preparing `x0`.
