# Observation layout

pc-gym stores everything the agent observes in one vector, which is also the environment's internal state. Its entries always come in this order:

| Block | Kind | Present when | Named |
|---|---|---|---|
| Model states | `state` | always | the model's state names, e.g. `Ca`, `T` |
| Setpoints | `setpoint` | `SP` is given | `<state>_SP`, one per entry in `SP` |
| Disturbances | `disturbance` | `disturbances` is given | the disturbance name, e.g. `Ti` |
| Uncertain parameters | `uncertainty` | `uncertainty_percentages` or `empirical_distribution` is given | the parameter name, e.g. `k0` |

## What goes in `x0` and `o_space`

`x0` and `o_space` cover **only the states and setpoints**. The disturbance and uncertain-parameter entries are appended automatically, and their bounds come from `disturbance_bounds` and `uncertainty_bounds`.

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
- `normalise=True` scales the vector the same way the environment does when `normalise_o` is set, which is useful for querying a trained policy at a chosen state.
