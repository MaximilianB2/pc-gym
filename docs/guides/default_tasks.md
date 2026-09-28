# Default tasks

Every built-in model with default spaces also has a **default task**, so a minimal configuration is a complete, meaningful environment:

```python
env = make_env({"model": "cstr", "N": 60, "tsim": 26})
```

The default task is used only when **no reward is configured**, meaning none of `SP`, `custom_reward` or `reward_states` is given. An explicitly configured reward is never modified.

## Regulation

The tracked state(s) are held at a constant setpoint for the whole episode. The reward at each step is

```
r_t = - sum_k (x_k,t - SP_k)^2 / (o_space_high_k - o_space_low_k)^2
```

Each squared tracking error is divided by the squared width of that state's `o_space` range, which makes returns comparable across models and papers. Passing your own `r_scale` replaces this scaling.

When the default task is used:

- `SP` is set to the constant setpoint.
- `x0`'s setpoint entries are set to the setpoint values. If `x0` has only the model states, the setpoint entries are appended.
- If `o_space` has only the model states, the tracked states' bounds are reused for the setpoint entries.

## Batch

The reward is zero until the final step, which pays the value of the reward states (maximised).

## Default task per model

| Model | Task | Setpoint | Normalising range |
|---|---|---|---|
| `batch` | Batch | maximise `Cb` at the final step | – |
| `biofilm_reactor` | Regulation | `S2_A` = 2 | `S2_A`: 10 |
| `crystallization` | Regulation | `CV` = 1, `Ln` = 15 | `CV`: 2, `Ln`: 20 |
| `cstr` | Regulation | `Ca` = 0.9 | `Ca`: 0.3 |
| `cstr_series_recycle` | Regulation | `C2` = 60 | `C2`: 100 |
| `distillation_column` | Regulation | `X0` = 0.9 | `X0`: 1 |
| `first_order_system` | Regulation | `x` = 0.7 | `x`: 1 |
| `four_tank` | Regulation | `h3` = 0.1, `h4` = 0.3 | `h3`: 0.6, `h4`: 0.6 |
| `heat_exchanger` | Regulation | `Tt8` = 320 | `Tt8`: 120 |
| `multistage_extraction` | Regulation | `X5` = 0.4 | `X5`: 1 |
| `multistage_extraction_reactive` | Regulation | `XA5` = 0.5 | `XA5`: 2 |
| `nonsmooth_control` | Regulation | `X1` = 0 | `X1`: 2 |
| `photobioreactor` | Regulation | `c_q` = 120 | `c_q`: 300 |
| `polymerisation_reactor` | Regulation | `M` = 3 | `M`: 10 |

Models not listed have no default task. For those, configure `SP`, `custom_reward` or `reward_states`.

The setpoints are the final targets of the benchmark configurations. None of them is already satisfied by the default `x0`.
