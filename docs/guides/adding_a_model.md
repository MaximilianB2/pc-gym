# Adding a model

A new built-in model is a single file in `src/pcgym/models/`, plus one import line, a reference trajectory and a docs page. Use `src/pcgym/models/cstr.py` as the worked example.

## Checklist

Copy this into your PR description and tick every item.

- [ ] **One model file**: `src/pcgym/models/<name>.py`, registered with `@register_model("<name>", defaults=_defaults)` and imported in `src/pcgym/models/__init__.py`.
- [ ] **Units**: every parameter, state and input has its unit in a comment next to its definition.
- [ ] **Verified steady state as the default `x0`**: `_defaults()` returns `a_space`, `o_space` and `x0`. `x0` is a checked steady state (or a documented operating point), followed by one entry per canonical setpoint. `o_space` has the same layout (see [Observation layout](observations.md)).
- [ ] **Works under both integrators**: `__call__` only indexes `x[i]` / `u[i]`, never unpacks with `a, b = x`, never uses NumPy-only operations on the state, and never assigns to `self` (see below).
- [ ] **Reference trajectory**: `python scripts/gen_reference_trajectories.py <name>` has been run and `tests/reference_trajectories/<name>.npz` is committed. If the model has no defaults, add an `EXPLICIT` entry in the script. If it has defaults, add its `dt` to `DT`.
- [ ] **Docs page**: `docs/env/<name>.md` (equations, states, inputs, parameters with units, a suggested setpoint), linked from `mkdocs.yml`.
- [ ] **Tests pass**: `pytest` and `pytest -m slow` (the JAX reference check).

## The model file

```python
from dataclasses import dataclass

import numpy as np

from pcgym.models._base import BaseModel, jnp
from pcgym.models._registry import register_model


def _defaults():
    """Canonical a_space / o_space / x0 used when they are omitted from env_params."""
    return {
        "a_space": {"low": np.array([295.0]), "high": np.array([302.0])},
        "o_space": {"low": np.array([0.7, 300, 0.8]), "high": np.array([1.0, 350, 0.9])},
        "x0": np.array([0.8, 330, 0.8]),  # [Ca, T, Ca_SP]
    }


@register_model("cstr", defaults=_defaults)
@dataclass(frozen=False, kw_only=True)
class cstr(BaseModel):
    k0: float = 7.2e10  # pre-exponential factor [unit]
    ...

    def __post_init__(self):
        self.states = ["Ca", "T"]
        self.inputs = ["Tc"]
        self.disturbances = ["Ti", "Caf"]

    def __call__(self, x, u):
        ca, T = x[0], x[1]
        xp = jnp if self.int_method == "jax" else np
        ...
        ret = [dcadt, dTdt]
        return jnp.array(ret) if self.int_method == "jax" else np.array(ret)
```

`aliases=("other_name",)` lets the model also be found under alternative names.

## Backend rules

Every model is evaluated with NumPy arrays, JAX arrays and CasADi symbols (for the MPC oracle and the CasADi integrator). These three rules keep one set of equations working for all of them:

1. **Index, don't unpack.** Write `ca, T = x[0], x[1]`, not `ca, T = x`. CasADi symbols are not iterable.
2. **Use `xp` for maths functions.** Write `xp.exp(...)` with `xp = jnp if self.int_method == "jax" else np`, and use scalars like `u[0]` rather than the whole `u` vector in the equations.
3. **Keep `__call__` side-effect free.** Read disturbance inputs into local variables (`Ti = u[1]`), never onto `self`. The same model instance is shared between the environment, the integrator and the oracle.

`tests/models/test_casadi_backend.py` and the reference-trajectory tests check all registered models automatically.
