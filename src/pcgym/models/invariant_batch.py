from dataclasses import dataclass

import numpy as np

from pcgym.models._base import BaseModel, jnp
from pcgym.models._registry import register_model


# -------------------------------------------------
# 5. Batch Reactor with Reaction Invariants
# -------------------------------------------------
@register_model("invariant_batch", simulation_only=True)
@dataclass(frozen=False, kw_only=True)
class invariant_batch(BaseModel):
    """Autonomous A+B <-> C, A+C <-> D reaction example.

    Nonnegative concentrations preserve A+C+2D and B+C+D. There is no
    manipulated input or default training task.
    """

    k1f: float = 55.0
    k1r: float = 1.0
    k2f: float = 2.0
    k2r: float = 1.0
    int_method: str = "jax"
    states: list = None
    inputs: list = None
    disturbances: list = None
    uncertainties: dict = None

    def __post_init__(self):
        rates = [self.k1f, self.k1r, self.k2f, self.k2r]
        if not np.isfinite(rates).all() or min(rates) < 0:
            raise ValueError("invariant_batch requires finite nonnegative reaction rates")
        self.states = ["xA", "xB", "xC", "xD"]
        self.inputs = []
        self.disturbances = []

    def validate_initial_state(self, state):
        values = np.asarray(state, dtype=float)
        if values.shape != (4,) or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("invariant_batch requires four finite nonnegative concentrations")

    def validate_input_bounds(self, low, high):
        self.__post_init__()

    def __call__(self, x, u=None):
        xA, xB, xC, xD = x[0], x[1], x[2], x[3]
        dxAdt = -(self.k1f * xA * xB - self.k1r * xC) - (self.k2f * xA * xC - self.k2r * xD)
        dxBdt = -(self.k1f * xA * xB - self.k1r * xC)
        dxCdt = (self.k1f * xA * xB - self.k1r * xC) - (self.k2f * xA * xC - self.k2r * xD)
        dxDdt = self.k2f * xA * xC - self.k2r * xD

        ret = [dxAdt, dxBdt, dxCdt, dxDdt]

        return jnp.array(ret) if self.int_method == "jax" else np.array(ret)
