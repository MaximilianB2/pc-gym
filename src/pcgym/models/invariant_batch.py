from dataclasses import dataclass

import numpy as np

from pcgym.models._base import BaseModel, jnp
from pcgym.models._registry import register_model


# -------------------------------------------------
# 5. Batch Reactor with Reaction Invariants
# -------------------------------------------------
@register_model("invariant_batch")
@dataclass(frozen=False, kw_only=True)
class invariant_batch(BaseModel):
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
        self.states = ["xA", "xB", "xC", "xD"]
        self.inputs = []
        self.disturbances = []

    def __call__(self, x, u=None):
        xA, xB, xC, xD = x[0], x[1], x[2], x[3]
        dxAdt = -(self.k1f * xA * xB - self.k1r * xC) - (self.k2f * xA * xC - self.k2r * xD)
        dxBdt = -(self.k1f * xA * xB - self.k1r * xC)
        dxCdt = (self.k1f * xA * xB - self.k1r * xC) - (self.k2f * xA * xC - self.k2r * xD)
        dxDdt = self.k2f * xA * xC - self.k2r * xD

        ret = [dxAdt, dxBdt, dxCdt, dxDdt]

        return jnp.array(ret) if self.int_method == "jax" else np.array(ret)
