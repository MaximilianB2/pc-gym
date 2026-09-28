from dataclasses import dataclass

import numpy as np

from pcgym.models._base import BaseModel, jnp
from pcgym.models._registry import register_model


# -------------------------------------------------
# 4. Batch Reactor (Exothermic consecutive reactions)
# -------------------------------------------------
@register_model("batch")
@dataclass(frozen=False, kw_only=True)
class batch(BaseModel):
    k01: float = 1.0
    k02: float = 0.5
    EA1: float = 5000
    EA2: float = 6000
    R: float = 8.314
    dH1: float = -1000
    dH2: float = -1500
    rho: float = 1000
    Cp: float = 4.0
    UA: float = 100
    V: float = 1.0
    int_method: str = "jax"
    states: list = None
    inputs: list = None
    disturbances: list = None
    uncertainties: dict = None

    def __post_init__(self):
        self.states = ["Ca", "Cb", "Cc", "T"]
        self.inputs = ["Tc"]
        self.disturbances = []

    def __call__(self, x, u):
        CA, CB, _CC, T = x[0], x[1], x[2], x[3]
        Tc = u[0]
        xp = jnp if self.int_method == "jax" else np
        r1 = self.k01 * xp.exp(-self.EA1 / (self.R * T)) * CA
        r2 = self.k02 * xp.exp(-self.EA2 / (self.R * T)) * CB
        dCAdt = -r1
        dCBdt = 2 * r1 - r2
        dCCdt = r2
        dTdt = -(self.dH1 * r1 + self.dH2 * r2) / (self.rho * self.Cp) + self.UA / (self.rho * self.Cp * self.V) * (
            Tc - T
        )

        ret = [dCAdt, dCBdt, dCCdt, dTdt]

        return jnp.array(ret) if self.int_method == "jax" else np.array(ret)
