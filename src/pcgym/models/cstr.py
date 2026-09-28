from dataclasses import dataclass

import numpy as np

from pcgym.models._base import BaseModel, jnp
from pcgym.models._registry import register_model


def _defaults():
    """Canonical a_space / o_space / x0 used when they are omitted from env_params."""
    return {
        "a_space": {"low": np.array([295.0]), "high": np.array([302.0])},
        "o_space": {"low": np.array([0.7, 300, 0.8]), "high": np.array([1.0, 350, 0.9])},
        "x0": np.array([0.8, 330, 0.8]),
    }


@register_model("cstr", defaults=_defaults)
@dataclass(frozen=False, kw_only=True)
class cstr(BaseModel):
    q: float = 100
    V: float = 100
    rho: float = 1000
    C: float = 0.239
    deltaHr: float = -5e4
    EA_over_R: float = 8750
    k0: float = 7.2e10
    UA: float = 5e4
    Ti: float = 350
    Caf: float = 1
    int_method: str = "jax"
    states: list = None
    inputs: list = None
    disturbances: list = None
    uncertainties: dict = None

    def __post_init__(self):
        self.states = ["Ca", "T"]
        self.inputs = ["Tc"]
        self.disturbances = ["Ti", "Caf"]

    def __call__(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        ca, T = x[0], x[1]
        xp = jnp if self.int_method == "jax" else np
        # Disturbance inputs are read into locals so calling the model never mutates it.
        if u.shape[0] == 1:
            Tc, Ti, Caf = u[0], self.Ti, self.Caf
        else:
            Tc, Ti, Caf = u[0], u[1], u[2]
        rA = self.k0 * xp.exp(-self.EA_over_R / T) * ca
        dcadt = self.q / self.V * (Caf - ca) - rA
        dTdt = (
            self.q / self.V * (Ti - T)
            + ((-self.deltaHr) * rA) * (1 / (self.rho * self.C))
            + self.UA * (Tc - T) * (1 / (self.rho * self.C * self.V))
        )

        ret = [dcadt, dTdt]

        return jnp.array(ret) if self.int_method == "jax" else np.array(ret)
