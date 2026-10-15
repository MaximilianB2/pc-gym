from dataclasses import dataclass

import numpy as np

from pcgym.models._base import BaseModel, jnp
from pcgym.models._registry import Batch, register_model


def _defaults():
    return {
        "a_space": {"low": np.array([290.0]), "high": np.array([310.0])},
        # 2 Ca + Cb + Cc = 2; total available reaction heat raises T by at most 1 K.
        "o_space": {"low": np.array([0.0, 0.0, 0.0, 290.0]), "high": np.array([1.0, 2.0, 2.0, 311.0])},
        "x0": np.array([1.0, 0.0, 0.0, 300.0]),
    }


# -------------------------------------------------
# 4. Batch Reactor (Exothermic consecutive reactions)
# -------------------------------------------------
@register_model("batch", defaults=_defaults, task=Batch(reward_states=("Cb",)))
@dataclass(frozen=False, kw_only=True)
class batch(BaseModel):
    """A -> 2 B -> 2 C batch benchmark; time in seconds, temperature in K.

    Concentrations are kmol/m3, reaction enthalpies kJ/kmol, and UA kJ/(s K).
    The default task maximizes intermediate B at 15 s and exposes time remaining.
    """

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

    def validate_initial_state(self, state):
        if np.any(np.asarray(state)[:3] < 0) or state[3] <= 0:
            raise ValueError("batch requires nonnegative concentrations and positive absolute temperature")

    def validate_input_bounds(self, low, high):
        positive = [self.R, self.rho, self.Cp, self.V]
        rates = [self.k01, self.k02, self.EA1, self.EA2, self.UA]
        if not np.isfinite(positive + rates).all() or min(positive) <= 0 or min(rates) < 0 or low[0] <= 0:
            raise ValueError("batch requires physical rate/capacity parameters and positive coolant temperatures")

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
