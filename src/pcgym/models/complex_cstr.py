from dataclasses import dataclass

import numpy as np

from pcgym.models._base import BaseModel, jnp
from pcgym.models._registry import Regulation, register_model


def _defaults():
    return {
        "a_space": {"low": np.array([295.0]), "high": np.array([302.0])},
        "o_space": {
            "low": np.array([0.0, 0.0, 0.0, 290.0, 0.0]),
            "high": np.array([1.0, 2.0, 2.0, 400.0, 2.0]),
        },
        "x0": np.array([0.8, 0.1, 0.1, 330.0, 0.25]),
    }


@register_model("complex_cstr", defaults=_defaults, task=Regulation(setpoint={"Cb": 0.25}))
@dataclass(frozen=False, kw_only=True)
class complex_cstr(BaseModel):
    # Reactor parameters
    q: float = 100  # Volumetric flow rate [m³/s]
    V: float = 100  # Reactor volume [m³]
    rho: float = 1000  # Density [kg/m³]
    C: float = 0.239  # Heat capacity [kJ/kg·K]

    # Reaction 1 parameters (A -> 2B)
    deltaHr1: float = -5e4  # Enthalpy of reaction 1 [kJ/kmol]
    EA1_over_R: float = 8750  # Activation energy/R for reaction 1 [K]
    k01: float = 7.2e10  # Pre-exponential factor for reaction 1 [1/s]

    # Reaction 2 parameters (B -> C)
    deltaHr2: float = -3e4  # Enthalpy of reaction 2 [kJ/kmol]
    EA2_over_R: float = 9000  # Activation energy/R for reaction 2 [K]
    k02: float = 1.0e10  # Pre-exponential factor for reaction 2 [1/s]

    # Heat transfer parameters
    UA: float = 5e4  # Heat transfer coefficient × area [kJ/(s·K)]

    # Operating parameters
    Ti: float = 350  # Inlet temperature [K]
    Caf: float = 1  # Feed concentration of A [kmol/m³]

    # Inherited parameters
    int_method: str = "jax"
    states: list = None
    inputs: list = None
    disturbances: list = None
    uncertainties: dict = None

    def __post_init__(self):
        self.states = ["Ca", "Cb", "Cc", "T"]
        self.inputs = ["Tc"]
        self.disturbances = ["Ti", "Caf"]

    def validate_initial_state(self, state):
        if np.any(np.asarray(state)[:3] < 0) or state[3] <= 0:
            raise ValueError("complex_cstr requires nonnegative concentrations and positive absolute temperature")

    def validate_input_bounds(self, low, high):
        positive = [self.V, self.rho, self.C, self.Ti]
        rates = [self.q, self.k01, self.k02, self.EA1_over_R, self.EA2_over_R, self.UA, self.Caf]
        if not np.isfinite(positive + rates).all() or min(positive) <= 0 or min(rates) < 0 or low[0] <= 0:
            raise ValueError("complex_cstr requires physical rate/capacity parameters and positive temperatures")

    def __call__(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        ca, cb, cc, T = x[0], x[1], x[2], x[3]
        xp = jnp if self.int_method == "jax" else np
        # Disturbance inputs are read into locals so calling the model never mutates it.
        if u.shape[0] == 1:
            Tc, Ti, Caf = u[0], self.Ti, self.Caf
        else:
            Tc, Ti, Caf = u[0], u[1], u[2]

        r1 = self.k01 * xp.exp(-self.EA1_over_R / T) * ca
        r2 = self.k02 * xp.exp(-self.EA2_over_R / T) * cb

        dca_dt = (self.q / self.V) * (Caf - ca) - r1
        dcb_dt = (self.q / self.V) * (0 - cb) + 2 * r1 - r2
        dcc_dt = (self.q / self.V) * (0 - cc) + r2

        heat_gen = (-self.deltaHr1 * r1) + (-self.deltaHr2 * r2)
        dTdt = (
            (self.q / self.V) * (Ti - T)
            + heat_gen / (self.rho * self.C)
            + (self.UA / (self.rho * self.C * self.V)) * (Tc - T)
        )

        ret = [dca_dt, dcb_dt, dcc_dt, dTdt]

        return jnp.array(ret) if self.int_method == "jax" else np.array(ret)
