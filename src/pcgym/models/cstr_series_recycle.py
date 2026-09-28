from dataclasses import dataclass

import numpy as np

from pcgym.models._base import jnp
from pcgym.models._registry import register_model


def _defaults():
    """Canonical a_space / o_space / x0 used when they are omitted from env_params."""
    return {
        "a_space": {
            "low": np.array([0.001, 0.001, 290.0, 290.0]),
            "high": np.array([0.01, 0.01, 320.0, 320.0]),
        },
        "o_space": {
            "low": np.array([0.0, 290.0, 0.0, 290.0, 55.0]),
            "high": np.array([100.0, 400.0, 100.0, 400.0, 85.0]),
        },
        "x0": np.array([90.0, 310.0, 85.0, 310.0, 80.0]),
    }


@register_model("cstr_series_recycle", defaults=_defaults)
@dataclass(frozen=False, kw_only=True)
class cstr_series_recycle:
    """
    CSTR series with recycle model.

    Attributes:
        C_O (float): Initial concentration (mol/m3)
        T_O (float): Initial temperature (K)
        V1, V2 (float): Reactor volumes (m3)
        U1A1, U2A2 (float): Heat transfer coefficients times areas (kJ/s*K)
        rho (float): Density (kg/m3)
        cp (float): Heat capacity (kJ/kg*K)
        k (float): Reaction rate constant (s-1)
        E (float): Activation energy (kJ/mol)
        deltaH (float): Heat of reaction (kJ/mol)
        R (float): Gas constant (kJ/mol K)
    """

    # Parameters
    C_O: float = 97.35  # mol/m3
    T_O: float = 298  # K
    V1: float = 1e-3  # m3
    V2: float = 2e-3  # m3
    U1A1: float = 0.461  # kJ/s*K
    U2A2: float = 0.732  # kJ/s*K
    rho: float = 1.05e3  # kg/m3
    cp: float = 3.766  # kJ/kg*K
    k: float = 3.118e5  # s-1
    E: float = 46.14  # kJ/mol
    deltaH: float = 58.41  # kJ/mol
    R: float = 8.3145e-3  # kJ/mol K
    int_method: str = "jax"

    def __call__(self, x, u):
        """
        Calculate the state derivatives for the CSTR series with recycle.

        Args:
            x (np.ndarray): Current state [C1, T1, C2, T2]
            u (np.ndarray): Input [F, L, Tc1, Tc2]

        Returns:
            np.ndarray: State derivatives
        """
        C1, T1, C2, T2 = x[0], x[1], x[2], x[3]
        F, L, Tc1, Tc2 = u[0], u[1], u[2], u[3]
        xp = jnp if self.int_method == "jax" else np

        ret = [
            (self.C_O / self.V1) * F
            + (1 / self.V1) * L * C2
            - (1 / self.V1) * (F + L) * C1
            - self.k * C1 * xp.exp((-self.E / (self.R * T1))),
            (self.T_O / self.V1) * F
            + (1 / self.V1) * L * T2
            - ((self.U1A1) / (self.V1 * self.rho * self.cp)) * (T1 - Tc1)
            - (1 / self.V1) * (F + L) * T1
            + ((self.k * (-self.deltaH)) / (self.rho * self.cp)) * C1 * xp.exp((-self.E / (self.R * T1))),
            (1 / self.V2) * (F + L) * (C1 - C2) - self.k * C2 * xp.exp((-self.E / (self.R * T2))),
            (1 / self.V2) * (F + L) * (T1 - T2)
            - ((self.U2A2) / (self.V2 * self.rho * self.cp)) * (T2 - Tc2)
            + ((self.k * (-self.deltaH)) / (self.rho * self.cp)) * C2 * xp.exp((-self.E / (self.R * T2))),
        ]

        return jnp.array(ret) if self.int_method == "jax" else np.array(ret)

    def info(self):
        """
        Get model information.

        Returns:
            dict: Dictionary containing model parameters, states, inputs, and disturbances.
        """
        # Return a dictionary with the model information
        info = {
            "parameters": self.__dict__.copy(),
            "states": ["C1", "T1", "C2", "T2"],
            "inputs": ["F", "L", "Tc1", "Tc2"],
            "disturbances": [],
        }
        return info
