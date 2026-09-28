from dataclasses import dataclass

import numpy as np

from pcgym.models._base import jnp
from pcgym.models._registry import register_model


def _defaults():
    """Canonical a_space / o_space / x0 used when they are omitted from env_params."""
    return {
        "a_space": {"low": np.array([5.0, 10.0]), "high": np.array([500.0, 1000.0])},
        "o_space": {"low": np.array([0.0] * 10 + [0.3]), "high": np.array([1.0] * 10 + [0.4])},
        "x0": np.array([0.55, 0.3, 0.45, 0.25, 0.4, 0.2, 0.35, 0.15, 0.25, 0.1, 0.3]),
    }


@register_model("multistage_extraction", defaults=_defaults)
@dataclass(frozen=False, kw_only=True)
class multistage_extraction:
    """
    Multistage extraction model.

    Attributes:
        Vl (float): Liquid volume in each stage
        Vg (float): Gas volume in each stage
        m (float): Equilibrium constant
        Kla (float): Mass transfer capacity constant (1/hr)
        eq_exponent (float): Nonlinearity of the equilibrium relationship
        X0 (float): Feed concentration of liquid
        Y6 (float): Feed concentration of gas
        int_method (str): Integration method ('jax' or other)
    """

    Vl: float = 5
    Vg: float = 5
    m: float = 1
    Kla: float = 5
    eq_exponent: float = 2
    X0: float = 0.6
    Y6: float = 0.05
    int_method: str = "jax"

    def __call__(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Calculate the state derivatives for the multistage extraction model.

        Args:
            x (np.ndarray): Current state [X1, Y1, X2, Y2, X3, Y3, X4, Y4, X5, Y5]
            u (np.ndarray): Input [L, G] or [L, G, X0, Y6]

        Returns:
            np.ndarray: State derivatives
        """
        X1, Y1, X2, Y2, X3, Y3, X4, Y4, X5, Y5 = (
            x[0],
            x[1],
            x[2],
            x[3],
            x[4],
            x[5],
            x[6],
            x[7],
            x[8],
            x[9],
        )
        # Disturbance inputs are read into locals so calling the model never mutates it.
        if u.shape[0] == 2:
            L, G, X0, Y6 = u[0], u[1], self.X0, self.Y6
        else:
            L, G, X0, Y6 = u[0], u[1], u[2], u[3]

        X1_eq = (Y1**self.eq_exponent) / self.m
        X2_eq = (Y2**self.eq_exponent) / self.m
        X3_eq = (Y3**self.eq_exponent) / self.m
        X4_eq = (Y4**self.eq_exponent) / self.m
        X5_eq = (Y5**self.eq_exponent) / self.m

        Q1 = self.Kla * (X1 - X1_eq) * self.Vl
        Q2 = self.Kla * (X2 - X2_eq) * self.Vl
        Q3 = self.Kla * (X3 - X3_eq) * self.Vl
        Q4 = self.Kla * (X4 - X4_eq) * self.Vl
        Q5 = self.Kla * (X5 - X5_eq) * self.Vl

        ret = [
            (1 / self.Vl) * (L * (X0 - X1) - Q1),
            (1 / self.Vg) * (G * (Y2 - Y1) + Q1),
            (1 / self.Vl) * (L * (X1 - X2) - Q2),
            (1 / self.Vg) * (G * (Y3 - Y2) + Q2),
            (1 / self.Vl) * (L * (X2 - X3) - Q3),
            (1 / self.Vg) * (G * (Y4 - Y3) + Q3),
            (1 / self.Vl) * (L * (X3 - X4) - Q4),
            (1 / self.Vg) * (G * (Y5 - Y4) + Q4),
            (1 / self.Vl) * (L * (X4 - X5) - Q5),
            (1 / self.Vg) * (G * (Y6 - Y5) + Q5),
        ]

        return jnp.array(ret) if self.int_method == "jax" else np.array(ret)

    def info(self) -> dict:
        """
        Get model information.

        Returns:
            dict: Dictionary containing model parameters, states, inputs, and disturbances.
        """
        info = {
            "parameters": self.__dict__.copy(),
            "states": ["X1", "Y1", "X2", "Y2", "X3", "Y3", "X4", "Y4", "X5", "Y5"],
            "inputs": ["L", "G"],
            "disturbances": ["X0", "Y6"],
            "uncertaintes": [],
        }
        info["parameters"].pop("int_method", None)
        return info
