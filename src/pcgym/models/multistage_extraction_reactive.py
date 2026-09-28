from dataclasses import dataclass

import numpy as np

from pcgym.models._base import jnp
from pcgym.models._registry import register_model


def _defaults():
    """Canonical a_space / o_space / x0 used when they are omitted from env_params."""
    return {
        "a_space": {"low": np.array([5.0, 10.0]), "high": np.array([500.0, 1000.0])},
        "o_space": {"low": np.array([0.0] * 20 + [0.25]), "high": np.array([2.0] * 20 + [0.55])},
        "x0": np.array([1.0, 0.0, 1.0, 0.0] * 5 + [0.3]),
    }


@register_model("multistage_extraction_reactive", defaults=_defaults)
@dataclass(frozen=False, kw_only=True)
class multistage_extraction_reactive:
    """
    Multistage extraction with reactive components model.

    Attributes:
        Vl (float): Liquid volume in each stage
        Vg (float): Gas volume in each stage
        m (float): Equilibrium constant
        Kla (float): Mass transfer capacity constant (1/hr)
        k (float): Reaction equilibrium constant
        eq_exponent (float): Nonlinearity of the equilibrium relationship
        XA0 (float): Feed concentration of component A in liquid phase
        YA6, YB6, YC6 (float): Feed concentrations in gas phase
    """

    # Parameters
    Vl: float = 5.0  # Liquid volume in each stage
    Vg: float = 5.0  # Gas volume in each stage
    m: float = 1.0  # Equilibrium constant [-]
    Kla: float = 0.01  # Mass transfer capacity constant 1/hr
    k: float = 0.1  # Reaction equilibrium constant
    eq_exponent: float = 2.0  # Change the nonlinearity of the equilibrium relationship
    XA0: float = 2.00  # Feed concentration of component A in liquid phase
    YA6: float = 0.00  # Feed conc of component A in gas phase
    YB6: float = 2.00  # Feed conc of component B in gas phase
    YC6: float = 0.00  # Feed conc of component C in gas phase
    int_method: str = "jax"

    def __call__(self, x, u):
        """
        Calculate the state derivatives for the multistage extraction with reactive components.

        Args:
            x (np.ndarray): Current state
                [XA1, YA1, YB1, YC1, XA2, YA2, YB2, YC2, XA3, YA3, YB3, YC3,
                 XA4, YA4, YB4, YC4, XA5, YA5, YB5, YC5]
            u (np.ndarray): Input [L, G]

        Returns:
            np.ndarray: State derivatives
        """
        (
            XA1,
            YA1,
            YB1,
            YC1,
            XA2,
            YA2,
            YB2,
            YC2,
            XA3,
            YA3,
            YB3,
            YC3,
            XA4,
            YA4,
            YB4,
            YC4,
            XA5,
            YA5,
            YB5,
            YC5,
        ) = (x[i] for i in range(20))
        L, G = u[0], u[1]

        XA1_eq = (YA1**self.eq_exponent) / self.m
        XA2_eq = (YA2**self.eq_exponent) / self.m
        XA3_eq = (YA3**self.eq_exponent) / self.m
        XA4_eq = (YA4**self.eq_exponent) / self.m
        XA5_eq = (YA5**self.eq_exponent) / self.m

        Q1 = self.Kla * (XA1 - XA1_eq) * self.Vl
        Q2 = self.Kla * (XA2 - XA2_eq) * self.Vl
        Q3 = self.Kla * (XA3 - XA3_eq) * self.Vl
        Q4 = self.Kla * (XA4 - XA4_eq) * self.Vl
        Q5 = self.Kla * (XA5 - XA5_eq) * self.Vl

        r1 = self.k * YA1 * YB1
        r2 = self.k * YA2 * YB2
        r3 = self.k * YA3 * YB3
        r4 = self.k * YA4 * YB4
        r5 = self.k * YA5 * YB5

        ret = [
            (1 / self.Vl) * (L * (self.XA0 - XA1) - Q1),
            (1 / self.Vg) * (G * (YA2 - YA1) + Q1 - r1 * self.Vg),
            (1 / self.Vg) * (G * (YB2 - YB1) - r1 * self.Vg),
            (1 / self.Vg) * (G * (YC2 - YC1) + r1 * self.Vg),
            (1 / self.Vl) * (L * (XA1 - XA2) - Q2),
            (1 / self.Vg) * (G * (YA3 - YA2) + Q2 - r2 * self.Vg),
            (1 / self.Vg) * (G * (YB3 - YB2) - r2 * self.Vg),
            (1 / self.Vg) * (G * (YC3 - YC2) + r2 * self.Vg),
            (1 / self.Vl) * (L * (XA2 - XA3) - Q3),
            (1 / self.Vg) * (G * (YA4 - YA3) + Q3 - r3 * self.Vg),
            (1 / self.Vg) * (G * (YB4 - YB3) - r3 * self.Vg),
            (1 / self.Vg) * (G * (YC4 - YC3) + r3 * self.Vg),
            (1 / self.Vl) * (L * (XA3 - XA4) - Q4),
            (1 / self.Vg) * (G * (YA5 - YA4) + Q4 - r4 * self.Vg),
            (1 / self.Vg) * (G * (YB5 - YB4) - r4 * self.Vg),
            (1 / self.Vg) * (G * (YC5 - YC4) + r4 * self.Vg),
            (1 / self.Vl) * (L * (XA4 - XA5) - Q5),
            (1 / self.Vg) * (G * (self.YA6 - YA5) + Q5 - r5 * self.Vg),
            (1 / self.Vg) * (G * (self.YB6 - YB5) - r5 * self.Vg),
            (1 / self.Vg) * (G * (self.YC6 - YC5) + r5 * self.Vg),
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
            "states": [
                "XA1",
                "YA1",
                "YB1",
                "YC1",
                "XA2",
                "YA2",
                "YB2",
                "YC2",
                "XA3",
                "YA3",
                "YB3",
                "YC3",
                "XA4",
                "YA4",
                "YB4",
                "YC4",
                "XA5",
                "YA5",
                "YB5",
                "YC5",
            ],
            "inputs": ["L", "G"],
            "disturbances": [],
        }
        return info
