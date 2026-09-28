from dataclasses import dataclass

import numpy as np

from pcgym.models._base import jnp
from pcgym.models._registry import Regulation, register_model


def _defaults():
    """Canonical a_space / o_space / x0 used when they are omitted from env_params."""
    return {
        "a_space": {"low": np.array([0.1, 0.1]), "high": np.array([10.0, 10.0])},
        "o_space": {"low": np.array([0.0] * 6), "high": np.array([0.6] * 6)},
        "x0": np.array([0.141, 0.112, 0.072, 0.42, 0.5, 0.2]),
    }


@register_model("four_tank", defaults=_defaults, task=Regulation(setpoint={"h3": 0.1, "h4": 0.3}))
@dataclass(frozen=False, kw_only=True)
class four_tank:
    """
    Four-tank system model.

    Attributes:
        g (float): Acceleration due to gravity (m/s2)
        gamma_1, gamma_2 (float): Fraction bypassed by valves
        k1, k2 (float): Pump gains (m3/Volts S)
        a1, a2, a3, a4 (float): Cross-sectional areas of outlets (m2)
        A1, A2, A3, A4 (float): Cross-sectional areas of tanks (m2)
        int_method (str): Integration method ('jax' or other)
    """

    # Parameters
    g: float = 9.81  # Acceleration due to gravity [m/s2]
    gamma_1: float = 0.2  # Fraction bypassed by valve to tank 1 [-]
    gamma_2: float = 0.2  # Fraction bypassed by valve to tank 2 [-]
    k1: float = 0.00085  # 1st pump gain [m3/Volts S]
    k2: float = 0.00095  # 2nd pump gain [m3/Volts S]
    a1: float = 0.0035  # Cross sectional area of outlet of tank 1 [m2]
    a2: float = 0.0030  # Cross sectional area of outlet of tank 2 [m2]
    a3: float = 0.0020  # Cross sectional area of outlet of tank 3 [m2]
    a4: float = 0.0025  # Cross sectional area of outlet of tank 4 [m2]
    A1: float = 1  # Cross sectional area of tank 1 [m2]
    A2: float = 1  # Cross sectional area of tank 2 [m2]
    A3: float = 1  # Cross sectional area of tank 3 [m2]
    A4: float = 1  # Cross sectional area of tank 4 [m2]
    int_method: str = "jax"

    def __call__(self, x, u):
        """
        Calculate the state derivatives for the four-tank system.

        Args:
            x (np.ndarray): Current state [h1, h2, h3, h4]
            u (np.ndarray): Input [v1, v2]

        Returns:
            np.ndarray: State derivatives
        """
        h1, h2, h3, h4 = x[0], x[1], x[2], x[3]
        v1, v2 = u[0], u[1]
        xp = jnp if self.int_method == "jax" else np

        ret = [
            (-self.a1 / self.A1) * xp.sqrt(2 * self.g * h1)
            + (self.a3 / self.A1) * xp.sqrt(2 * self.g * h3)
            + ((self.gamma_1 * self.k1) / (self.A1)) * v1,
            (-self.a2 / self.A2) * xp.sqrt(2 * self.g * h2)
            + (self.a4 / self.A2) * xp.sqrt(2 * self.g * h4)
            + ((self.gamma_2 * self.k2) / (self.A2)) * v2,
            (-self.a3 / self.A3) * xp.sqrt(2 * self.g * h3) + (((1 - self.gamma_2) * self.k2) / (self.A3)) * v2,
            (-self.a4 / self.A4) * xp.sqrt(2 * self.g * h4) + (((1 - self.gamma_1) * self.k1) / (self.A4)) * v1,
        ]

        return jnp.array(ret) if self.int_method == "jax" else np.array(ret)

    def info(self):
        """
        Get model information.

        Returns:
            dict: Dictionary containing model parameters, states, inputs, and disturbances.
        """
        info = {
            "parameters": self.__dict__.copy(),
            "states": ["h1", "h2", "h3", "h4"],
            "inputs": ["v1", "v2"],
            "disturbances": ["None"],
        }
        info["parameters"].pop(
            "int_method", None
        )  # Remove 'int_method' from the dictionary since it is not a parameter of the model
        return info
