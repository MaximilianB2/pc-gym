from dataclasses import dataclass

import numpy as np

from pcgym.models._base import jnp
from pcgym.models._registry import register_model


def _defaults():
    """Canonical a_space / o_space / x0 used when they are omitted from env_params."""
    return {
        "a_space": {"low": np.array([-1.0]), "high": np.array([1.0])},
        "o_space": {"low": np.array([-1.0, -1.0, -1.0]), "high": np.array([1.0, 1.0, 1.0])},
        "x0": np.array([0.8, 0.0, 0.0]),
    }


@register_model("nonsmooth_control", defaults=_defaults)
@dataclass(frozen=False, kw_only=True)
class nonsmooth_control:
    """
    Nonsmooth control model (Bang-Bang Control).

    Attributes:
        int_method (str): Integration method ('jax' or other)
        a_11, a_12, a_21, a_22 (float): System matrix coefficients
        b_1, b_2 (float): Input vector coefficients
    """

    int_method: str = "jax"
    a_11: float = 0
    a_12: float = 1
    a_21: float = -2
    a_22: float = -3
    b_1: float = 0
    b_2: float = 1

    def __call__(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Calculate the state derivatives for the nonsmooth control model.

        Args:
            x (np.ndarray): Current state [x1, x2]
            u (np.ndarray): Input [u]

        Returns:
            np.ndarray: State derivatives [dx1/dt, dx2/dt]
        """
        x1, x2 = x[0], x[1]
        dx1dt = self.a_11 * x1 + self.a_12 * x2 + self.b_1 * u
        dx2dt = self.a_21 * x1 + self.a_22 * x2 + self.b_2 * u

        ret = [dx1dt, dx2dt]

        return jnp.array(ret) if self.int_method == "jax" else np.array(ret)

    def info(self) -> dict:
        """
        Get model information.

        Returns:
            dict: Dictionary containing model parameters, states, inputs, and disturbances.
        """
        info = {
            "parameters": self.__dict__.copy(),
            "states": ["X1", "X2"],
            "inputs": ["U"],
            "disturbances": ["None"],
        }
        return info
