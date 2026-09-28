from dataclasses import dataclass

import numpy as np

from pcgym.models._base import jnp
from pcgym.models._registry import register_model


def _defaults():
    """Canonical a_space / o_space / x0 used when they are omitted from env_params."""
    return {
        "a_space": {"low": np.array([0.0]), "high": np.array([1.0])},
        "o_space": {"low": np.array([0.0, 0.0]), "high": np.array([1.0, 1.0])},
        "x0": np.array([0.1, 0.4]),
    }


@register_model("first_order_system", defaults=_defaults)
@dataclass(frozen=False, kw_only=True)
class first_order_system:
    """
    First-order system model.

    Attributes:
        K (float): Gain
        tau (float): Time constant
        int_method (str): Integration method ('jax' or other)
    """

    K: float = 1
    tau: float = 0.5
    int_method: str = "jax"

    def __call__(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Calculate the state derivative for the first-order system.

        Args:
            x (np.ndarray): Current state [x]
            u (np.ndarray): Input [u]

        Returns:
            np.ndarray: State derivative [dx/dt]
        """
        x = x[0]
        u = u[0]
        dxdt = (self.K * u - x) * 1 / self.tau

        ret = [dxdt]

        return jnp.array(ret) if self.int_method == "jax" else np.array(ret)

    def info(self) -> dict:
        """
        Get model information.

        Returns:
            dict: Dictionary containing model parameters, states, inputs, and disturbances.
        """
        info = {
            "parameters": self.__dict__.copy(),
            "states": ["x"],
            "inputs": ["u"],
            "disturbances": ["None"],
        }
        info["parameters"].pop("int_method", None)
        return info
