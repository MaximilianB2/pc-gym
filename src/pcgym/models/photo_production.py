from dataclasses import dataclass

import numpy as np

from pcgym.models._base import jnp
from pcgym.models._registry import register_model


def _defaults():
    """Canonical a_space / o_space / x0 used when they are omitted from env_params."""
    return {
        "a_space": {"low": np.array([100.0, 0.0]), "high": np.array([400.0, 10.0])},
        "o_space": {
            "low": np.array([0.0, 0.0, 0.0, 50.0]),
            "high": np.array([10.0, 1000.0, 300.0, 200.0]),
        },
        "x0": np.array([1.0, 150.0, 0.0, 80.0]),
    }


@register_model("photobioreactor", aliases=("photo_production",), defaults=_defaults)
@dataclass(frozen=False, kw_only=True)
class photo_production:
    """
    Photo Production of Phycocyanin from
    Cyanobacteria Arthrospira platensis

    Attributes:
    u_m (float):
    u_d (float):
    Y_NX (float):
    k_m (float):
    k_d (float):
    k_sq (float):
    K_Nq (float):
    k_iq (float):
    int_method (str): Integration method ('jax' or other)

    Uncertain Parameters:
    k_s, k_i, k_N: floats with normal mean and 10% std dev

    """

    u_m: float = 0.0572
    u_d: float = 0.0
    Y_NX: float = 504.5
    k_m: float = 0.00016
    k_d: float = 0.281
    k_sq: float = 23.51
    K_Nq: float = 16.89
    k_iq: float = 800.0
    k_s: float = 178.9  # Normal(178.9, 17.89)
    k_i: float = 447.1  # Normal(447.1, 44.71)
    k_N: float = 393.1  # Normal(393.1, 39.31)
    int_method: str = "jax"

    def __call__(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Calculates the state derivatives for the photo production model

        Args:
            x (np.ndarray): Current State [c_x, c_N, c_q]; concentrations of biomass, nitrate and product
            u (np.ndarray): Input [I, F_N]; light intensity and nitrate feed rate

        Returns:
            np.ndarray: State derivatives [dc_x, dc_N, dc_q]
        """
        c_x, c_N, c_q = x[0], x[1], x[2]
        I, F_N = u[0], u[1]

        dc_x = self.u_m * I / (I + self.k_s + (I**2 / self.k_i)) * c_x * c_N / (c_N + self.k_N) - self.u_d * c_x
        dc_N = -self.Y_NX * self.u_m * I / (I + self.k_s + (I**2 / self.k_i)) * c_x * c_N / (c_N + self.k_N) + F_N
        dc_q = self.k_m * I / (I + self.k_sq + (I**2 / self.k_iq)) * c_x - (self.k_d * c_q) / (c_N + self.K_Nq)

        ret = [dc_x, dc_N, dc_q]

        return jnp.array(ret) if self.int_method == "jax" else np.array(ret)

    def info(self) -> dict:
        """
        Get Model information

        Returns:
            dict: Model information containing model parameters, states, inputs and disturbances
        """

        info = {
            "parameters": self.__dict__.copy(),
            "states": ["c_x", "c_N", "c_q"],
            "inputs": ["I", "F_N"],
            "disturbances": [],
            "uncertainties": [],
        }
        info["parameters"].pop("int_method")
        return info
