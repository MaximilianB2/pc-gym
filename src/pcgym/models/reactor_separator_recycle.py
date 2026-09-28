from dataclasses import dataclass

import numpy as np

from pcgym.models._base import jnp
from pcgym.models._registry import register_model


@register_model("reactor_separator_recycle", aliases=("RSR",))
@dataclass(frozen=False, kw_only=True)
class RSR:
    # Parameters
    int_method: str = "jax"
    rho: float = 1.0  # Liquid density
    alpha_1: float = 90.0  # Volatility
    k_1: float = 0.0167  # Rate constant
    k_2: float = 0.0167  # Rate constant
    A_R: float = 10.0  # Vessel area
    A_M: float = 10.0  # Vessel area
    A_B: float = 10.0  # Vessel area
    x1_O: float = 1.00  # Initial molar liquid fraction of component 1

    def __call__(self, x, u):
        H_R, x1_R, x2_R, x3_R, H_M, x1_M, x2_M, x3_M, H_B, x1_B, x2_B, x3_B = (
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
            x[10],
            x[11],
        )
        F_O, F_R, F_M, B, D = u[0], u[1], u[2], u[3], u[4]

        x1_D = (x1_B * self.alpha_1) / (1 - x1_B + x1_B * self.alpha_1)
        x2_D = 1 - x1_D

        ret = [
            (1 / (self.rho * self.A_R)) * (F_O + D - F_R),
            ((F_O * (self.x1_O - x1_R) + D * (x1_D - x1_R)) / (self.rho * self.A_R * H_R)) - self.k_1 * x1_R,
            ((-F_O * x2_R + D * (x2_D - x2_R)) / (self.rho * self.A_R * H_R)) + self.k_1 * x1_R - self.k_2 * x2_R,
            ((-x3_R * (F_O + D)) / (self.rho * self.A_R * H_R)) + self.k_2 * x2_R,
            (1 / (self.rho * self.A_M)) * (F_R - F_M),
            ((F_R) / (self.rho * self.A_M * H_M)) * (x1_R - x1_M),
            ((F_R) / (self.rho * self.A_M * H_M)) * (x2_R - x2_M),
            ((F_R) / (self.rho * self.A_M * H_M)) * (x3_R - x3_M),
            (1 / (self.rho * self.A_B)) * (F_M - B - D),
            (1 / (self.rho * self.A_B * H_B)) * (F_M * (x1_M - x1_B) - D * (x1_D - x1_B)),
            (1 / (self.rho * self.A_B * H_B)) * (F_M * (x2_M - x2_B) - D * (x2_D - x2_B)),
            (1 / (self.rho * self.A_B * H_B)) * (F_M * (x3_M - x3_B) + D * (x3_B)),
        ]

        return jnp.array(ret) if self.int_method == "jax" else np.array(ret)

    def info(self):
        # Return a dictionary with the model information
        info = {
            "parameters": self.__dict__.copy(),
            "states": [
                "H_R",
                "x1_R",
                "x2_R",
                "x3_R",
                "H_M",
                "x1_M",
                "x2_M",
                "x3_M",
                "H_B",
                "x1_B",
                "x2_B",
                "x3_B",
            ],
            "inputs": ["F_O", "F_R", "F_M", "B", "D"],
            "disturbances": [],
        }
        info["parameters"].pop(
            "int_method", None
        )  # Remove 'int_method' from the dictionary since it is not a parameter of the model
        return info
