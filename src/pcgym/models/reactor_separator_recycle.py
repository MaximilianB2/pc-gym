from dataclasses import dataclass

import numpy as np

from pcgym.models._base import jnp
from pcgym.models._registry import Regulation, register_model


def _defaults():
    return {
        "a_space": {"low": np.array([0.9, 1.8, 1.8, 0.9, 0.9]), "high": np.array([1.1, 2.2, 2.2, 1.1, 1.1])},
        "o_space": {
            "low": np.array([8.0, 0.0, 0.0, 0.0] * 3 + [8.0] * 3),
            "high": np.array([12.0, 1.0, 1.0, 1.0] * 3 + [12.0] * 3),
        },
        "x0": np.array([9.5, 0.5, 0.3, 0.2, 10.5, 0.5, 0.3, 0.2, 9.5, 0.5, 0.3, 0.2] + [10.0] * 3),
    }


@register_model(
    "reactor_separator_recycle",
    aliases=("RSR",),
    defaults=_defaults,
    task=Regulation(setpoint={"H_R": 10.0, "H_M": 10.0, "H_B": 10.0}, finite_horizon=True),
)
@dataclass(frozen=False, kw_only=True)
class RSR:
    """A -> B -> C with recycle of volatile A/B and nonvolatile C.

    Uses consistent benchmark mass/length/time units. Inputs are mass per time,
    rho is mass per volume, A_* are areas and H_* are heights. The default
    30-time-unit task restores vessel levels, within a conservatively checked domain.
    """

    # Parameters
    int_method: str = "jax"
    rho: float = 1.0  # Liquid density [benchmark mass/volume]
    alpha_1: float = 90.0  # Relative volatility of A to B [-]
    k_1: float = 0.0167  # Rate constant [1/benchmark time]
    k_2: float = 0.0167  # Rate constant [1/benchmark time]
    A_R: float = 10.0  # Vessel area [benchmark length^2]
    A_M: float = 10.0  # Vessel area [benchmark length^2]
    A_B: float = 10.0  # Vessel area [benchmark length^2]
    x1_O: float = 1.00  # Fresh-feed A fraction; only a pure-A feed is modeled

    def validate_initial_state(self, state):
        vessels = np.asarray(state).reshape(3, 4)
        fractions = vessels[:, 1:]
        if (
            np.any(vessels[:, 0] <= 0)
            or np.any(fractions < 0)
            or not np.allclose(fractions.sum(axis=1), 1, atol=1e-7, rtol=0)
        ):
            raise ValueError(
                "reactor_separator_recycle requires positive heights and composition fractions summing to one"
            )
        if fractions[2, :2].sum() <= 0:
            raise ValueError("reactor_separator_recycle requires volatile material in the separator")

    def validate_input_bounds(self, low, high):
        positive = [self.rho, self.A_R, self.A_M, self.A_B, self.alpha_1]
        rates = [self.k_1, self.k_2]
        if not np.isfinite(positive + rates).all() or min(positive) <= 0 or min(rates) < 0 or np.any(low < 0):
            raise ValueError("reactor_separator_recycle requires positive capacities and nonnegative rates/flows")
        if self.x1_O != 1.0:
            raise ValueError("reactor_separator_recycle supports only the documented pure-A feed (x1_O=1)")

    def validate_episode(self, state, low, high, duration):
        minimum_net_flows = np.array([low[0] + low[4] - high[1], low[1] - high[2], low[2] - high[3] - high[4]])
        capacities = self.rho * np.array([self.A_R, self.A_M, self.A_B])
        if np.any(np.asarray(state)[::4] + duration * np.minimum(minimum_net_flows, 0) / capacities <= 0):
            raise ValueError(
                "reactor_separator_recycle horizon/input bounds can empty a vessel; "
                "shorten the horizon or restrict flows"
            )
        # Even ignoring incoming volatiles, outgoing B+D cannot exhaust the initial A/B inventory.
        volatile_mass = self.rho * self.A_B * state[8] * (state[9] + state[10])
        if volatile_mass <= duration * (high[3] + high[4]):
            raise ValueError(
                "reactor_separator_recycle horizon/input bounds cannot guarantee remaining separator volatiles"
            )

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

        # C is nonvolatile (x3_D=0). Normalize only the two volatile components;
        # using 1-x1_B in the denominator incorrectly turns liquid C into vapour B.
        volatile = self.alpha_1 * x1_B + x2_B
        x1_D = self.alpha_1 * x1_B / volatile
        x2_D = x2_B / volatile

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
