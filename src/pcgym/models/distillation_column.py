from dataclasses import dataclass

import numpy as np

from pcgym.models._base import jnp
from pcgym.models._registry import Regulation, register_model


def _defaults():
    """Canonical a_space / o_space / x0 used when they are omitted from env_params."""
    return {
        # Fixed distillate D=100 requires F>=100 so bottoms W=F-D stays nonnegative.
        "a_space": {"low": np.array([1.0, 100.0]), "high": np.array([10.0, 300.0])},
        "o_space": {"low": np.array([0.0] * 10), "high": np.array([1.0] * 10)},
        "x0": np.array([0.85, 0.75, 0.6, 0.4, 0.2, 0.15, 0.1, 0.05, 0.02, 0.85]),
    }


@register_model("distillation_column", defaults=_defaults, task=Regulation(setpoint={"X0": 0.5}))
@dataclass(frozen=False, kw_only=True)
class distillation_column:
    # Parameters
    """
    Distillation column model.

    Attributes:
        D (float): Distillate flow rate (kmol/hr)
        q (float): Feed quality (q=1 is saturated liquid)
        alpha (float): Relative volatility of more volatile component
        X_feed (float): Feed composition
        M0, Mb, M (float): Holdup in different sections of the column
    """

    D: float = 100.0  # kmol/hr
    q: float = 1.0  # Feed quality (q=1 is saturated liquid)
    alpha: float = 5.0  # Relative volatility of more volatile component
    X_feed: float = 0.2
    M0: float = 2000.0
    Mb: float = 2000.0
    M: float = 2000.0
    int_method: str = "jax"

    def validate_initial_state(self, state):
        composition = np.asarray(state, dtype=float)[:9]
        if (
            composition.shape != (9,)
            or not np.isfinite(composition).all()
            or np.any((composition < 0) | (composition > 1))
        ):
            raise ValueError("distillation_column requires nine finite initial mole fractions in [0, 1]")

    def validate_input_bounds(self, low, high):
        """Require nonnegative external and internal flows over the entire input box."""
        positive = [self.D, self.M0, self.Mb, self.M, self.alpha]
        if not np.isfinite(positive).all() or np.any(np.asarray(positive) <= 0):
            raise ValueError("distillation_column requires positive distillate, holdups and relative volatility")
        if not np.isfinite([self.q, self.X_feed]).all() or not 0 <= self.X_feed <= 1:
            raise ValueError("distillation_column requires finite feed quality and feed mole fraction in [0, 1]")
        minimum_liquid = self.D * low[0] + min(self.q * low[1], self.q * high[1])
        minimum_vapour = self.D * (low[0] + 1) - max((1 - self.q) * low[1], (1 - self.q) * high[1])
        if low[0] < 0 or low[1] < self.D or minimum_liquid < 0 or minimum_vapour < 0:
            raise ValueError(
                "distillation_column input bounds must keep reflux, bottoms and internal flows nonnegative"
            )

    def __call__(self, x, u):
        """
        Calculate the state derivatives for the distillation column.

        Args:
            x (np.ndarray): Current state [X0, X1, X2, X3, Xf, X4, X5, X6, Xb]
            u (np.ndarray): Input [R, F]

        Returns:
            np.ndarray: State derivatives
        """
        X0, X1, X2, X3, Xf, X4, X5, X6, Xb = (
            x[0],
            x[1],
            x[2],
            x[3],
            x[4],
            x[5],
            x[6],
            x[7],
            x[8],
        )
        R, F = u[0], u[1]

        L = R * self.D
        V = (R + 1) * self.D
        L_dash = L + self.q * F
        # Feed vapour joins the upward flow: V = V_dash + (1-q) F.
        V_dash = V - (1 - self.q) * F
        W = F - self.D

        Y1 = (self.alpha * X1) / (1 + (self.alpha - 1) * X1)
        Y2 = (self.alpha * X2) / (1 + (self.alpha - 1) * X2)
        Y3 = (self.alpha * X3) / (1 + (self.alpha - 1) * X3)
        Yf = (self.alpha * Xf) / (1 + (self.alpha - 1) * Xf)
        Y4 = (self.alpha * X4) / (1 + (self.alpha - 1) * X4)
        Y5 = (self.alpha * X5) / (1 + (self.alpha - 1) * X5)
        Y6 = (self.alpha * X6) / (1 + (self.alpha - 1) * X6)
        Yb = (self.alpha * Xb) / (1 + (self.alpha - 1) * Xb)

        ret = [
            (1 / self.M0) * ((V * Y1) - (L + self.D) * X0),
            (1 / self.M) * (L * (X0 - X1) + V * (Y2 - Y1)),
            (1 / self.M) * (L * (X1 - X2) + V * (Y3 - Y2)),
            (1 / self.M) * (L * (X2 - X3) + V * (Yf - Y3)),
            (1 / self.M) * (L * X3 - L_dash * Xf + V_dash * Y4 - V * Yf + F * self.X_feed),
            (1 / self.M) * (L_dash * (Xf - X4) + V_dash * (Y5 - Y4)),
            (1 / self.M) * (L_dash * (X4 - X5) + V_dash * (Y6 - Y5)),
            (1 / self.M) * (L_dash * (X5 - X6) + V_dash * (Yb - Y6)),
            (1 / self.Mb) * (L_dash * X6 - W * Xb - V_dash * Yb),
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
            "states": ["X0", "X1", "X2", "X3", "Xf", "X4", "X5", "X6", "Xb"],
            "inputs": ["R", "F"],
            "disturbances": [],
        }
        return info
