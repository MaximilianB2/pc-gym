from dataclasses import dataclass

import numpy as np

from pcgym.models._base import BaseModel, jnp
from pcgym.models._registry import Regulation, register_model


def _defaults():
    return {
        "a_space": {"low": np.array([-0.5]), "high": np.array([0.5])},
        "o_space": {"low": np.array([0.0, 0.0, 0.5]), "high": np.array([1.5, 1.5, 1.0])},
        "x0": np.array([1.0, 0.5, 0.9]),
    }


@register_model("hydraulic_tank", defaults=_defaults, task=Regulation(setpoint={"q1": 0.9}))
@dataclass(frozen=False, kw_only=True)
class hydraulic_tank(BaseModel):
    """Linear transfer of volume between two tanks, with q1 + q2 conserved.

    Uses consistent benchmark volume/time units; u is signed volume per time and
    D is inverse time. The default target (0.9, 0.6) is held by u=0.3.
    """

    D: float = 1.0  # [1/time]
    int_method: str = "jax"
    states: list = None
    inputs: list = None
    disturbances: list = None
    uncertainties: dict = None

    def __post_init__(self):
        self.states = ["q1", "q2"]
        self.inputs = ["u"]
        self.disturbances = []

    def validate_initial_state(self, state):
        if np.any(np.asarray(state) < 0):
            raise ValueError("hydraulic_tank requires nonnegative initial volumes")

    def validate_input_bounds(self, low, high):
        if not np.isfinite(self.D) or self.D <= 0:
            raise ValueError("hydraulic_tank requires a positive transfer coefficient D")

    def validate_episode(self, state, low, high, duration):
        # At either empty-tank boundary, every admitted input must point inward.
        capacity = self.D * np.sum(state)
        if low[0] < -capacity or high[0] > capacity:
            raise ValueError("hydraulic_tank input bounds can drain a tank below zero for this total volume")

    def __call__(self, x, u):
        q1, q2 = x[0], x[1]
        u_in = u[0]
        dq1dt = -self.D * (q1 - q2) + u_in
        dq2dt = self.D * (q1 - q2) - u_in

        ret = [dq1dt, dq2dt]

        return jnp.array(ret) if self.int_method == "jax" else np.array(ret)
