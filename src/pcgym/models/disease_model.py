from dataclasses import dataclass

import numpy as np

from pcgym.models._base import BaseModel, jnp
from pcgym.models._registry import CustomReward, register_model


def _defaults():
    return {
        "a_space": {"low": np.array([0.0]), "high": np.array([0.05])},
        "o_space": {"low": np.zeros(3), "high": np.ones(3)},
        "x0": np.array([0.99, 0.01, 0.0]),
    }


def _vaccination_reward(env, state, action, constraint_violated):
    """Right-endpoint daily infection burden plus quadratic intervention effort."""
    return -env.dt * (state[1] + 20.0 * action[0] ** 2)


@register_model(
    "disease",
    aliases=("disease_model",),
    defaults=_defaults,
    task=CustomReward(_vaccination_reward, finite_horizon=True),
)
@dataclass(frozen=False, kw_only=True)
class disease_model(BaseModel):
    """SIR fractions with vaccination S -> R, permanent immunity and time in days.

    The nominal 60-day task penalizes infections and intervention intensity. It is
    a synthetic control benchmark, not a calibrated epidemiological recommendation.
    """

    beta: float = 0.3  # transmission rate [1/day]
    gamma: float = 0.1  # recovery rate [1/day]
    int_method: str = "jax"
    states: list = None
    inputs: list = None
    disturbances: list = None
    uncertainties: dict = None

    def __post_init__(self):
        self.states = ["S", "I", "R"]
        self.inputs = ["u"]  # vaccination rate
        self.disturbances = []

    def validate_initial_state(self, state):
        fractions = np.asarray(state)
        if np.any(fractions < 0) or not np.isclose(fractions.sum(), 1.0, rtol=0, atol=1e-7):
            raise ValueError("disease requires nonnegative SIR fractions summing to one")

    def validate_input_bounds(self, low, high):
        if not np.isfinite([self.beta, self.gamma]).all() or min(self.beta, self.gamma, low[0]) < 0:
            raise ValueError("disease requires nonnegative transmission, recovery and vaccination rates")

    def __call__(self, x, u):
        S, I, _R = x[0], x[1], x[2]
        u_in = u[0]
        dSdt = -self.beta * S * I - u_in * S
        dIdt = self.beta * S * I - self.gamma * I
        dRdt = self.gamma * I + u_in * S

        ret = [dSdt, dIdt, dRdt]

        return jnp.array(ret) if self.int_method == "jax" else np.array(ret)
