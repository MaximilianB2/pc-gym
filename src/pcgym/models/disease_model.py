from dataclasses import dataclass

import numpy as np

from pcgym.models._base import BaseModel, jnp
from pcgym.models._registry import register_model


# -------------------------------------------------
# 2. SIRS Disease Model with Vaccination
# -------------------------------------------------
@register_model("disease", aliases=("disease_model",))
@dataclass(frozen=False, kw_only=True)
class disease_model(BaseModel):
    beta: float = 0.3
    gamma: float = 0.1
    int_method: str = "jax"
    states: list = None
    inputs: list = None
    disturbances: list = None
    uncertainties: dict = None

    def __post_init__(self):
        self.states = ["S", "I", "R"]
        self.inputs = ["u"]  # vaccination rate
        self.disturbances = []

    def __call__(self, x, u):
        S, I, _R = x[0], x[1], x[2]
        u_in = u[0]
        dSdt = -self.beta * S * I - u_in * S
        dIdt = self.beta * S * I - self.gamma * I
        dRdt = self.gamma * I + u_in * S

        ret = [dSdt, dIdt, dRdt]

        return jnp.array(ret) if self.int_method == "jax" else np.array(ret)
