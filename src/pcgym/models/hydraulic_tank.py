from dataclasses import dataclass

import numpy as np

from pcgym.models._base import BaseModel, jnp
from pcgym.models._registry import register_model


@register_model("hydraulic_tank")
@dataclass(frozen=False, kw_only=True)
class hydraulic_tank(BaseModel):
    D: float = 1.0
    int_method: str = "jax"
    states: list = None
    inputs: list = None
    disturbances: list = None
    uncertainties: dict = None

    def __post_init__(self):
        self.states = ["q1", "q2"]
        self.inputs = ["u"]
        self.disturbances = []

    def __call__(self, x, u):
        q1, q2 = x[0], x[1]
        u_in = u[0]
        dq1dt = -self.D * (q1 - q2) + u_in
        dq2dt = self.D * (q1 - q2) - u_in

        ret = [dq1dt, dq2dt]

        return jnp.array(ret) if self.int_method == "jax" else np.array(ret)
