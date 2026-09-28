from dataclasses import dataclass

import numpy as np

from pcgym.models._base import BaseModel, jnp
from pcgym.models._registry import register_model


# -------------------------------------------------
# 3. Coupled Oscillator System
# -------------------------------------------------
@register_model("coupled_oscillator", aliases=("coupled_oscillators",))
@dataclass(frozen=False, kw_only=True)
class coupled_oscillators(BaseModel):
    N: int = 10
    k: float = 1.0  # spring constant
    m: float = 1.0  # mass
    int_method: str = "jax"
    states: list = None
    inputs: list = None
    disturbances: list = None
    uncertainties: dict = None

    def __post_init__(self):
        self.states = [f"x{i + 1}" for i in range(self.N)] + [f"p{i + 1}" for i in range(self.N)]
        self.inputs = []  # no external control except momentum-conserving forcing
        self.disturbances = []

    def __call__(self, x, u=None):
        N, k, m = self.N, self.k, self.m
        positions = x[:N]
        momenta = x[N:]
        xp = jnp if self.int_method == "jax" else np
        dxdt = momenta / m
        dptdt = []
        for i in range(N):
            left = positions[(i - 1) % N]
            right = positions[(i + 1) % N]
            dptdt.append(-k * (2 * positions[i] - left - right))

        ret = xp.concatenate([dxdt, xp.array(dptdt)])

        return jnp.array(ret) if self.int_method == "jax" else np.array(ret)
