from dataclasses import dataclass
from numbers import Integral

import numpy as np

from pcgym.models._base import BaseModel, jnp
from pcgym.models._registry import register_model


# -------------------------------------------------
# 3. Coupled Oscillator System
# -------------------------------------------------
@register_model("coupled_oscillator", aliases=("coupled_oscillators",), simulation_only=True)
@dataclass(frozen=False, kw_only=True)
class coupled_oscillators(BaseModel):
    """Autonomous periodic spring-mass chain; conserves total momentum and energy.

    States are N positions followed by N momenta. There is no forcing input,
    so this is a simulation example, with no default control objective.
    """

    N: int = 10
    k: float = 1.0  # spring constant
    m: float = 1.0  # mass
    int_method: str = "jax"
    states: list = None
    inputs: list = None
    disturbances: list = None
    uncertainties: dict = None

    def __post_init__(self):
        if isinstance(self.N, (bool, np.bool_)) or not isinstance(self.N, Integral) or self.N < 1:
            raise ValueError("coupled_oscillator requires a positive integer number of masses N")
        if not np.isfinite([self.k, self.m]).all() or self.k < 0 or self.m <= 0:
            raise ValueError("coupled_oscillator requires nonnegative stiffness and positive mass")
        self.states = [f"x{i + 1}" for i in range(self.N)] + [f"p{i + 1}" for i in range(self.N)]
        self.inputs = []
        self.disturbances = []

    def validate_initial_state(self, state):
        values = np.asarray(state, dtype=float)
        if values.shape != (2 * self.N,) or not np.isfinite(values).all():
            raise ValueError("coupled_oscillator requires N finite positions and N finite momenta")

    def validate_input_bounds(self, low, high):
        self.__post_init__()

    def __call__(self, x, u=None):
        N, k, m = self.N, self.k, self.m
        positions = x[:N]
        momenta = x[N:]
        # Index symbolic entries individually; numpy.concatenate cannot consume
        # a CasADi column vector as a numeric array.
        dxdt = [momenta[i] / m for i in range(N)]
        dptdt = []
        for i in range(N):
            left = positions[(i - 1) % N]
            right = positions[(i + 1) % N]
            dptdt.append(-k * (2 * positions[i] - left - right))

        ret = dxdt + dptdt

        return jnp.array(ret) if self.int_method == "jax" else np.array(ret)
