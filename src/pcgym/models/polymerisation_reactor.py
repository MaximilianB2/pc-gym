from dataclasses import dataclass

import numpy as np

from pcgym.models._base import jnp
from pcgym.models._registry import Regulation, register_model


def _defaults():
    """Canonical a_space / o_space / x0 used when they are omitted from env_params."""
    return {
        "a_space": {"low": np.array([0.1, 320.0, 4.0, 0.3]), "high": np.array([2.0, 360.0, 8.0, 1.0])},
        "o_space": {"low": np.array([300.0, 0.0, 0.0, 1.5]), "high": np.array([400.0, 10.0, 1.0, 3.5])},
        "x0": np.array([340.0, 5.0, 0.3, 2.0]),
    }


@register_model("polymerisation_reactor", defaults=_defaults, task=Regulation(setpoint={"M": 3.0}))
@dataclass(frozen=False, kw_only=True)
class polymerisation_reactor:
    """
    Polymerisation reactor model.

    Attributes:
        Ap, Ad, At (float): Pre-exponential factors (1/sec)
        Ep_over_R, Ed_over_R, Et_over_R (float): Activation energies over R (K)
        f (float): Reactivity fraction for free radicals
        V (float): Reactor volume (m3)
        deltaHp (float): Heat of reaction per monomer unit (kJ/kmol)
        rho (float): Density of input fluid mixture (kg/m3)
        cp (float): Heat capacity of fluid mixture (kj/kg K)
    """

    # Parameters
    Ap: float = 6e10  # Pre-exponential factor for step p [1/sec]
    Ad: float = 4e10  # Pre-exponential factor for step d [1/sec]
    At: float = 9e10  # Pre-exponential factor for step t [1/sec]
    Ep_over_R: float = 7750  # Activation energy over R for step p [K]
    Ed_over_R: float = 8500  # Activation energy over R for step d [K]
    Et_over_R: float = 8250  # Activation energy over R for step t [K]
    f: float = 0.5  # Reactivity fraction for free radicals [-]
    V: float = 1.0  # Reactor volume [m3]
    deltaHp: float = -3e4  # Heat of reaction per monomer unit [kJ/kmol]
    rho: float = 1200.0  # Density of input fluid mixture [kg/m3]
    cp: float = 2.0  # Heat capacity of fluid mixture [kj/kg K]
    int_method: str = "jax"

    def __call__(self, x, u):
        """
        Calculate the state derivatives for the polymerisation reactor.

        Args:
            x (np.ndarray): Current state [T, M, I]
            u (np.ndarray): Input [F, Tf, Mf, If]

        Returns:
            np.ndarray: State derivatives
        """
        T, M, I = x[0], x[1], x[2]
        F, Tf, Mf, If = u[0], u[1], u[2], u[3]
        xp = jnp if self.int_method == "jax" else np

        kp = self.Ap * xp.exp(-self.Ep_over_R / T)
        kd = self.Ad * xp.exp(-self.Ed_over_R / T)
        kt = self.At * xp.exp(-self.Et_over_R / T)

        ri = 2 * self.f * kd * I
        rp = kp * ((self.f * kd * I) / kt) ** 0.5

        ret = [
            (F / self.V) * (Tf - T) + ((-self.deltaHp) / (self.rho * self.cp)) * rp,
            (F / self.V) * (Mf - M) - rp,
            (F / self.V) * (If - I) - ri,
        ]

        return jnp.array(ret) if self.int_method == "jax" else np.array(ret)

    def info(self):
        # Return a dictionary with the model information
        """
        Get model information.

        Returns:
            dict: Dictionary containing model parameters, states, inputs, and disturbances.
        """
        info = {
            "parameters": self.__dict__.copy(),
            "states": ["T", "M", "I"],
            "inputs": ["F", "Tf", "Mf", "If"],
            "disturbances": [],
        }
        return info
