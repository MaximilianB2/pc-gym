from dataclasses import dataclass

import numpy as np

from pcgym.models._base import jnp
from pcgym.models._registry import Regulation, register_model


def _defaults():
    """Canonical a_space / o_space / x0 used when they are omitted from env_params."""
    return {
        "a_space": {"low": np.array([0.1, 320.0, 4.0, 0.3]), "high": np.array([2.0, 360.0, 8.0, 1.0])},
        # T + (-deltaHp / rho / cp) M cannot exceed max(E_initial, E_feed).
        # The largest feed energy is 360 + 12.5 * 8 = 460 K equivalent.
        "o_space": {"low": np.array([300.0, 0.0, 0.0, 1.5]), "high": np.array([460.0, 10.0, 1.0, 3.5])},
        "x0": np.array([340.0, 5.0, 0.3, 2.0]),
    }


@register_model("polymerisation_reactor", defaults=_defaults, task=Regulation(setpoint={"M": 2.0}))
@dataclass(frozen=False, kw_only=True)
class polymerisation_reactor:
    """
    Polymerisation reactor model.

    Attributes:
        Ap, At (float): Bimolecular pre-exponential factors (m3/kmol/sec)
        Ad (float): Initiator decomposition pre-exponential factor (1/sec)
        Ep_over_R, Ed_over_R, Et_over_R (float): Activation energies over R (K)
        f (float): Fraction of generated radicals that initiate chains
        V (float): Reactor volume (m3)
        deltaHp (float): Heat of reaction per monomer unit (kJ/kmol)
        rho (float): Density of input fluid mixture (kg/m3)
        cp (float): Heat capacity of fluid mixture (kj/kg K)
    """

    # Parameters
    Ap: float = 6e10  # Pre-exponential factor for propagation [m3/kmol/sec]
    Ad: float = 4e10  # Pre-exponential factor for step d [1/sec]
    At: float = 9e10  # Pre-exponential factor for termination [m3/kmol/sec]
    Ep_over_R: float = 7750  # Activation energy over R for step p [K]
    Ed_over_R: float = 8500  # Activation energy over R for step d [K]
    Et_over_R: float = 8250  # Activation energy over R for step t [K]
    f: float = 0.5  # Fraction of generated radicals that initiate chains [-]
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

        # I -> 2 radicals consumes initiator at kd * I, irrespective of the
        # initiation efficiency f. Quasi-steady radicals satisfy
        # 2 f kd I = 2 kt R**2; propagation R + M -> R consumes kp * M * R.
        initiator_decay = kd * I
        rp = kp * M * ((self.f * kd * I) / kt) ** 0.5

        ret = [
            (F / self.V) * (Tf - T) + ((-self.deltaHp) / (self.rho * self.cp)) * rp,
            (F / self.V) * (Mf - M) - rp,
            (F / self.V) * (If - I) - initiator_decay,
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
