from dataclasses import dataclass

import numpy as np

from pcgym.models._base import jnp
from pcgym.models._registry import register_model


def _defaults():
    """Canonical a_space / o_space / x0 used when they are omitted from env_params."""
    return {
        "a_space": {
            "low": np.array([0.5, 0.5, 340.0, 300.0]),
            "high": np.array([5.0, 5.0, 360.0, 320.0]),
        },
        "o_space": {"low": np.array([280.0] * 24 + [300.0]), "high": np.array([400.0] * 24 + [360.0])},
        "x0": np.concatenate([np.tile([350.0, 340.0, 310.0], 8), [340.0]]),
    }


@register_model("heat_exchanger", defaults=_defaults)
@dataclass(frozen=False, kw_only=True)
class heat_exchanger:
    """
    Heat exchanger model.

    Attributes:
        Utm (float): Tube-metal overall heat transfer coefficient (kW/m2 K)
        Usm (float): Shell-metal overall heat transfer coefficient (kW/m2 K)
        L (float): Length segment of each stage
        Dt (float): Internal diameter of tube wall (m)
        Dm (float): Outside diameter of metal wall (m)
        Ds (float): Shell wall diameter (m)
        cpt, cpm, cps (float): Heat capacities (kJ/kg K)
        rhot, rhom, rhos (float): Densities (kg/m3)
    """

    # Parameters
    Utm: float = 1.0  # Tube-metal overall heat transfer coefficient [kW/m2 K]
    Usm: float = 1.0  # Shell-metal overall heat transfer coefficient [kW/m2 K]
    L: float = 1.0  # Length segment of each stage
    Dt: float = 1.0  # Internal diameter of tube wall [m]
    Dm: float = 2.0  # Outside diameter of metal wall [m]
    Ds: float = 3.0  # Shell wall diameter [m]
    cpt: float = 1.0  # Heat capacity of tube side fluid [kJ/kg K]
    cpm: float = 1.0  # Heat capacity of metal wall [kJ/kg K]
    cps: float = 1.0  # Heat capacity of shell side fluid [kJ/kg K]
    rhot: float = 1.0  # Density of tube side fluid [kg/m3]
    rhom: float = 1.0  # Density of metal [kg/m3]
    rhos: float = 1.0  # Density of shell side fluid [kg/m3]
    int_method: str = "jax"

    def __call__(self, x, u):
        """
        Calculate the state derivatives for the heat exchanger.

        Args:
            x (np.ndarray): Current state [Tt1, Tm1, Ts1, ..., Tt8, Tm8, Ts8]
            u (np.ndarray): Input [Ft, Fs, Tt0, Ts9]

        Returns:
            np.ndarray: State derivatives
        """
        (
            Tt1,
            Tm1,
            Ts1,
            Tt2,
            Tm2,
            Ts2,
            Tt3,
            Tm3,
            Ts3,
            Tt4,
            Tm4,
            Ts4,
            Tt5,
            Tm5,
            Ts5,
            Tt6,
            Tm6,
            Ts6,
            Tt7,
            Tm7,
            Ts7,
            Tt8,
            Tm8,
            Ts8,
        ) = (x[i] for i in range(24))
        Ft, Fs, Tt0, Ts9 = u[0], u[1], u[2], u[3]
        xp = jnp if self.int_method == "jax" else np

        Vt = self.L * xp.pi * self.Dt**2
        At = self.L * xp.pi * self.Dt
        Vm = self.L * xp.pi * (self.Dm**2 - self.Dt**2)
        Am = self.L * xp.pi * self.Dm
        Vs = self.L * xp.pi * (self.Ds**2 - self.Dm**2)

        Qt1 = self.Utm * At * (Tt1 - Tm1)
        Qm1 = self.Usm * Am * (Tm1 - Ts1)
        Qt2 = self.Utm * At * (Tt2 - Tm2)
        Qm2 = self.Usm * Am * (Tm2 - Ts2)
        Qt3 = self.Utm * At * (Tt3 - Tm3)
        Qm3 = self.Usm * Am * (Tm3 - Ts3)
        Qt4 = self.Utm * At * (Tt4 - Tm4)
        Qm4 = self.Usm * Am * (Tm4 - Ts4)
        Qt5 = self.Utm * At * (Tt5 - Tm5)
        Qm5 = self.Usm * Am * (Tm5 - Ts5)
        Qt6 = self.Utm * At * (Tt6 - Tm6)
        Qm6 = self.Usm * Am * (Tm6 - Ts6)
        Qt7 = self.Utm * At * (Tt7 - Tm7)
        Qm7 = self.Usm * Am * (Tm7 - Ts7)
        Qt8 = self.Utm * At * (Tt8 - Tm8)
        Qm8 = self.Usm * Am * (Tm8 - Ts8)

        ret = [
            (1 / (self.cpt * self.rhot * Vt)) * (Ft * self.cpt * (Tt0 - Tt1) - Qt1),
            (1 / (self.cpm * self.rhom * Vm)) * (Qt1 - Qm1),
            (1 / (self.cps * self.rhos * Vs)) * (Fs * self.cps * (Ts2 - Ts1) + Qm1),
            (1 / (self.cpt * self.rhot * Vt)) * (Ft * self.cpt * (Tt1 - Tt2) - Qt2),
            (1 / (self.cpm * self.rhom * Vm)) * (Qt2 - Qm2),
            (1 / (self.cps * self.rhos * Vs)) * (Fs * self.cps * (Ts3 - Ts2) + Qm2),
            (1 / (self.cpt * self.rhot * Vt)) * (Ft * self.cpt * (Tt2 - Tt3) - Qt3),
            (1 / (self.cpm * self.rhom * Vm)) * (Qt3 - Qm3),
            (1 / (self.cps * self.rhos * Vs)) * (Fs * self.cps * (Ts4 - Ts3) + Qm3),
            (1 / (self.cpt * self.rhot * Vt)) * (Ft * self.cpt * (Tt3 - Tt4) - Qt4),
            (1 / (self.cpm * self.rhom * Vm)) * (Qt4 - Qm4),
            (1 / (self.cps * self.rhos * Vs)) * (Fs * self.cps * (Ts5 - Ts4) + Qm4),
            (1 / (self.cpt * self.rhot * Vt)) * (Ft * self.cpt * (Tt4 - Tt5) - Qt5),
            (1 / (self.cpm * self.rhom * Vm)) * (Qt5 - Qm5),
            (1 / (self.cps * self.rhos * Vs)) * (Fs * self.cps * (Ts6 - Ts5) + Qm5),
            (1 / (self.cpt * self.rhot * Vt)) * (Ft * self.cpt * (Tt5 - Tt6) - Qt6),
            (1 / (self.cpm * self.rhom * Vm)) * (Qt6 - Qm6),
            (1 / (self.cps * self.rhos * Vs)) * (Fs * self.cps * (Ts7 - Ts6) + Qm6),
            (1 / (self.cpt * self.rhot * Vt)) * (Ft * self.cpt * (Tt6 - Tt7) - Qt7),
            (1 / (self.cpm * self.rhom * Vm)) * (Qt7 - Qm7),
            (1 / (self.cps * self.rhos * Vs)) * (Fs * self.cps * (Ts8 - Ts7) + Qm7),
            (1 / (self.cpt * self.rhot * Vt)) * (Ft * self.cpt * (Tt7 - Tt8) - Qt8),
            (1 / (self.cpm * self.rhom * Vm)) * (Qt8 - Qm8),
            (1 / (self.cps * self.rhos * Vs)) * (Fs * self.cps * (Ts9 - Ts8) + Qm8),
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
            "states": [
                "Tt1",
                "Tm1",
                "Ts1",
                "Tt2",
                "Tm2",
                "Ts2",
                "Tt3",
                "Tm3",
                "Ts3",
                "Tt4",
                "Tm4",
                "Ts4",
                "Tt5",
                "Tm5",
                "Ts5",
                "Tt6",
                "Tm6",
                "Ts6",
                "Tt7",
                "Tm7",
                "Ts7",
                "Tt8",
                "Tm8",
                "Ts8",
            ],
            "inputs": ["Ft", "Fs", "Tt0", "Ts9"],
        }
        return info
