from dataclasses import dataclass

import numpy as np

from pcgym.models._base import jnp
from pcgym.models._registry import Regulation, register_model

# Initial coefficient of variation / mean length, derived from the leading moment initial conditions.
_CRYST_CV0 = float(np.sqrt(1800863.24079725 * 1478.00986666666 / (22995.8230590611**2) - 1))
_CRYST_LN0 = 22995.8230590611 / (1478.00986666666 + 1e-6)


def _defaults():
    """Canonical a_space / o_space / x0 used when they are omitted from env_params."""
    return {
        "a_space": {"low": np.array([10.0]), "high": np.array([40.0])},
        "o_space": {
            "low": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 14.0]),
            "high": np.array([1e20, 1e20, 1e20, 1e20, 0.5, 2.0, 20.0, 1.1, 16.0]),
        },
        "x0": np.array(
            [
                1478.00986666666,
                22995.8230590611,
                1800863.24079725,
                248516167.940593,
                0.15861523304,
                _CRYST_CV0,
                _CRYST_LN0,
                1.0,
                15.0,
            ]
        ),
    }


@register_model("crystallization", defaults=_defaults, task=Regulation(setpoint={"CV": 1.0, "Ln": 15.0}))
@dataclass(frozen=False, kw_only=True)
class crystallization:
    """
    Crystallization of K2SO4 Control (PBE Model).

    This model represents a highly nonlinear crystallization process based on population balance equations (PBE).
    It simulates the evolution of crystal size distribution and concentration during the crystallization process.

    Attributes:
        ka (float): Nucleation rate constant
        kb (float): Nucleation activation energy parameter
        kc (float): Nucleation supersaturation exponent
        kd (float): Nucleation crystal density exponent
        kg (float): Growth rate constant
        k1 (float): Growth activation energy parameter
        k2 (float): Growth supersaturation exponent
        a (float): Moment model parameter for nucleation
        b (float): Moment model parameter for growth
        alfa (float): Shape factor for volume calculation
        ro (float): Crystal density (g/cm^3)
        int_method (str): Integration method ('jax' or other)

    Reference:
        https://pubs.acs.org/doi/10.1021/acs.iecr.3c00739
    """

    # Cristallization of K2SO4 Control (PBE Model).
    # highly nonlinear process
    # source: https://pubs.acs.org/doi/10.1021/acs.iecr.3c00739
    # Parameters
    ka: float = 0.923714966
    kb: float = -6754.878558
    kc: float = 0.92229965554
    kd: float = 1.341205945
    kg: float = 48.07514464
    k1: float = -4921.261419
    k2: float = 1.871281405
    a: float = 0.50523693
    b: float = 7.271241375
    alfa: float = 7.510905767
    ro: float = 2.658  # [roc] = g/cm^3
    int_method: str = "jax"

    def __call__(self, x, u):
        """
        Calculate the state derivatives for the crystallization model.

        This method computes the rates of change for the moments of the crystal size distribution
        and the solute concentration based on the current state and input temperature.

        Args:
            x (np.ndarray): Current state vector containing:
                - mu0 (float): 0th moment of crystal size distribution
                - mu1 (float): 1st moment of crystal size distribution
                - mu2 (float): 2nd moment of crystal size distribution
                - mu3 (float): 3rd moment of crystal size distribution
                - conc (float): Solute concentration

            u (np.ndarray): Input vector containing:
                - T (float): Temperature (°C)

        Returns:
            np.ndarray: State derivatives vector containing:
                - dmu0/dt: Rate of change of 0th moment
                - dmu1/dt: Rate of change of 1st moment
                - dmu2/dt: Rate of change of 2nd moment
                - dmu3/dt: Rate of change of 3rd moment
                - dconc/dt: Rate of change of solute concentration
        """
        mu0, mu1, mu2, mu3, conc = x[0], x[1], x[2], x[3], x[4]
        T = u[0]
        xp = jnp if self.int_method == "jax" else np

        Ceq = -686.2686 + 3.579165 * (T + 273.15) - 0.00292874 * (T + 273.15) ** 2
        S = conc * 1e3 - Ceq
        B0 = self.ka * xp.exp(self.kb / (T + 273.15)) * (S**2) ** (self.kc / 2) * ((mu3**2) ** (self.kd / 2))
        Ginf = self.kg * xp.exp(self.k1 / (T + 273.15)) * (S**2) ** (self.k2 / 2)

        dmi0dt = B0
        dmi1dt = Ginf * (self.a * mu0 + self.b * mu1 * 1e-4) * 1e4
        dmi2dt = 2 * Ginf * (self.a * mu1 * 1e-4 + self.b * mu2 * 1e-8) * 1e8
        dmi3dt = 3 * Ginf * (self.a * mu2 * 1e-8 + self.b * mu3 * 1e-12) * 1e12
        dcdt = -0.5 * self.ro * self.alfa * Ginf * (self.a * mu2 * 1e-8 + self.b * mu3 * 1e-12)

        CV = xp.sqrt(mu2 * mu0 / (mu1**2) - 1)
        dCVdt = (
            1
            / (2 * CV + 1e-10)
            * ((dmi2dt * mu0 + mu2 * dmi0dt) * mu1**2 - mu2 * mu0 * 2 * mu1 * dmi1dt)
            / (mu1**4 + 1e-10)
        )
        dLndt = (dmi1dt * mu0 - mu1 * dmi0dt) / (mu0**2 + 1e-10)

        ret = [dmi0dt, dmi1dt, dmi2dt, dmi3dt, dcdt, dCVdt, dLndt]

        return jnp.array(ret) if self.int_method == "jax" else np.array(ret)

    def info(self):
        # Return a dictionary with the model information
        """
        Get model information.

        This method returns a dictionary containing information about the model's
        parameters, states, inputs, and disturbances.

        Returns:
            dict: Dictionary containing:
                - parameters: Model parameters
                - states: Names of state variables
                - inputs: Names of input variables
                - disturbances: Names of disturbance variables (if any)
        """
        info = {
            "parameters": self.__dict__.copy(),
            "states": ["Mu0", "Mu1", "Mu2", "Mu3", "Conc", "CV", "Ln"],
            "inputs": ["Tc"],
            "disturbances": ["ka", "kg", "UA"],
        }
        info["parameters"].pop("int_method", None)  # Remove 'int_method' since it's not a parameter of the model
        return info
