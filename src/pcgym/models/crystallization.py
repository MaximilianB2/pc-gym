from dataclasses import dataclass

import numpy as np
from casadi import MX, SX, if_else

from pcgym.models._base import jnp
from pcgym.models._registry import Regulation, register_model


def _quality_from_moments(mu0, mu1, mu2):
    return (mu0 * mu2 - mu1**2) ** 0.5 / mu1, mu1 / mu0


def _defaults():
    """Canonical a_space / o_space / x0 used when they are omitted from env_params."""
    moments = (1478.00986666666, 22995.8230590611, 1800863.24079725, 248516167.940593)
    return {
        "a_space": {"low": np.array([10.0]), "high": np.array([40.0])},
        "o_space": {
            "low": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 14.0]),
            # Scales for the nominal 30-minute seeded batch (moments in micrometres).
            # A 1e20 scale erased all moment changes from float32 observations.
            "high": np.array([1e6, 1e7, 5e8, 3e10, 0.2, 3.0, 50.0, 1.1, 16.0]),
        },
        "x0": np.array([*moments, 0.15861523304, *_quality_from_moments(*moments[:3]), 1.0, 15.0]),
    }


@register_model("crystallization", defaults=_defaults, task=Regulation(setpoint={"CV": 1.0, "Ln": 15.0}))
@dataclass(frozen=False, kw_only=True)
class crystallization:
    """
    Crystallization of K2SO4 Control (PBE Model).

    This model represents a highly nonlinear crystallization process based on population balance equations (PBE).
    It simulates the evolution of crystal size distribution and concentration during the crystallization process.
    This is a seeded, growth-only approximation: nucleation and growth stop at or
    below saturation. Dissolution is not represented. CV and Ln are redundant
    differential states and must start consistently with the leading moments.

    Attributes:
        ka (float): Nucleation rate constant
        kb (float): Nucleation activation energy parameter
        kc (float): Nucleation supersaturation exponent
        kd (float): Nucleation crystal density exponent
        kg (float): Growth rate constant
        k1 (float): Growth activation energy parameter
        k2 (float): Growth supersaturation exponent
        a (float): Dimensionless intercept of the size-dependent growth law
        b (float): Slope of the size-dependent growth law (1/cm)
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

    def validate_initial_state(self, state):
        """Require realizable, nondegenerate seed moments and consistent quality states."""
        values = np.asarray(state, dtype=float)[:7]
        if values.shape != (7,) or not np.isfinite(values).all() or np.any(values[:5] < 0):
            raise ValueError("crystallization requires seven finite states and nonnegative moments/concentration")
        mu0, mu1, mu2, mu3 = values[:4]
        if mu0 <= 0 or mu1 <= 0 or mu0 * mu2 <= mu1**2 or mu1 * mu3 < mu2**2:
            raise ValueError("crystallization requires realizable seed moments with positive number, mean and variance")
        quality = _quality_from_moments(mu0, mu1, mu2)
        if values[5] <= 0 or not np.allclose(values[5:7], quality, rtol=1e-7, atol=1e-8):
            raise ValueError("crystallization initial CV and Ln must be derived from Mu0, Mu1 and Mu2")

    def project_state(self, state):
        """Reconcile redundant states after integration, preserving number and solute inventory.

        Retain the evolved CV: recovering a small variance by subtracting nearly
        equal raw moments can turn solver truncation error into artificial dispersion.
        """
        values = np.asarray(state, dtype=float).copy()
        if values[5] <= 0:
            raise ValueError("crystallization integration returned a nonpositive CV for a dispersed seed")
        mean = values[1] / values[0]
        values[2] = values[1] * mean * (1 + values[5] ** 2)
        values[6] = mean
        return values

    @staticmethod
    def equilibrium_concentration(temperature):
        """Solubility in g/cm³ at °C: the published Kelvin quadratic expanded about 25 °C."""
        offset = temperature - 25.0
        return 0.12051372253735 + offset * (0.001832757338 - 0.00000292874 * offset)

    def __call__(self, x, u):
        """
        Return derivatives of [Mu0, Mu1, Mu2, Mu3, Conc, CV, Ln] per minute.

        The input is the suspension temperature in °C, optionally followed by ka, kg.
        A five-state physical snapshot is also accepted; its CV is derived from the moments.
        """
        mu0, mu1, mu2, mu3, conc = x[0], x[1], x[2], x[3], x[4]
        if u.shape[0] not in (1, 3):
            raise ValueError("crystallization inputs must be [Tc] or [Tc, ka, kg]")
        temperature = u[0]
        kelvin = temperature + 273.15
        ka, kg = (self.ka, self.kg) if u.shape[0] == 1 else (u[1], u[2])
        xp = jnp if self.int_method == "jax" else np

        S = (conc - self.equilibrium_concentration(temperature)) * 1e3
        # Never extend the fitted supersaturated rate laws using abs(S): that
        # creates crystals from an undersaturated solution. Keep the fractional
        # powers away from zero in the inactive branch so symbolic derivatives
        # remain finite there (kc < 1).
        if isinstance(S, (SX, MX)):
            active = if_else(S > 0, 1.0, 0.0)
            rate_S = if_else(S > 0, S, 1.0)
        else:
            active = xp.where(S > 0, 1.0, 0.0)
            rate_S = xp.where(S > 0, S, 1.0)
        births = active * ka * xp.exp(self.kb / kelvin) * rate_S**self.kc * mu3**self.kd
        growth = active * kg * xp.exp(self.k1 / kelvin) * rate_S**self.k2
        intercept = 1e4 * self.a * growth
        slope = self.b * growth
        dmu1 = intercept * mu0 + slope * mu1
        dmu2 = 2 * (intercept * mu1 + slope * mu2)
        dmu3 = 3 * (intercept * mu2 + slope * mu3)
        dconc = -(self.ro * self.alfa / 6) * 1e-12 * dmu3

        mean = mu1 / mu0
        cv = x[5] if np.shape(x)[0] > 5 else _quality_from_moments(mu0, mu1, mu2)[0]
        births_per_crystal = births / mu0
        dmean = intercept + (slope - births_per_crystal) * mean
        dcv = births_per_crystal * (1 + cv**2) / (2 * cv) - intercept * cv / mean
        return xp.array([births, dmu1, dmu2, dmu3, dconc, dcv, dmean])

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
            "disturbances": ["ka", "kg"],
        }
        info["parameters"].pop("int_method", None)  # Remove 'int_method' since it's not a parameter of the model
        return info
