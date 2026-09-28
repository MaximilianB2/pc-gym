from dataclasses import dataclass

import numpy as np

from pcgym.models._base import jnp
from pcgym.models._registry import Regulation, register_model


def _defaults():
    """Canonical a_space / o_space / x0 used when they are omitted from env_params."""
    return {
        "a_space": {
            "low": np.array([0.0, 1.0, 0.05, 0.05, 0.05]),
            "high": np.array([10.0, 30.0, 1.0, 1.0, 1.0]),
        },
        "o_space": {
            "low": np.array([0.0, 0.0, 0.0, 0.0] * 4 + [0.9]),
            "high": np.array([10.0, 10.0, 10.0, 500.0] * 4 + [2.5]),
        },
        "x0": np.array([2.0, 0.1, 10.0, 0.1] * 4 + [1.0]),
    }


@register_model("biofilm_reactor", defaults=_defaults, task=Regulation(setpoint={"S2_A": 2.0}))
@dataclass(frozen=False, kw_only=True)
class biofilm_reactor:
    """
    Biofilm reactor model.

    Attributes:
        V (float): Volume of one reactor stage (L)
        Va (float): Volume of absorber tank (L)
        Kla (float): Transfer coefficient (hr)
        m (float): Equilibrium constant
        eq_exponent (float): Nonlinearity of equilibrium relationship
        O_air (float): Concentration of oxygen in air (mg/L)
        vm_1, vm_2 (float): Maximum velocities through fluidized bed (mg/L hr)
        K1, K2 (float): Equilibrium constants for reactions
        KO_1, KO_2 (float): Equilibrium constants for oxygen
        int_method (str): Integration method ('jax' or other)
    """

    # Parameters
    V: float = 10.0  # Volume of one reactor stage [L]
    Va: float = 15.0  # Volume of absorber tank [L]
    Kla: float = 1.5  # Transfer coefficient [hr]
    m: float = 0.5  # Equilibrium constant [-]
    eq_exponent: float = 1.0
    O_air: float = 300  # Concentration of oxygen in air [mg/L]
    vm_1: float = 0.8  # Maximum velocity through fluidized bed for reaction 1 [mg/L hr]
    vm_2: float = 1.0  # Maximum velocity through fluidized bed for reaction 2 [mg/L hr]
    K1: float = 0.5  # Equilibrium constant for reaction 1 (Saturation constant for ammonia in reaction 1) [mg/L]
    K2: float = 0.1  # Equilibrium constant for reaction 2 (Saturation constant for ammonia in reaction 2) [mg/L]
    KO_1: float = 1.5  # Equilibrium constant for oxygen in reaction 1 (saturation constant for oxygen) [mg/L]
    KO_2: float = 0.5  # Equilibrium constant for oxygen in reaction 2 (saturation constant for oxygen) [mg/L]
    int_method: str = "jax"

    def __call__(self, x, u):
        """
        Calculate the state derivatives for the biofilm reactor.

        Args:
            x (np.ndarray): Current state [S1_1, S2_1, S3_1, O_1, ..., S1_A, S2_A, S3_A, O_A]
            u (np.ndarray): Input [F, Fr, S1_F, S2_F, S3_F]

        Returns:
            np.ndarray: State derivatives
        """
        (
            S1_1,
            S2_1,
            S3_1,
            O_1,
            S1_2,
            S2_2,
            S3_2,
            O_2,
            S1_3,
            S2_3,
            S3_3,
            O_3,
            S1_A,
            S2_A,
            S3_A,
            O_A,
        ) = (
            x[0],
            x[1],
            x[2],
            x[3],
            x[4],
            x[5],
            x[6],
            x[7],
            x[8],
            x[9],
            x[10],
            x[11],
            x[12],
            x[13],
            x[14],
            x[15],
        )
        F, Fr, S1_F, S2_F, S3_F = u[0], u[1], u[2], u[3], u[4]

        r1_1 = ((self.vm_1 * S1_1) / (self.K1 + S1_1)) * ((O_1) / (self.KO_1 + O_1))
        r2_1 = ((self.vm_2 * S2_1) / (self.K2 + S2_1)) * ((O_1) / (self.KO_2 + O_1))
        ro_1 = -r1_1 * 3.5 - r2_1 * 1.1

        r1_2 = ((self.vm_1 * S1_2) / (self.K1 + S1_2)) * ((O_2) / (self.KO_1 + O_2))
        r2_2 = ((self.vm_2 * S2_2) / (self.K2 + S2_2)) * ((O_2) / (self.KO_2 + O_2))
        ro_2 = -r1_2 * 3.5 - r2_2 * 1.1

        r1_3 = ((self.vm_1 * S1_3) / (self.K1 + S1_3)) * ((O_3) / (self.KO_1 + O_3))
        r2_3 = ((self.vm_2 * S2_3) / (self.K2 + S2_3)) * ((O_3) / (self.KO_2 + O_3))
        ro_3 = -r1_3 * 3.5 - r2_3 * 1.1

        rs1_1 = -r1_1
        rs2_1 = +r1_1 - r2_1
        rs3_1 = r2_1

        rs1_2 = -r1_2
        rs2_2 = +r1_2 - r2_2
        rs3_2 = r2_2

        rs1_3 = -r1_3
        rs2_3 = +r1_3 - r2_3
        rs3_3 = r2_3

        O_Aeq = (self.O_air**self.eq_exponent) / self.m

        ret = [
            (Fr / self.V) * (S1_A - S1_1) - rs1_1,
            (Fr / self.V) * (S2_A - S2_1) - rs2_1,
            (Fr / self.V) * (S3_A - S3_1) - rs3_1,
            (Fr / self.V) * (O_A - O_1) - ro_1,
            (Fr / self.V) * (S1_1 - S1_2) - rs1_2,
            (Fr / self.V) * (S2_1 - S2_2) - rs2_2,
            (Fr / self.V) * (S3_1 - S3_2) - rs3_2,
            (Fr / self.V) * (O_1 - O_2) - ro_2,
            (Fr / self.V) * (S1_2 - S1_3) - rs1_3,
            (Fr / self.V) * (S2_2 - S2_3) - rs2_3,
            (Fr / self.V) * (S3_2 - S3_3) - rs3_3,
            (Fr / self.V) * (O_2 - O_3) - ro_3,
            (Fr / self.Va) * (S1_3 - S1_A) + (F / self.Va) * (S1_F - S1_A),
            (Fr / self.Va) * (S2_3 - S2_A) + (F / self.Va) * (S2_F - S2_A),
            (Fr / self.Va) * (S3_3 - S3_A) + (F / self.Va) * (S3_F - S3_A),
            (Fr / self.Va) * (O_3 - O_A) + self.Kla * (O_Aeq - O_A),
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
                "S1_1",
                "S2_1",
                "S3_1",
                "O_1",
                "S1_2",
                "S2_2",
                "S3_2",
                "O_2",
                "S1_3",
                "S2_3",
                "S3_3",
                "O_3",
                "S1_A",
                "S2_A",
                "S3_A",
                "O_A",
            ],
            "inputs": ["F", "Fr", "S1_F", "S2_F", "S3_F"],
            "disturbances": [],
        }
        info["parameters"].pop("int_method", None)
        return info
