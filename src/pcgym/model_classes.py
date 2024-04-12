from dataclasses import dataclass
import jax.numpy as jnp
import numpy as np


# Store the model and its parameters as dataclasses
# frozen: makes the objets immutable after creation
# so parameters can not be modified at runtime
# it also makes the class hashable, as required by Equinox:
# ValueError: Non-hashable static arguments are not supported.

# kw_only: require the parameter names if they want
# to be set when the object is created

# ==== CSTR Model ====#


# Temperature Control of an unstable CSTR reactor.
# highly exothermic reaction. This is an example of a highly nonlinear process prone to
# exponential run-away when the temperature rises too quickly.
# source: https://apmonitor.com/pdc/index.php/Main/StirredReactor
@dataclass(frozen=False, kw_only=True)
class cstr_ode:
    # Parameters
    q: float = 100  # m3/s
    V: float = 100  # m3
    rho: float = 1000  # kg/m3
    C: float = 0.239  # Joules/kg K
    deltaHr: float = -5e4  # Joules/kg K
    EA_over_R: float = 8750  # K
    k0: float = 7.2e10  # 1/sec
    UA: float = 5e4  # W/K
    Ti: float = 350  # K
    Caf: float = 1
    int_method: str = "jax"

    def __call__(self, x, u):
        # JAX requires jnp functions and arrays hence two versions
        if self.int_method == "jax":
            ca, T = x[0], x[1]
            if u.shape == (1,):
                Tc = u[0]
            else:
                Tc, self.Ti, self.Caf = u[0], u[1], u[2]
            Tc = u[0]
            rA = self.k0 * jnp.exp(-self.EA_over_R / T) * ca
            dxdt = jnp.array(
                [
                    self.q / self.V * (self.Caf - ca) - rA,
                    self.q / self.V * (self.Ti - T)
                    + ((-self.deltaHr) * rA) * (1 / (self.rho * self.C))
                    + self.UA * (Tc - T) * (1 / (self.rho * self.C * self.V)),
                ]
            )
            return dxdt
        else:
            ca, T = x[0], x[1]
            if u.shape == (1, 1):
                Tc = u[0]
            else:
                Tc, self.Ti, self.Caf = u[0], u[1], u[2]
            rA = self.k0 * np.exp(-self.EA_over_R / T) * ca
            dxdt = [
                self.q / self.V * (self.Caf - ca) - rA,
                self.q / self.V * (self.Ti - T)
                + ((-self.deltaHr) * rA) * (1 / (self.rho * self.C))
                + self.UA * (Tc - T) * (1 / (self.rho * self.C * self.V)),
            ]
            return dxdt

    def info(self):
        # Return a dictionary with the model information
        info = {
            "parameters": self.__dict__.copy(),
            "states": ["Ca", "T"],
            "inputs": ["Tc"],
            "disturbances": ["Ti", "Caf"],
        }
        info["parameters"].pop(
            "int_method", None
        )  # Remove 'int_method' from the dictionary since it is not a parameter of the model
        return info


# ==== First Order System Model ====#
@dataclass(frozen=False, kw_only=True)
class first_order_system_ode:
    # Parameters
    K: float = 1
    tau: float = 0.5
    int_method: str = "jax"

    def __call__(self, x, u):
        # JAX requires jnp functions and arrays hence two versions
        if self.int_method == "jax":
            # Model Equations
            x = x[0]
            u = u[0]

            dxdt = jnp.array([(self.K * u - x) * 1 / self.tau])

            return dxdt
        else:
            x = x[0]
            u = u[0]

            dxdt = [(self.K * u - x) * 1 / self.tau]

            return dxdt

    def info(self):
        # Return a dictionary with the model information
        info = {
            "parameters": self.__dict__.copy(),
            "states": ["x"],
            "inputs": ["u"],
            "disturbances": ["None"],
        }
        info["parameters"].pop(
            "int_method", None
        )  # Remove 'int_method' from the dictionary since it is not a parameter of the model
        return info


@dataclass(frozen=False, kw_only=True)
class multistage_extraction_ode:
    # Parameters
    Vl: float = 5  # Liquid volume in each stage
    Vg: float = 5  # Gas volume in each stage
    m: float = 1  # Equilibrium constant [-]
    Kla: float = 5  # Mass transfer capacity constant 1/hr
    eq_exponent: float = 2  # Change the nonlinearity of the equilibrium relationship
    X0: float = 0.6  # Feed concentration of liquid
    Y6: float = 0.05  # Feed conc of gas
    int_method: str = "jax"

    def __call__(self, x, u):
        # JAX requires jnp functions and arrays hence two versions
        if self.int_method == "jax":
            if u.shape == (2,):
                L, G = u[0], u[1]
            else:
                L, G, self.X0, self.Y6 = u[0], u[1], u[2], u[3]
            ###Model Equations###

            ##States##
            # Xn - Concentration of solute in liquid pase of stage n [kg/m3]
            # Yn - Concentration of solute in gas phase of stage n [kg/m3]
            X1, Y1, X2, Y2, X3, Y3, X4, Y4, X5, Y5 = (
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
            )

            ##Inputs##
            # L - Liquid flowrate m3/hr
            # G - Gas flowrate m3/hr
            X1_eq = (Y1**self.eq_exponent) / self.m
            X2_eq = (Y2**self.eq_exponent) / self.m
            X3_eq = (Y3**self.eq_exponent) / self.m
            X4_eq = (Y4**self.eq_exponent) / self.m
            X5_eq = (Y5**self.eq_exponent) / self.m

            Q1 = self.Kla * (X1 - X1_eq) * self.Vl
            Q2 = self.Kla * (X2 - X2_eq) * self.Vl
            Q3 = self.Kla * (X3 - X3_eq) * self.Vl
            Q4 = self.Kla * (X4 - X4_eq) * self.Vl
            Q5 = self.Kla * (X5 - X5_eq) * self.Vl

            dxdt = jnp.array(
                [
                    (1 / self.Vl) * (L * (self.X0 - X1) - Q1),
                    (1 / self.Vg) * (G * (Y2 - Y1) + Q1),
                    (1 / self.Vl) * (L * (X1 - X2) - Q2),
                    (1 / self.Vg) * (G * (Y3 - Y2) + Q2),
                    (1 / self.Vl) * (L * (X2 - X3) - Q3),
                    (1 / self.Vg) * (G * (Y4 - Y3) + Q3),
                    (1 / self.Vl) * (L * (X3 - X4) - Q4),
                    (1 / self.Vg) * (G * (Y5 - Y4) + Q4),
                    (1 / self.Vl) * (L * (X4 - X5) - Q5),
                    (1 / self.Vg) * (G * (self.Y6 - Y5) + Q5),
                ]
            )
            return dxdt
        else:
            if u.shape == (2, 1):
                L, G = u[0], u[1]
            else:
                L, G, self.X0, self.Y6 = u[0], u[1], u[2], u[3]
            ###Model Equations###

            ##States##
            # Xn - Concentration of solute in liquid pase of stage n [kg/m3]
            # Yn - Concentration of solute in gas phase of stage n [kg/m3]

            X1, Y1, X2, Y2, X3, Y3, X4, Y4, X5, Y5 = (
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
            )

            ##Inputs##
            # L - Liquid flowrate m3/hr
            # G - Gas flowrate m3/hr

            X1_eq = (Y1**self.eq_exponent) / self.m
            X2_eq = (Y2**self.eq_exponent) / self.m
            X3_eq = (Y3**self.eq_exponent) / self.m
            X4_eq = (Y4**self.eq_exponent) / self.m
            X5_eq = (Y5**self.eq_exponent) / self.m

            Q1 = self.Kla * (X1 - X1_eq) * self.Vl
            Q2 = self.Kla * (X2 - X2_eq) * self.Vl
            Q3 = self.Kla * (X3 - X3_eq) * self.Vl
            Q4 = self.Kla * (X4 - X4_eq) * self.Vl
            Q5 = self.Kla * (X5 - X5_eq) * self.Vl

            dxdt = [
                (1 / self.Vl) * (L * (self.X0 - X1) - Q1),
                (1 / self.Vg) * (G * (Y2 - Y1) + Q1),
                (1 / self.Vl) * (L * (X1 - X2) - Q2),
                (1 / self.Vg) * (G * (Y3 - Y2) + Q2),
                (1 / self.Vl) * (L * (X2 - X3) - Q3),
                (1 / self.Vg) * (G * (Y4 - Y3) + Q3),
                (1 / self.Vl) * (L * (X3 - X4) - Q4),
                (1 / self.Vg) * (G * (Y5 - Y4) + Q4),
                (1 / self.Vl) * (L * (X4 - X5) - Q5),
                (1 / self.Vg) * (G * (self.Y6 - Y5) + Q5),
            ]

            return dxdt

    def info(self):
        # Return a dictionary with the model information
        info = {
            "parameters": self.__dict__.copy(),
            "states": ["X1", "Y1", "X2", "Y2", "X3", "Y3", "X4", "Y4", "X5", "Y5"],
            "inputs": ["L", "G"],
            "disturbances": ["X0", "Y6"],
        }
        info["parameters"].pop(
            "int_method", None
        )  # Remove 'int_method' from the dictionary since it is not a parameter of the model
        return info


# ==== Bang-Bang Control Model ====#
@dataclass(frozen=False, kw_only=True)
class nonsmooth_control_ode:
    # Parameters
    int_method: str = "jax"
    a_11: float = 0
    a_12: float = 1
    a_21: float = -2
    a_22: float = -3
    b_1: float = 0
    b_2: float = 1

    def __call__(self, x, u):
        # JAX requires jnp functions and arrays hence two versions
        if self.int_method == "jax":
            # states
            x1, x2 = x[0], x[1]

            # ode system
            dxdt = jnp.array(
                [
                    self.a_11 * x1 + self.a_12 * x2 + self.b_1 * u,
                    self.a_21 * x1 + self.a_22 * x2 + self.b_2 * u,
                ]
            )

            return dxdt
        else:
            # states
            x1, x2 = x[0], x[1]

            dxdt = [
                self.a_11 * x1 + self.a_12 * x2 + self.b_1 * u,
                self.a_21 * x1 + self.a_22 * x2 + self.b_2 * u,
            ]

            return dxdt

    def info(self):
        # Return a dictionary with the model information
        info = {
            "parameters": self.__dict__.copy(),
            "states": ["X1", "X2"],
            "inputs": ["U"],
            "disturbances": ["None"],
        }
        info["parameters"].pop(
            "int_method", None
        )  # Remove 'int_method' from the dictionary since it is not a parameter of the model
        return info
