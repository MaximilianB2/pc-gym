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
class cstr:
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
class first_order_system:
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
class multistage_extraction:
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
class nonsmooth_control:
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
@dataclass(frozen=False, kw_only=True)
class RSR:
    # Parameters
    rho: float = 1.0  # Liquid density
    alpha_1: float = 90.0  # Volatility
    k_1: float = 0.0167  # Rate constant
    k_2: float = 0.0167  # Rate constant
    A_R: float = 5.0  # Vessel area
    A_M: float = 10.0  # Vessel area
    A_B: float = 5.0  # Vessel area
    x1_O: float = 1.00  # Initial molar liquid fraction of component 1

    def __call__(self, x, u):
        # States
        H_R, x1_R, x2_R, x3_R, H_M, x1_M, x2_M, x3_M, H_B, x1_B, x2_B, x3_B = x

        # Inputs
        F_O, F_R, F_M, B, D = u

        # Calculate distillate composition
        x1_D = (x1_B * self.alpha_1) / (1 - x1_B + x1_B * self.alpha_1)
        x2_D = 1 - x1_D

        dxdt = [
            (1 / (self.rho * self.A_R)) * (F_O + D - F_R),
            ((F_O * (self.x1_O - x1_R) + D * (x1_D - x1_R)) / (self.rho * self.A_R * H_R)) - self.k_1 * x1_R,
            ((-F_O * x2_R + D * (x2_D - x2_R)) / (self.rho * self.A_R * H_R)) + self.k_1 * x1_R - self.k_2 * x2_R,
            ((-x3_R * (F_O + D)) / (self.rho * self.A_R * H_R)) + self.k_2 * x2_R,
            (1 / (self.rho * self.A_M)) * (F_R - F_M),
            ((F_R) / (self.rho * self.A_M * H_M)) * (x1_R - x1_M),
            ((F_R) / (self.rho * self.A_M * H_M)) * (x2_R - x2_M),
            ((F_R) / (self.rho * self.A_M * H_M)) * (x3_R - x3_M),
            (1 / (self.rho * self.A_B)) * (F_M - B - D),
            (1 / (self.rho * self.A_B * H_B)) * (F_M * (x1_M - x1_B) - D * (x1_D - x1_B)),
            (1 / (self.rho * self.A_B * H_B)) * (F_M * (x2_M - x2_B) - D * (x2_D - x2_B)),
            (1 / (self.rho * self.A_B * H_B)) * (F_M * (x3_M - x3_B) + D * (x3_B))
        ]

        return dxdt

    def info(self):
        # Return a dictionary with the model information
        info = {
            "parameters": self.__dict__.copy(),
            "states": ["H_R", "x1_R", "x2_R", "x3_R", "H_M", "x1_M", "x2_M", "x3_M", "H_B", "x1_B", "x2_B", "x3_B"],
            "inputs": ["F_O", "F_R", "F_M", "B", "D"],
            "disturbances": []
        }
        return info
    
@dataclass(frozen=False, kw_only=True)
class cstr_series_recycle:
    # Parameters
    C_O: float = 97.35  # mol/m3
    T_O: float = 298  # K
    V1: float = 1e-3  # m3
    V2: float = 2e-3  # m3
    U1A1: float = 0.461  # kJ/s*K
    U2A2: float = 0.732  # kJ/s*K
    rho: float = 1.05e3  # kg/m3
    cp: float = 3.766  # kJ/kg*K
    k: float = 3.118e5  # s-1
    E: float = 46.14  # kJ/mol
    deltaH: float = 58.41  # kJ/mol
    R: float = 8.3145e-3  # kJ/mol K

    def __call__(self, x, u):
        # States
        C1, T1, C2, T2 = x

        # Inputs
        F, L, Tc1, Tc2 = u

        dxdt = [
            (self.C_O / self.V1) * F + (1 / self.V1) * L * C2 - (1 / self.V1) * (F + L) * C1 - self.k * C1 * np.exp((-self.E / (self.R * T1))),
            (self.T_O / self.V1) * F + (1 / self.V1) * L * T2 - ((self.U1A1) / (self.V1 * self.rho * self.cp)) * (T1 - Tc1) - (1 / self.V1) * (F + L) * T1 + ((self.k * (-self.deltaH)) / (self.rho * self.cp)) * C1 * np.exp((-self.E / (self.R * T1))),
            (1 / self.V2) * (F + L) * (C1 - C2) - self.k * C2 * np.exp((-self.E / (self.R * T2))),
            (1 / self.V2) * (F + L) * (T1 - T2) - ((self.U2A2) / (self.V2 * self.rho * self.cp)) * (T2 - Tc2) + ((self.k * (-self.deltaH)) / (self.rho * self.cp)) * C2 * np.exp((-self.E / (self.R * T2)))
        ]

        return dxdt

    def info(self):
        # Return a dictionary with the model information
        info = {
            "parameters": self.__dict__.copy(),
            "states": ["C1", "T1", "C2", "T2"],
            "inputs": ["F", "L", "Tc1", "Tc2"],
            "disturbances": []
        }
        return info
@dataclass(frozen=False, kw_only=True)
class distillation_column:
    # Parameters
    D: float = 100.0  # kmol/hr
    q: float = 1.0  # Feed quality (q=1 is saturated liquid)
    alpha: float = 5.0  # Relative volatility of more volatile component
    X_feed: float = 0.2
    M0: float = 2000.0
    Mb: float = 2000.0
    M: float = 2000.0

    def __call__(self, x, u):
        # States
        X0, X1, X2, X3, Xf, X4, X5, X6, Xb = x

        # Inputs
        R, F = u

        L = R * self.D
        V = (R + 1) * self.D
        L_dash = L + self.q * F
        V_dash = V + (1 - self.q) * F
        W = F - self.D

        Y1 = (self.alpha * X1) / (1 + (self.alpha - 1) * X1)
        Y2 = (self.alpha * X2) / (1 + (self.alpha - 1) * X2)
        Y3 = (self.alpha * X3) / (1 + (self.alpha - 1) * X3)
        Yf = (self.alpha * Xf) / (1 + (self.alpha - 1) * Xf)
        Y4 = (self.alpha * X4) / (1 + (self.alpha - 1) * X4)
        Y5 = (self.alpha * X5) / (1 + (self.alpha - 1) * X5)
        Y6 = (self.alpha * X6) / (1 + (self.alpha - 1) * X6)
        Yb = (self.alpha * Xb) / (1 + (self.alpha - 1) * Xb)

        dxdt = [
            (1 / self.M0) * ((V * Y1) - (L + self.D) * X0),  # reflux drum
            (1 / self.M) * (L * (X0 - X1) + V * (Y2 - Y1)),  # Plate 1
            (1 / self.M) * (L * (X1 - X2) + V * (Y3 - Y2)),  # Plate 2
            (1 / self.M) * (L * (X2 - X3) + V * (Yf - Y3)),  # Plate 3
            (1 / self.M) * (L * X3 - L_dash * Xf + V_dash * Y4 - V * Yf + F * self.X_feed),  # Feed plate
            (1 / self.M) * (L_dash * (Xf - X4) + V_dash * (Y5 - Y4)),  # Plate 4
            (1 / self.M) * (L_dash * (X4 - X5) + V_dash * (Y6 - Y5)),  # Plate 5
            (1 / self.M) * (L_dash * (X5 - X6) + V_dash * (Yb - Y6)),  # Plate 6
            (1 / self.Mb) * (L_dash * X6 - W * Xb - V_dash * Yb)  # Reboiler
        ]

        return dxdt

    def info(self):
        # Return a dictionary with the model information
        info = {
            "parameters": self.__dict__.copy(),
            "states": ["X0", "X1", "X2", "X3", "Xf", "X4", "X5", "X6", "Xb"],
            "inputs": ["R", "F"],
            "disturbances": []
        }
        return info

@dataclass(frozen=False, kw_only=True)
class multistage_extraction_reactive:
    # Parameters
    Vl: float = 5.0  # Liquid volume in each stage
    Vg: float = 5.0  # Gas volume in each stage
    m: float = 1.0  # Equilibrium constant [-]
    Kla: float = 0.01  # Mass transfer capacity constant 1/hr
    k: float = 0.1  # Reaction equilibrium constant
    eq_exponent: float = 2.0  # Change the nonlinearity of the equilibrium relationship
    XA0: float = 2.00  # Feed concentration of component A in liquid phase
    YA6: float = 0.00  # Feed conc of component A in gas phase
    YB6: float = 2.00  # Feed conc of component B in gas phase
    YC6: float = 0.00  # Feed conc of component C in gas phase

    def __call__(self, x, u):
        # States
        XA1, YA1, YB1, YC1, XA2, YA2, YB2, YC2, XA3, YA3, YB3, YC3, XA4, YA4, YB4, YC4, XA5, YA5, YB5, YC5 = x

        # Inputs
        L, G = u

        XA1_eq = ((YA1 ** self.eq_exponent) / self.m)
        XA2_eq = ((YA2 ** self.eq_exponent) / self.m)
        XA3_eq = ((YA3 ** self.eq_exponent) / self.m)
        XA4_eq = ((YA4 ** self.eq_exponent) / self.m)
        XA5_eq = ((YA5 ** self.eq_exponent) / self.m)

        Q1 = self.Kla * (XA1 - XA1_eq) * self.Vl
        Q2 = self.Kla * (XA2 - XA2_eq) * self.Vl
        Q3 = self.Kla * (XA3 - XA3_eq) * self.Vl
        Q4 = self.Kla * (XA4 - XA4_eq) * self.Vl
        Q5 = self.Kla * (XA5 - XA5_eq) * self.Vl

        r1 = self.k * YA1 * YB1
        r2 = self.k * YA2 * YB2
        r3 = self.k * YA3 * YB3
        r4 = self.k * YA4 * YB4
        r5 = self.k * YA5 * YB5

        dxdt = [
            (1 / self.Vl) * (L * (self.XA0 - XA1) - Q1),
            (1 / self.Vg) * (G * (YA2 - YA1) + Q1 - r1 * self.Vg),
            (1 / self.Vg) * (G * (YB2 - YB1) - r1 * self.Vg),
            (1 / self.Vg) * (G * (YC2 - YC1) + r1 * self.Vg),
            (1 / self.Vl) * (L * (XA1 - XA2) - Q2),
            (1 / self.Vg) * (G * (YA3 - YA2) + Q2 - r2 * self.Vg),
            (1 / self.Vg) * (G * (YB3 - YB2) - r2 * self.Vg),
            (1 / self.Vg) * (G * (YC3 - YC2) + r2 * self.Vg),
            (1 / self.Vl) * (L * (XA2 - XA3) - Q3),
            (1 / self.Vg) * (G * (YA4 - YA3) + Q3 - r3 * self.Vg),
            (1 / self.Vg) * (G * (YB4 - YB3) - r3 * self.Vg),
            (1 / self.Vg) * (G * (YC4 - YC3) + r3 * self.Vg),
            (1 / self.Vl) * (L * (XA3 - XA4) - Q4),
            (1 / self.Vg) * (G * (YA5 - YA4) + Q4 - r4 * self.Vg),
            (1 / self.Vg) * (G * (YB5 - YB4) - r4 * self.Vg),
            (1 / self.Vg) * (G * (YC5 - YC4) + r4 * self.Vg),
            (1 / self.Vl) * (L * (XA4 - XA5) - Q5),
            (1 / self.Vg) * (G * (self.YA6 - YA5) + Q5 - r5 * self.Vg),
            (1 / self.Vg) * (G * (self.YB6 - YB5) - r5 * self.Vg),
            (1 / self.Vg) * (G * (self.YC6 - YC5) + r5 * self.Vg),
        ]

        return dxdt

    def info(self):
        # Return a dictionary with the model information
        info = {
            "parameters": self.__dict__.copy(),
            "states": ["XA1", "YA1", "YB1", "YC1", "XA2", "YA2", "YB2", "YC2", "XA3", "YA3", "YB3", "YC3", "XA4", "YA4", "YB4", "YC4", "XA5", "YA5", "YB5", "YC5"],
            "inputs": ["L", "G"],
            "disturbances": []
        }
        return info
    
@dataclass(frozen=False, kw_only=True)
class four_tank:
    # Parameters
    g: float = 9.81  # Acceleration due to gravity [m/s2]
    gamma_1: float = 0.2  # Fraction bypassed by valve to tank 1 [-]
    gamma_2: float = 0.2  # Fraction bypassed by valve to tank 2 [-]
    k1: float = 0.00085  # 1st pump gain [m3/Volts S]
    k2: float = 0.00095  # 2nd pump gain [m3/Volts S]
    a1: float = 0.0035  # Cross sectional area of outlet of tank 1 [m2]
    a2: float = 0.0030  # Cross sectional area of outlet of tank 2 [m2]
    a3: float = 0.0020  # Cross sectional area of outlet of tank 3 [m2]
    a4: float = 0.0025  # Cross sectional area of outlet of tank 4 [m2]
    A1: float = 1  # Cross sectional area of tank 1 [m2]
    A2: float = 1  # Cross sectional area of tank 2 [m2]
    A3: float = 1  # Cross sectional area of tank 3 [m2]
    A4: float = 1  # Cross sectional area of tank 4 [m2]

    def __call__(self, x, u):
        # States
        h1, h2, h3, h4 = x

        # Inputs
        v1, v2 = u

        dxdt = [
            (-self.a1 / self.A1) * np.sqrt(2 * self.g * h1) + (self.a3 / self.A1) * np.sqrt(2 * self.g * h3) + ((self.gamma_1 * self.k1) / (self.A1)) * v1,
            (-self.a2 / self.A2) * np.sqrt(2 * self.g * h2) + (self.a4 / self.A2) * np.sqrt(2 * self.g * h4) + ((self.gamma_2 * self.k2) / (self.A2)) * v2,
            (-self.a3 / self.A3) * np.sqrt(2 * self.g * h3) + (((1 - self.gamma_2) * self.k2) / (self.A3)) * v2,
            (-self.a4 / self.A4) * np.sqrt(2 * self.g * h4) + (((1 - self.gamma_1) * self.k1) / (self.A4)) * v1,
        ]
        return dxdt

    def info(self):
        info = {
            "parameters": self.__dict__.copy(),
            "states": ["h1", "h2", "h3", "h4"],
            "inputs": ["v1", "v2"],
            
        }
        return info


@dataclass(frozen=False, kw_only=True)
class heat_exchanger:
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

    def __call__(self, x, u):
        # States
        Tt1, Tm1, Ts1, Tt2, Tm2, Ts2, Tt3, Tm3, Ts3, Tt4, Tm4, Ts4, Tt5, Tm5, Ts5, Tt6, Tm6, Ts6, Tt7, Tm7, Ts7, Tt8, Tm8, Ts8 = x

        # Inputs
        Ft, Fs, Tt0, Ts9 = u

        Vt = self.L * np.pi * self.Dt**2
        At = self.L * np.pi * self.Dt
        Vm = self.L * np.pi * (self.Dm**2 - self.Dt**2)
        Am = self.L * np.pi * self.Dm
        Vs = self.L * np.pi * (self.Ds**2 - self.Dm**2)

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

        dxdt = [
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

        return dxdt

    def info(self):
        # Return a dictionary with the model information
        info = {
            "parameters": self.__dict__.copy(),
            "states": ['Tt1', 'Tm1', 'Ts1', 'Tt2', 'Tm2', 'Ts2', 'Tt3', 'Tm3', 'Ts3', 'Tt4', 'Tm4', 'Ts4', 'Tt5', 'Tm5', 'Ts5', 'Tt6', 'Tm6', 'Ts6', 'Tt7', 'Tm7', 'Ts7', 'Tt8', 'Tm8', 'Ts8'],
            "inputs":['Ft', 'Fs', 'Tt0', 'Ts9']
        }
        return info
@dataclass(frozen=False, kw_only=True)
class biofilm_reactor:
    # Parameters
    V: float = 1.0  # Volume of one reactor stage [L]
    Va: float = 1.0  # Volume of absorber tank [L]
    Kla: float = 1.0  # Transfer coefficient [hr]
    m: float = 0.5  # Equilibrium constant [-]
    eq_exponent: float = 1.0
    O_air: float = 1.0  # Concentration of oxygen in air [mg/L]
    vm_1: float = 1.0  # Maximum velocity through fluidized bed for reaction 1 [mg/L hr]
    vm_2: float = 1.0  # Maximum velocity through fluidized bed for reaction 2 [mg/L hr]
    K1: float = 1.0  # Equilibrium constant for reaction 1 (Saturation constant for ammonia in reaction 1) [mg/L]
    K2: float = 1.0  # Equilibrium constant for reaction 2 (Saturation constant for ammonia in reaction 2) [mg/L]
    KO_1: float = 1.0  # Equilibrium constant for oxygen in reaction 1 (saturation constant for oxygen) [mg/L]
    KO_2: float = 1.0  # Equilibrium constant for oxygen in reaction 2 (saturation constant for oxygen) [mg/L]

    def __call__(self, x, u):
        # States
        S1_1, S2_1, S3_1, O_1, S1_2, S2_2, S3_2, O_2, S1_3, S2_3, S3_3, O_3, S1_A, S2_A, S3_A, O_A = x

        # Inputs
        F, Fr, S1_F, S2_F, S3_F = u

        # Reaction rates and stoichiometry
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

        O_Aeq = ((self.O_air ** self.eq_exponent) / self.m)  # Oxygen dissolved in water in equilibrium with air

        dxdt = [
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
            (Fr / self.Va) * (O_3 - O_A) + self.Kla * (O_Aeq - O_A)
        ]

        return dxdt

    def info(self):
        # Return a dictionary with the model information
        info = {
            "parameters": self.__dict__.copy(),
            "states": ["S1_1", "S2_1", "S3_1", "O_1", "S1_2", "S2_2", "S3_2", "O_2", "S1_3", "S2_3", "S3_3", "O_3", "S1_A", "S2_A", "S3_A", "O_A"],
            "inputs": ["F", "Fr", "S1_F", "S2_F", "S3_F"],
            "disturbances": []
        }
        return info

@dataclass(frozen=False, kw_only=True)
class polymerisation_reactor:
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

    def __call__(self, x, u):
        # States
        T, M, I = x

        # Inputs
        F, Tf, Mf, If = u

        kp = self.Ap * np.exp(-self.Ep_over_R / T)
        kd = self.Ad * np.exp(-self.Ed_over_R / T)
        kt = self.At * np.exp(-self.Et_over_R / T)

        ri = 2 * self.f * kd * I
        rp = kp * ((self.f * kd * I) / kt) ** 0.5

        dxdt = [
            (F / self.V) * (Tf - T) + ((-self.deltaHp) / (self.rho * self.cp)) * rp,
            (F / self.V) * (Mf - M) - rp,
            (F / self.V) * (If - I) - ri
        ]

        return dxdt

    def info(self):
        # Return a dictionary with the model information
        info = {
            "parameters": self.__dict__.copy(),
            "states": ["T", "M", "I"],
            "inputs": ["F", "Tf", "Mf", "If"],
            "disturbances": []
        }
        return info
@dataclass(frozen=False, kw_only=True)
class crystallization:
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
    ro: float = 2.658  # g/cm³
    V: float = 0.0005  # m³
    ro_t: float = 997  # kg/m³
    Cp_t: float = 4.117  # kJ/kg*K
    UA: float = 0.2154  # kJ/min*K
    int_method: str = "jax"

    def __call__(self, x, u):
        mu0, mu1, mu2, mu3, conc, temp = x
        Tc = u[0]

        if self.int_method == "jax":
            Ceq = -686.2686 + 3.579165 * jnp(temp) - 0.00292874 * jnp(temp) ** 2  # g/L
            S = jnp(conc) * 1e3 - Ceq  # g/L
            B0 = self.ka * jnp.exp(self.kb / temp) * (S ** 2) ** (self.kc / 2) * ((mu3 ** 2) ** (self.kd / 2))  # /(cm³*min)
            Ginf = self.kg * jnp.exp(self.k1 / temp) * (S ** 2) ** (self.k2 / 2)  # [G] = [Ginf] = cm/min

            dmi0dt = B0
            dmi1dt = Ginf * (self.a * mu0 + self.b * mu1 * 1e-4) * 1e4
            dmi2dt = 2 * Ginf * (self.a * mu1 * 1e-4 + self.b * mu2 * 1e-8) * 1e8
            dmi3dt = 3 * Ginf * (self.a * mu2 * 1e-8 + self.b * mu3 * 1e-12) * 1e12
            dcdt = -0.5 * self.ro * self.alfa * Ginf * (self.a * mu2 * 1e-8 + self.b * mu3 * 1e-12)
            dTdt = self.UA * (Tc - temp) / (self.V * self.ro_t * self.Cp_t)

            dxdt = jnp.array([dmi0dt, dmi1dt, dmi2dt, dmi3dt, dcdt, dTdt])
        else:
            Ceq = -686.2686 + 3.579165 * temp - 0.00292874 * temp ** 2  # g/L
            S = conc * 1e3 - Ceq  # g/L
            B0 = self.ka * np.exp(self.kb / temp) * (S ** 2) ** (self.kc / 2) * ((mu3 ** 2) ** (self.kd / 2))  # /(cm³*min)
            Ginf = self.kg * np.exp(self.k1 / temp) * (S ** 2) ** (self.k2 / 2)  # [G] = [Ginf] = cm/min

            dmi0dt = B0
            dmi1dt = Ginf * (self.a * mu0 + self.b * mu1 * 1e-4) * 1e4
            dmi2dt = 2 * Ginf * (self.a * mu1 * 1e-4 + self.b * mu2 * 1e-8) * 1e8
            dmi3dt = 3 * Ginf * (self.a * mu2 * 1e-8 + self.b * mu3 * 1e-12) * 1e12
            dcdt = -0.5 * self.ro * self.alfa * Ginf * (self.a * mu2 * 1e-8 + self.b * mu3 * 1e-12)
            dTdt = self.UA * (Tc - temp) / (self.V * self.ro_t * self.Cp_t)

            dxdt = [dmi0dt, dmi1dt, dmi2dt, dmi3dt, dcdt, dTdt]

        return dxdt

    def info(self):
        # Return a dictionary with the model information
        info = {
            "parameters": self.__dict__.copy(),
            "states": ["Mu0", "Mu1", "Mu2", "Mu3", "Conc", "Temp"],
            "inputs": ["Tc"],
            "disturbances": ["ka", "kg", "UA"],
        }
        info["parameters"].pop("int_method", None)  # Remove 'int_method' since it's not a parameter of the model
        return info

@dataclass(frozen=False, kw_only=True)
class FastOffshoreWellsModel:
    # Parameters
    Prt_max: float = 50.0
    Prb_max: float = 270.0
    Ppdg_max: float = 230.0
    Ptt_max: float = 200.0
    Ptb_max: float = 210.0
    Pbh_max: float = 240.0
    Wlout_max: float = 100.0
    Wgout_max: float = 20.0

    Prt_min: float = 10.0
    Prb_min: float = 80.0
    Ppdg_min: float = 180.0
    Ptt_min: float = 120.0
    Ptb_min: float = 160.0
    Pbh_min: float = 190.0
    Wlout_min: float = 0.0
    Wgout_min: float = 0.0

    z_min: float = 0.0
    Wgc_min: float = 0.0
    z_max: float = 1.0
    Wgc_max: float = 2.49

    # Constants
    Rol: float = 900.0
    R: float = 8314.0
    T: float = 298.0
    M: float = 18.0
    g: float = 9.81
    teta: float = math.pi / 4
    Ps: float = 1013250.0
    Pr: float = 2.25e7
    ALFAgw: float = 0.0188
    Romres: float = 891.9523
    L: float = 4497.0
    Lt: float = 1639.0
    La: float = 1118.0
    D: float = 0.152
    Dt: float = 0.150
    Da: float = 0.140
    Hvgl: float = 916.0
    Hpdg: float = 1117.0
    Ht: float = 1279.0
    epsi: float = 1e-10
    A: float = D * D * math.pi / 4
    Vt: float = Lt * Dt * Dt * math.pi / 4
    Va: float = La * Da * Da * math.pi / 4

    # Initial Conditions
    x0 = [9347.5467727, 5347.53586993, 35014.1218045, 2395.94295262, 1788.01439135, 11725.3198923]

    def __init__(self, reference="Rodrigues"):
        self.reference = reference
        if self.reference == 'Rodrigues':
            self.z = 0.84
            self.Wgc = 1.406
            self.mlstill = 2.020e4
            self.Cg = 1e-3
            self.Cout = 1.626e-3
            self.Veb = 1.5e2
            self.E = 4.5e-1
            self.Kw = 4.958e-4
            self.Ka = 1e-3
            self.Vr = 226.74
            self.Kr = 1.2e2
            self.x0 = [9347.5467727, 5347.53586993, 35014.1218045, 2395.94295262, 1788.01439135, 11725.3198923]
        elif self.reference == 'Apio':
            self.z = 0.84
            self.Wgc = 1.406
            self.mlstill = 7.183e1
            self.Cg = 1.329e-3
            self.Cout = 3.975e-3
            self.Veb = 7.047e1
            self.E = 2.095e-1
            self.Kw = 5.936e-4
            self.Ka = 4.236e-5
            self.Vr = 620.364197531
            self.Kr = 2.470e2
            self.x0 = [9347.5467727, 5347.53586993, 35014.1218045, 2395.94295262, 1788.01439135, 11725.3198923]
        elif self.reference == 'Huffner':
            self.z = 0.18
            self.Wgc = 0.6574074074
            self.mlstill = 1957.031302138902
            self.Cg = 0.000205390205
            self.Cout = 0.019677777778
            self.Veb = 83.5090666667
            self.E = 0.571431327160
            self.Kw = 0.000867943572
            self.Ka = 0.000159061894
            self.Vr = 620.364197531
            self.Kr = 131.313519772
            self.x0 = [8955, 5286, 33373, 2197, 1776, 11211]
        elif self.reference == 'Diehl':
            self.z = 0.18
            self.Wgc = 0.6574074074
            self.mlstill = 6.222e1
            self.Cg = 1.137e-3
            self.Cout = 2.039e-3
            self.Veb = 6.098e1
            self.E = 1.545e-1
            self.Kw = 6.876e-4
            self.Ka = 2.293e-5
            self.Vr = 226.74
            self.Kr = 1.269e2
            self.x0 = [9347.5467727, 5347.53586993, 35014.1218045, 2395.94295262, 1788.01439135, 11725.3198923]

    def __call__(self, x, u):
        # States
        m_Ga, m_Gt, m_Lt, m_Gb, m_Gr, m_Lr = x

        # Inputs
        z, Wgc = u

        Peb = m_Ga * self.R * self.T / (self.M * self.Veb)
        Prt = m_Gt * self.R * self.T / (self.M * (self.Vr - (m_Lt + self.mlstill) / self.Rol))
        Prb = Prt + (m_Lt + self.mlstill) * self.g * math.sin(self.teta) / self.A
        ALFAg = m_Gt / (m_Gt + m_Lt)
        ALFAl = 1 - ALFAg
        Wout = self.Cout * z * math.sqrt(self.Rol * ((Prt - self.Ps) + math.sqrt((Prt - self.Ps) ** 2 + self.epsi))) * (1 / 2)
        Wlout = ALFAl * Wout
        Wgout = ALFAg * Wout
        Wg = self.Cg * ((Peb - Prb) + math.sqrt((Peb - Prb) ** 2 + self.epsi)) * (1 / 2)

        Vgt = self.Vt - m_Lr / self.Rol
        ROgt = m_Gr / Vgt
        ROmt = (m_Gr + m_Lr) / self.Vt
        Ptt = ROgt * self.R * self.T / self.M
        Ptb = Ptt + ROmt * self.g * self.Hvgl
        Ppdg = Ptb + self.Romres * self.g * (self.Hpdg - self.Hvgl)
        Pbh = Ppdg + self.Romres * self.g * (self.Ht - self.Hpdg)
        ALFAgt = m_Gr / (m_Lr + m_Gr)
        Wwh = self.Kw * math.sqrt(self.Rol * ((Ptt - Prb) + math.sqrt((Ptt - Prb) ** 2 + self.epsi))) * (1 / 2)
        Wwhg = Wwh * ALFAgt
        Wwhl = Wwh * (1 - ALFAgt)
        Wr = self.Kr * (1 - 0.2 * Pbh / self.Pr - 0.8 * (Pbh / self.Pr) ** 2)

        Pai = ((self.R * self.T / (self.Va * self.M)) + (self.g * self.La / self.Va)) * m_Gb
        ROai = self.M * Pai / (self.R * self.T)
        Wiv = self.Ka * math.sqrt(ROai * ((Pai - Ptb) + math.sqrt((Pai - Ptb) ** 2 + self.epsi))) * (1 / 2)

        dx1 = (1 - self.E) * (Wwhg) - Wg
        dx2 = self.E * (Wwhg) + Wg - Wgout
        dx3 = Wwhl - Wlout
        dx4 = Wgc - Wiv
        dx5 = Wr * self.ALFAgw + Wiv - Wwhg
        dx6 = Wr * (1 - self.ALFAgw) - Wwhl

        dxdt = np.array([dx1, dx2, dx3, dx4, dx5, dx6])

        return dxdt

    def info(self):
        # Return a dictionary with the model information
        info = {
            "parameters": self.__dict__.copy(),
            "states": ["m_Ga", "m_Gt", "m_Lt", "m_Gb", "m_Gr", "m_Lr"],
            "inputs": ["z", "Wgc"],
            "disturbances": []
        }
        return info
