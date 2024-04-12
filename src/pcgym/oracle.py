from casadi import MX, Function, integrator, Opti, vertcat, sum1, sum2
import numpy as np


class oracle:
    """
    Oracle Class - Class to solve the optimal control problem with perfect
    knowledge of the environment.
    Oracle is a nonlinear model predictive controller (nMPC),
    using the multiple shooting method.

    Inputs: Env

    Outputs: Optimal control and state trajectories
    """

    def __init__(self, env, env_params, MPC_params=False):
        self.env_params = env_params
        self.env_params["integration_method"] = "casadi"
        self.env = env(env_params)

        self.x0 = env_params["x0"]
        self.T = self.env.tsim
        if not MPC_params:
            self.N = 5
            self.R = 0.05
        else:
            self.N = MPC_params["N"]  # Horizon length
            self.R = MPC_params["R"]  # Control penalty scaling factors
        self.model_info = self.env.model.info()

    def model_gen(self):
        """
        Generates a model for the given environment.

        Returns:
        f: A casadi function that can be used to solve the differential equations defined by the model.
        """

        self.u = MX.sym("u", self.env.Nu)
        self.x = MX.sym("x", self.env.Nx_oracle)
        dxdt = self.env.model(self.x, self.u)
        dxdt = vertcat(*dxdt)
        f = Function("f", [self.x, self.u], [dxdt], ["x", "u"], ["dxdt"])
        return f

    def integrator_gen(self):
        """
        Generates an integrator object for the given model.

        Returns:
        F: A casadi function that can be used to integrate the model over a given time horizon.
        """

        f = self.model_gen()
        tf = self.env.dt
        t0 = 0
        dae = {"x": self.x, "p": self.u, "ode": f(self.x, self.u)}
        opts = {"simplify": True, "number_of_finite_elements": 4}
        intg = integrator("intg", "rk", dae, t0, tf, opts)
        res = intg(x0=self.x, p=self.u)
        x_next = res["xf"]
        F = Function("F", [self.x, self.u], [x_next], ["x", "u"], ["x_next"])
        return F

    def disturbance_index(self):
        """
        Generates the indices of when the disturbance or setpoint value changes.

        Inputs: self

        Returns: index of when either the disturbance or setpoint value changes.

        """

        index = []
        if self.env_params.get("disturbances") is not None:
            for key in self.env_params["disturbances"]:
                disturbance = self.env_params["disturbances"][key]
                for i in range(disturbance.shape[0] - 1):
                    if disturbance[i] != disturbance[i + 1]:
                        index.append(i + 1)
        for key in self.env_params["SP"]:
            SP = self.env_params["SP"][key]
            for i in range(len(SP) - 1):
                if SP[i] != SP[i + 1]:
                    index.append(i + 1)

        index = list(set(index))
        return index

    def ocp(self, t_step):
        """
        Solves an optimal control problem (OCP) using the IPOPT solver.

        Returns:
        - M: A function that takes current state x_0 (p) and returns the optimal control input u.

        """

        opti = Opti()
        F = self.integrator_gen()
        x = opti.variable(self.env.Nx_oracle, self.N + 1)
        u = opti.variable(self.env.Nu, self.N)
        p = opti.parameter(self.env.Nx_oracle, 1)
        setpoint = opti.parameter(len(self.env_params["SP"]), self.N + 1)

        # Cost function sum of squared error plus control penalty.
        # Both states and controls are normalised to errors equally
        cost = 0
        Sp_i = 0
        for k in self.env_params["SP"]:
            i = self.model_info["states"].index(k)

            o_space_low = self.env_params["o_space"]["low"][i] * np.ones(
                (1, self.N + 1)
            )
            o_space_high = self.env_params["o_space"]["high"][i] * np.ones(
                (1, self.N + 1)
            )
            x_normalized = (x[i, :] - o_space_low) / (o_space_high - o_space_low)
            setpoint_normalized = (setpoint[Sp_i, :] - o_space_low) / (
                o_space_high - o_space_low
            )

            r_scale = self.env_params.get(
                "r_scale", {}
            )  # if no r_scale: set r_scale to 1
            cost += sum1(sum2((x_normalized - setpoint_normalized) ** 2)) * r_scale.get(
                k, 1
            )
            Sp_i += 1
        u_normalized = (u - self.env_params["a_space"]["low"]) / (
            self.env_params["a_space"]["high"] - self.env_params["a_space"]["low"]
        )

        # Add the control cost
        cost += self.R * sum1(sum2(u_normalized**2))

        opti.minimize(cost)

        # Dynamics
        for k in range(self.N):
            opti.subject_to(x[:, k + 1] == F(x[:, k], u[:, k]))

        # Control constraints

        for i in range(self.env.Nu - self.env.Nd_model):
            opti.subject_to(u[i, :] >= self.env_params["a_space"]["low"][i])
            opti.subject_to(u[i, :] <= self.env_params["a_space"]["high"][i])

        # Define disturbance as a control input equality constraint
        # TODO: Add an option to foresee any disturbance.
        if self.env_params.get("disturbances") is not None:
            for i, k in enumerate(self.env.model.info()["disturbances"], start=0):
                if k in self.env.disturbances.keys():
                    opti.subject_to(
                        u[
                            self.env.Nu
                            - len(self.env.model.info()["disturbances"])
                            + i,
                            :,
                        ]
                        == self.env.disturbances[k][t_step]
                    )  # Add disturbance to control vector
                    opti.set_initial(
                        u[
                            self.env.Nu
                            - len(self.env.model.info()["disturbances"])
                            + i,
                            :,
                        ],
                        self.env.disturbances[k][t_step],
                    )
                else:
                    opti.subject_to(
                        u[
                            self.env.Nu
                            - len(self.env.model.info()["disturbances"])
                            + i,
                            :,
                        ]
                        == self.model_info["parameters"][str(k)]
                    )  # if there is no disturbance at this timestep, use the default value
                    opti.set_initial(
                        u[
                            self.env.Nu
                            - len(self.env.model.info()["disturbances"])
                            + i,
                            :,
                        ],
                        self.model_info["parameters"][str(k)],
                    )

        # Initial condition
        opti.subject_to(x[:, 0] == p)

        # Add user-defined constraint
        if self.env_params.get("constraints") is not None:
            for k in self.env_params["constraints"]:
                for j in range(len(k)):
                    if self.env_params["cons_type"][k][j] == "<=":
                        opti.subject_to(
                            x[self.model_info["states"].index(k), :]
                            <= self.env_params["constraints"][k][j]
                        )
                    elif self.env_params["cons_type"][k][j] == ">=":
                        opti.subject_to(
                            x[self.model_info["states"].index(k), :]
                            >= self.env_params["constraints"][k][j]
                        )
                    else:
                        raise ValueError("Invalid constraint type")

        # Define the setpoint for the cost function
        SP_i = np.fromiter(
            {k: v[t_step] for k, v in self.env_params["SP"].items()}.values(),
            dtype=float,
        )
        setpoint_value = SP_i * np.ones((self.N + 1, 1))
        opti.set_value(setpoint, setpoint_value.T)

        # Initial values
        opti.set_value(p, self.x0[: self.env.Nx_oracle])
        initial_x_values = np.zeros((self.env.Nx_oracle, self.N + 1))
        initial_x_values = (
            self.x0[: self.env.Nx_oracle] * np.ones((self.N + 1, self.env.Nx_oracle))
        ).T
        opti.set_initial(x, initial_x_values)
        for i in range(self.env.Nu - self.env.Nd_model):
            opti.set_initial(
                u[i, :], self.env_params["a_space"]["low"][i] * np.ones((1, self.N))
            )

        # Silence the solver
        opts = {
            "ipopt.print_level": 0,
            "ipopt.sb": "no",
            "print_time": 0,
            "ipopt.print_user_options": "no",
        }

        opti.solver("ipopt", opts)

        # Make the opti object a function
        M = opti.to_function("M", [p], [u[:, 1]], ["p"], ["u"])
        return M

    def mpc(self):
        """
        Solves a model predictive control problem (MPC) using the optimal control problem (OCP) solver.

        Returns:
        - x_opt: Optimal state trajectory
        - u_opt: Optimal control trajectory
        """

        regen_index = self.disturbance_index()

        M = self.ocp(t_step=0)
        F = self.integrator_gen()

        u_log = np.zeros((self.env.Nu, self.env.N))
        x_log = np.zeros((self.env.Nx_oracle, self.env.N))
        x = np.array(self.x0[: self.env.Nx_oracle])
        for i in range(self.env.N):
            if i - 1 in regen_index:
                M = self.ocp(t_step=i)
            try:
                x_log[:, i] = x
            except Exception:
                x_log[:, i] = x.reshape(-1)

            if self.env_params.get("noise", False):
                noise_percentage = self.env_params.get("noise_percentage", 0)
                try:
                    x += (
                        np.random.normal(0, 1, (self.env.Nx_oracle))
                        * x
                        * noise_percentage
                    )

                except Exception:
                    x += (
                        np.random.normal(0, 1, (self.env.Nx_oracle, 1))
                        * x
                        * noise_percentage
                    )
            u = M(x).full()
            u_log[:, i] = u[0]
            x = F(x, u).full()
        return x_log, u_log
