import numpy as np
import do_mpc
from casadi import *

class oracle:
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
            self.N = MPC_params["N"]
            self.R = MPC_params["R"]
        self.model_info = self.env.model.info()
        
        self.integral_error = np.zeros(len(self.env_params["SP"]))

    def setup_mpc(self):
        model_type = 'continuous'
        model = do_mpc.model.Model(model_type)

        # States
        x = model.set_variable(var_type='_x', var_name='x', shape=(self.env.Nx_oracle, 1))

        # Inputs
        u = model.set_variable(var_type='_u', var_name='u', shape=(self.env.Nu, 1))

        # Set point (as a parameter)
        SP = model.set_variable(var_type='_p', var_name='SP', shape=(self.N, 1))

        # System dynamics
        dx_list = self.env.model(x, u)
        dx = vertcat(*dx_list)  # Convert list to CasADi symbolic expression
        model.set_rhs('x', dx)
        # Setup the model
        
        model.setup()

        # Setup MPC
        mpc = do_mpc.controller.MPC(model)
        setup_mpc = {
            'n_horizon': self.N,
            't_step': self.env.dt,
            'n_robust': 0,
            'store_full_solution': True,
        }
        mpc.set_param(**setup_mpc)

        # Objective function
        # Stage cost (lterm)
        lterm = 0
        for i, sp_key in enumerate(self.env_params["SP"]):
            state_index = self.model_info["states"].index(sp_key)
            lterm += (x[state_index] - SP[i])**2

        u_normalized = (u - self.env_params["a_space"]["low"]) / (self.env_params["a_space"]["high"] - self.env_params["a_space"]["low"])
        lterm += sum1(self.R * u_normalized**2)

        # Terminal cost (mterm) - only includes state costs
        mterm = 0
        for i, sp_key in enumerate(self.env_params["SP"]):
            state_index = self.model_info["states"].index(sp_key)
            mterm += (x[state_index] - SP[i])**2

        mpc.set_objective(lterm=lterm, mterm=mterm)
        r_term_dict = {'u':self.R * np.ones(self.env.Nu)}
        mpc.set_rterm(**r_term_dict)
        # Constraints
        mpc.bounds['lower', '_u', 'u'] = self.env_params["a_space"]["low"]
        mpc.bounds['upper', '_u', 'u'] = self.env_params["a_space"]["high"]

        # User-defined constraints
        if self.env_params.get("constraints") is not None:
            for k in self.env_params["constraints"]:
                state_index = self.model_info["states"].index(k)
                for j, constraint_value in enumerate(self.env_params["constraints"][k]):
                    if self.env_params["cons_type"][k][j] == "<=":
                        mpc.bounds['upper', '_x', 'x', state_index] = constraint_value
                    elif self.env_params["cons_type"][k][j] == ">=":
                            mpc.bounds['lower', '_x', 'x', state_index] = constraint_value


        p_template = mpc.get_p_template(1)  # We use 1 here as we don't have multiple scenarios
        simulator = do_mpc.simulator.Simulator(model)
        simulator.set_param(t_step=self.env.dt)
        p_template_sim = simulator.get_p_template()
        # Define parameter function
        def p_fun_mpc(t_now):
            SP_values = np.array([self.env_params["SP"][k][max(0, min(int(t_now/self.env.dt) - 1, len(self.env_params["SP"][k])-1))] for k in self.env_params["SP"]])
            p_template['_p', 0, 'SP'] = SP_values.reshape(-1, 1)
            return p_template
        def p_fun_sim(t_now):
            SP_values = np.array([self.env_params["SP"][k][max(0, min(int(t_now/self.env.dt) - 1, len(self.env_params["SP"][k])-1))] for k in self.env_params["SP"]])
            p_template_sim['SP'] = SP_values.reshape(-1, 1)
            return p_template_sim
        # Set parameter function for both MPC and simulator
        mpc.set_p_fun(p_fun_mpc)
        simulator.set_p_fun(p_fun_sim)

        simulator.setup()

        mpc.set_param(nlpsol_opts={'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'})
        mpc.setup()


        return mpc, simulator


    def mpc(self):
        mpc, simulator = self.setup_mpc()

        x0 = np.array(self.x0[:self.env.Nx_oracle])
        mpc.x0 = x0
        simulator.x0 = x0

        mpc.set_initial_guess()

        u_log = np.zeros((self.env.Nu, self.env.N))
        x_log = np.zeros((self.env.Nx_oracle, self.env.N))

        for i in range(self.env.N):
            u0 = mpc.make_step(x0)
            y_next = simulator.make_step(u0)
            x0 = y_next

            u_log[:, i] = u0.flatten()
            x_log[:, i] = x0.flatten()

        return x_log, u_log