import numpy as np
import do_mpc
from casadi import vertcat, sum1, reshape
import typing
from gymnasium import Env
class oracle:
    def __init__(self, env:Env, env_params:dict, MPC_params:bool=False) -> None:
        self.env_params = env_params
        self.env_params["integration_method"] = "casadi"
        self.env = env(env_params)

        self.x0 = env_params["x0"]
        self.T = self.env.tsim
        if not MPC_params:
            self.N = 5
            self.R = 0
        else:
            self.N = MPC_params["N"]
            self.R = MPC_params["R"]
        self.model_info = self.env.model.info()
        
        self.integral_error = np.zeros(len(self.env_params["SP"]))
        pass

    def setup_mpc(self) -> tuple[do_mpc.controller.MPC, do_mpc.simulator.Simulator]:
        model_type = 'continuous'
        model = do_mpc.model.Model(model_type)

        # States
        x = model.set_variable(var_type='_x', var_name='x', shape=(self.env.Nx_oracle, 1))

        # Inputs
        u = model.set_variable(var_type='_u', var_name='u', shape=(self.env.Nu, 1))

        # Set point (as a parameter)
        SP = model.set_variable(var_type='_p', var_name='SP', shape=(len(self.env_params["SP"]), 1))
        # System dynamics
        dx_list = self.env.model(x, u)
        try:
            dx = vertcat(*dx_list)  # Convert list to CasADi symbolic expression
        except Exception: 
            dx_list_reshaped = [reshape(dx_i, 1, 1) for dx_i in dx_list]
            dx = vertcat(*dx_list_reshaped)
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
        mpc.n_combinations = 1
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

        
        # p_template = mpc.get_p_template(1)  # We use 1 here as we don't have multiple scenarios
        simulator = do_mpc.simulator.Simulator(model)
        simulator.set_param(t_step=self.env.dt)
        p_template_sim = simulator.get_p_template()
        # Define parameter function
        def p_fun_mpc(t_now):
            p_template = mpc.get_p_template(1)
            SP_values = []
            for k in self.env_params["SP"]:
                sp_array = self.env_params["SP"][k]
                current_index = min(int(t_now/self.env.dt-1), len(sp_array) - 1)
                SP_values.append(sp_array[current_index])
            
            p_template['_p', 0, 'SP'] = np.array(SP_values).reshape(-1, 1)
            
            return p_template

        def p_fun_sim(t_now):
            SP_values = []
            for k in self.env_params["SP"]:
                sp_array = self.env_params["SP"][k]
                current_index = min(int(t_now/self.env.dt), len(sp_array) - 1)
                SP_values.append(sp_array[current_index])
            
            SP_values = np.array(SP_values).reshape(-1, 1)  # Reshape to column vector
            
            # Create a new p_template for each call
            p_template_sim = simulator.get_p_template()
            
            # Assign the SP_values to the p_template_sim
            p_template_sim['SP'] = SP_values
            
            return p_template_sim
        # Set parameter function for both MPC and simulator
        mpc.set_p_fun(p_fun_mpc)
        simulator.set_p_fun(p_fun_sim)

        simulator.setup()
        # mpc.set_param(nlpsol_opts={
        # })
        mpc.set_param(nlpsol_opts={'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'})
        # mpc.set_param(nlpsol_opts={'ipopt.nlp_scaling_method': 'gradient-based'})
        mpc.setup()

        # Set the initial guess
        mpc.set_initial_guess()

        return mpc, simulator


    def mpc(self) -> tuple[np.array, np.array]:
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