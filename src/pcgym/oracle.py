import numpy as np
import do_mpc
from casadi import vertcat, sum1, reshape, DM, mtimes

class oracle:
    def __init__(self, env, env_params, MPC_params=False):
        self.env_params = env_params
        self.env_params["integration_method"] = "casadi"
        try:
            self.env = env(env_params)
        except Exception:
            self.env = env
        self.use_delta_u = False
        self.x0 = env_params["x0"]
        self.T = self.env.tsim
        if not MPC_params:
            self.N = 5
            self.R = np.zeros((self.env.Nu - self.env.Nd_model, self.env.Nu - self.env.Nd_model))
            self.Q = np.eye(self.env.Nx_oracle)
        else:
            self.N = MPC_params.get("N", 5)
            self.R = MPC_params.get("R", np.zeros((self.env.Nu - self.env.Nd_model, self.env.Nu - self.env.Nd_model)))
            self.Q = MPC_params.get("Q", np.eye(self.env.Nx_oracle))
        self.model_info = self.env.model.info()
        self.R_sym = DM(self.R)
        if env_params.get('a_delta') is not None:
            self.u_0 = env_params.get("a_0", 0)  # Initialize u_0
            self.use_delta_u = True
        else:
            self.u_0 = None  # Initialize u_0 as None when not using delta_u
        self.has_disturbances = self.env_params.get("disturbances") is not None

    def setup_mpc(self):
        model_type = 'continuous'
        model = do_mpc.model.Model(model_type)

        # States
        x = model.set_variable(var_type='_x', var_name='x', shape=(self.env.Nx_oracle, 1))

        # Input
        if self.use_delta_u:
            u_prev = model.set_variable(var_type='_p', var_name='u_prev', shape=(self.env.Nu, 1))
            delta_u = model.set_variable(var_type='_u', var_name='delta_u', shape=(self.env.Nu, 1))
            u = u_prev + delta_u
        else:
            u = model.set_variable(var_type='_u', var_name='u', shape=(self.env.Nu - self.env.Nd_model, 1))

        if self.has_disturbances:
            d = model.set_variable(var_type='_p', var_name='d', shape=(self.env.Nd_model, 1))
            u_full = vertcat(u, d)
        else:
            u = model.set_variable(var_type='_u', var_name='u', shape=(self.env.Nu, 1))
            u_full = u

        # Set point (as a parameter)
        SP = model.set_variable(var_type='_p', var_name='SP', shape=(len(self.env.SP), 1))

        # System dynamics
        dx_list = self.env.model(x, u_full)

        try:
            dx = vertcat(*dx_list)
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
        lterm = 0
        for i, sp_key in enumerate(self.env_params["SP"]):
            state_index = self.model_info["states"].index(sp_key)
            lterm += self.Q[state_index, state_index] * (x[state_index] - SP[i])**2

        if self.use_delta_u:
            lterm += delta_u.T @ self.R_sym @ delta_u
        else:
            lterm += u.T @ self.R_sym @ u

        mterm = 0
        for i, sp_key in enumerate(self.env_params["SP"]):
            state_index = self.model_info["states"].index(sp_key)
            mterm += self.Q[state_index, state_index] * (x[state_index] - SP[i])**2

        mpc.set_objective(lterm=lterm, mterm=mterm)
        
        # Set r_term for controlled inputs
        r_term = np.diag(self.R)
        if self.use_delta_u:
            r_term_dict = {'delta_u': r_term}
        else:
            r_term_dict = {'u': r_term}
        mpc.set_rterm(**r_term_dict)


        # Constraints
        if self.use_delta_u:
            mpc.bounds['lower', '_u', 'delta_u'] = np.concatenate([self.env_params["a_space"]["low"]])
            mpc.bounds['upper', '_u', 'delta_u'] = np.concatenate([self.env_params["a_space"]["high"]])

            # Add constraint on u (u_prev + delta_u)
            u = model.p['u_prev'] + model.u['delta_u']

            # Lower bound constraint
            mpc.set_nl_cons('u_lower', u[:self.env.Nu - self.env.Nd_model] - self.env_params["a_space_act"]["low"], soft_constraint=True, penalty_term_cons=1e3)
            
            # Upper bound constraint
            mpc.set_nl_cons('u_upper', self.env_params["a_space_act"]["high"] - u[:self.env.Nu - self.env.Nd_model], soft_constraint=True, penalty_term_cons=1e3)
        else:
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

        # Parameter function
        def p_fun(t_now):
            p_template = mpc.get_p_template(1)
            
            SP_values = []
            for k in self.env_params["SP"]:
                sp_array = self.env_params["SP"][k]
                current_index = min(int(t_now/self.env.dt-1), len(sp_array) - 1)
                SP_values.append(sp_array[current_index])

            p_template['_p', 0, 'SP'] = np.array(SP_values).reshape(-1, 1)
            # Set u_prev only if delta_u is used
            if self.use_delta_u:
                p_template['_p', 0, 'u_prev'] = mpc.u0

            if self.has_disturbances:
                d = np.zeros((self.env.Nd_model, 1))
                for i, k in enumerate(self.env.model.info()["disturbances"], start=0):
                    if k in self.env_params["disturbances"]:
                        d_array = self.env_params["disturbances"][k]
                        current_index = min(int(t_now/self.env.dt-1), len(d_array) - 1)
                        d[i,0] = d_array[current_index]
                    else:
                        d[i,0] = self.model_info["parameters"][str(k)]
                p_template['_p', 0, 'd'] = d

            return p_template

        mpc.set_p_fun(p_fun)

        # Setup simulator
        simulator = do_mpc.simulator.Simulator(model)
        simulator.set_param(t_step=self.env.dt)

        def p_fun_sim(t_now):
            p_template_sim = simulator.get_p_template()
            
            SP_values = []
            for k in self.env_params["SP"]:
                sp_array = self.env_params["SP"][k]
                current_index = min(int(t_now/self.env.dt), len(sp_array) - 1)
                SP_values.append(sp_array[current_index])
            p_template_sim['SP'] = np.array(SP_values).reshape(-1, 1)
            
            if self.has_disturbances:
                d = np.zeros((self.env.Nd_model,1))
                for i, k in enumerate(self.env.model.info()["disturbances"], start=0):
                    if k in self.env_params["disturbances"]:
                        d_array = self.env_params["disturbances"][k]
                        current_index = min(int(t_now/self.env.dt-1), len(d_array) - 1)
                        d[i,0] = d_array[current_index]
                    else:
                        d[i,0] = self.model_info["parameters"][str(k)]
                p_template_sim['d'] = d
            return p_template_sim

        simulator.set_p_fun(p_fun_sim)

        simulator.setup()
        mpc.set_param(nlpsol_opts={'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'})
        mpc.setup()

        return mpc, simulator
    
    def mpc(self):
        mpc, simulator = self.setup_mpc()

        x0 = np.array(self.x0[:self.env.Nx_oracle])

        # Initialize u_prev only if delta_u is used
        if self.use_delta_u:
            u_prev = np.full((self.env.Nu, 1), self.u_0)  # Use the initial input from init
        mpc.x0 = x0
        simulator.x0 = x0
        mpc.set_initial_guess()

        # Compute correct size dynamically
        num_u_rows = self.env.Nu + (self.env.Nd_model if self.has_disturbances else 0)

        # ✅ Fix: Allocate enough space for u_log
        u_log = np.zeros((num_u_rows, self.env.N))


        x_log = np.zeros((self.env.Nx_oracle, self.env.N))
        delta_u_log = np.zeros((self.env.Nu, self.env.N)) if self.use_delta_u else None
        
        # Store initial state as first entry
        x_log[:, 0] = x0.flatten()
        
        # Calculate first control input
        if self.use_delta_u:
            mpc.u0 = u_prev
            simulator.u0 = u_prev
            delta_u0 = mpc.make_step(x0)
            u0 = u_prev + delta_u0
            delta_u_log[:, 0] = delta_u0.flatten()
        else:
            u0 = mpc.make_step(x0)

        if self.has_disturbances:
            d = mpc.p_fun(0 * self.env.dt)['_p', 0, 'd']
            u_full = np.vstack([u0, d])
            u_log[:, 0] = u_full.flatten()
        else:
            u_log[:, 0] = u0.flatten()
        
        # Get next state
        y_next = simulator.make_step(u0)
        x0 = y_next
        
        # Start loop from 1 since we already handled first step
        for i in range(1, self.env.N):
            # Update u_prev parameter if delta_u is used
            if self.use_delta_u:
                mpc.u0 = u_prev
                simulator.u0 = u_prev

            if self.use_delta_u:
                delta_u0 = mpc.make_step(x0)
                u0 = u_prev + delta_u0
            else:
                u0 = mpc.make_step(x0)

            y_next = simulator.make_step(u0)
            x0 = y_next

            if self.has_disturbances:
                d = mpc.p_fun(i * self.env.dt)['_p', 0, 'd']
                u_full = np.vstack([u0, d])  # ✅ Fix: Stack u0 and disturbances correctly

                # Debug print to verify shape consistency
                print(f"DEBUG: i={i}, u_full shape: {u_full.shape}, u_log shape: {u_log.shape}")

                print("Shape of u0:", u0.shape)  # Expecting (3,1)
                print("Shape of d:", d.shape)  # Expecting (2,1)
                print("Shape of u_full before assignment:", u_full.shape)  # Should be (5,1)
                print("Shape of u_log[:, i] before assignment:", u_log[:, i].shape)  # Should be (5,)

                # ✅ Fix: Assign correctly to u_log
                u_log[:, i] = u_full.flatten()
            else:
                u_log[:self.env.Nu, i] = u0.flatten()  # ✅ Fix: Ensure correct slicing

            if self.use_delta_u:
                delta_u_log[:, i] = delta_u0.flatten()
                u_prev = u0  # Update u_prev for the next iteration
            x_log[:, i] = x0.flatten()

        return x_log, u_log
