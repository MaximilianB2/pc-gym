import numpy as np
from casadi import *
import numpy as np
import gymnasium as gym
from gymnasium import  spaces
import torch
import matplotlib.pyplot as plt
from case_studies import *
    
class Models_env(gym.Env):
    '''
    Class for Reactor RL-Gym Environment
    '''
    def __init__(self,env_params):
        '''
        Constructor for the class
        '''
        
        self.env_params = env_params
        # Define action and observation space
        self.action_space = spaces.Box(low=env_params['a_space']['low'],high = env_params['a_space']['high'])
        self.observation_space = spaces.Box(low = env_params['o_space']['low'],high = env_params['o_space']['high'])  
        
        self.Nx = env_params['Nx']
        self.Nu = env_params['Nu']
        self.SP = env_params['SP']
        self.dt = env_params['dt']
        self.N = env_params['N']
        self.tsim = env_params['tsim']
        self.x0 = env_params['x0']
        self.r_scale = env_params['r_scale']
        self.done = False


        #Constraints
        self.constraint_active = False
        self.r_penalty = False
        self.info = {}
        if env_params.get('constraints') is not None:
            self.constraints = env_params['constraints']
            self.done_on_constraint = env_params['done_on_cons_vio']
            self.r_penalty = env_params['r_penalty']
            self.cons_type = env_params['cons_type']
            self.constraint_active = True
            self.info = np.zeros((len(self.constraints),self.N,1))

        #Disturbances
        self.disturbance_active = False
        if env_params.get('disturbances') is not None:
            self.disturbance_active = True
            self.disturbances = env_params['disturbances']
            self.Nu += len(self.disturbances)

        #Select model 
        model_mapping = {
        'cstr_ode': cstr_ode,
        'first_order_system_ode': first_order_system_ode,
        'second_order_system_ode': second_order_system_ode,
        'large_scale_ode': large_scale_ode,
        'cstr_series_recycle_ode': cstr_series_recycle_ode,
        'cstr_series_recycle_two_ode': cstr_series_recycle_two_ode,
        'distillation_ode': distillation_ode,
        'multistage_extraction_ode': multistage_extraction_ode,
        'multistage_extraction_reactive_ode': multistage_extraction_reactive_ode,
        'heat_ex_ode': heat_ex_ode,
        'biofilm_reactor_ode': biofilm_reactor_ode,
        'polymerisation_ode': polymerisation_ode,
        'four_tank_ode': four_tank_ode
        }   

        self.model = model_mapping.get(env_params['model'], None)
        
        # Handle the case where the model is not found (do this for all)
        if self.model is None:
            raise ValueError(f"Model '{env_params['model']}' not found in model_mapping.")
        
        # Generate casadi model
        self.sym_x = self.gen_casadi_variable(self.Nx, "x")
        self.sym_u = self.gen_casadi_variable(self.Nu, "u")    
        self.casadi_sym_model = self.casadify(self.model, self.sym_x, self.sym_u)
        self.casadi_model_func = self.gen_casadi_function([self.sym_x, self.sym_u],[self.casadi_sym_model],
                                                          "model_func", ["x","u"], ["model_rhs"])
        self.reset()
    def reset(self, seed=None):
        """
        Resets the state of the system and the noise generator

        Returns the state of the system
        """
        
        self.state = self.x0
        self.t = 0
        self.done = False
        return self.state, {}
    
    def step(self, action):
        """
        Simulate one timestep of the environment

        Parameters
        ----------
        action : action taken by agent


        Returns
        -------
        state: array
            state of the system after timestep.
        rew : float
            reward obtained
        done : {0,1}
            0 if target not reached. 1 if reached
        info :

        """
        # Create control vector 
        uk = np.zeros(self.Nu)

      
   
        # Add disturbance to control vector
        if self.disturbance_active:
            uk[:self.Nu-len(self.disturbances)] = action
            for i, k in enumerate(self.disturbances, start=0):
              
                uk[self.Nu-len(self.disturbances)+i] = self.disturbances[k][self.t]
                if self.disturbances[k][self.t] == None:
                    uk[self.Nu-len(self.disturbances)+i] = default_values[self.env_params['model']][str(k)]
        else:
            uk = action  # Add action to control vector

        # Simulate one timestep
        plant_func = self.casadi_model_func
        discretised_plant = self.discretise_model(plant_func, self.dt)
        xk = self.state[:self.Nx]
        Fk = discretised_plant(x0=xk, p=uk)
        self.state[:self.Nx] = np.array(Fk['xf'].full()).reshape(self.Nx)

        # Check if constraints are violated
        
        constraint_violated = False
        if self.constraint_active:
            constraint_violated = self.constraint_check(self.state)
        
        # Compute reward
        rew = self.reward_fn(self.state, constraint_violated)
        
        # For each set point, if it exists, append its value at the current time step to the list
        SP_t = []
        for k in self.SP.keys():
            if k in self.SP:
                SP_t.append(self.SP[k][self.t]) 
        self.state[self.Nx:] = np.array(SP_t)   
                
       
        # Update timestep
        self.t += 1
        if self.t == self.N:
            self.done = True
      
        # add noise to state
        self.state[:self.Nx] += np.random.normal(0,0.001,self.Nx)
        return self.state, rew, self.done, False, self.info
    def reward_fn(self, state,c_violated):
        """
        Compute reward for one timestep and penalise constraint violation if requested by the user.

        """
        r = 0.

        for i in range(self.Nx):
            if str(i) in self.SP:
                r +=  (-((state[i] - np.array(self.SP[str(i)][self.t]))**2))*self.r_scale[i]
                if self.r_penalty and c_violated:
                    
                    r -= 1000
        return r
    
    def constraint_check(self,state):
        """
        Check if constraints are violated and update info array accordingly.
        """
        constraint_violated = False
        for c_i in self.constraints:
            if self.constraints[c_i] is not None:
                if ((self.cons_type[c_i] == '>=' and state[int(c_i)] <= self.constraints[c_i]) or
                    (self.cons_type[c_i] == '<=' and state[int(c_i)] >= self.constraints[c_i])):
                    self.info[int(c_i), self.t, :] = abs(state[int(c_i)] - self.constraints[c_i])
                    constraint_violated = True
                    self.done = self.done_on_constraint
                        

       
        return constraint_violated
            
              
    def casadify(self, model, sym_x, sym_u):
        """
        Given a model with Nx states and Nu inputs and returns rhs of ode,
        return casadi symbolic model (Not function!)
        
        Inputs:
            model - model to be casidified i.e. a list of ode rhs of size Nx
            
        Outputs:
            dxdt - casadi symbolic model of size Nx of rhs of ode
        """

        dxdt = model(sym_x, sym_u)
        dxdt = vertcat(*dxdt) #Return casadi list of size Nx

        return dxdt



    def gen_casadi_variable(self, n_dim, name = "x"):
        """
        Generates casadi symbolic variable given n_dim and name for variable
        
        Inputs:
            n_dim - symbolic variable dimension
            name - name for symbolic variable
            
        Outputs:
            var - symbolic version of variable
        """

        var = SX.sym(name, n_dim)

        return var

    def gen_casadi_function(self, casadi_input, casadi_output, name, input_name=[], output_name=[]):
        """
        Generates a casadi function which maps inputs (casadi symbolic inputs) to outputs (casadi symbolic outputs)
        
        Inputs:
            casadi_input - list of casadi symbolics constituting inputs
            casadi_output - list of casadi symbolic output of function
            name - name of function
            input_name - list of names for each input
            output_name - list of names for each output
        
        Outputs:
            casadi function mapping [inputs] -> [outputs]
        
        """

        function = Function(name, casadi_input, casadi_output, input_name, output_name)

        return function


    


    def plot_simulation_results(self, x, u, cons, variable_names = None):
        """
        Plots the results of System.simulate()
        
        Input:
            tsim - simulation time as numpy array
            x - numpy array of output of size len(tsim), Nx
            u - numpy array of input of size len(tsim), Nu
            variable_names = dictionary of list of variable names to be used for plotting
        
        Output: 
            Nx + Nu plots for each output and input
        """
        t = np.linspace(0,self.tsim,self.N)
        
        if variable_names == None:
            variable_names = {}
            variable_names["x"] = ["x_" + str(i) for i in range(self.Nx)]
            variable_names["u"] = ["u_" + str(i) for i in range(self.Nu)]
            
        # Plot states, setpoints and constraints
        for j in range(self.Nx):
            plt.plot(t, x[:,j],"k")
            if self.constraint_active:
                if str(j) in self.constraints:
                    plt.hlines(self.constraints[str(j)], 0,self.tsim,'r',label='Constraint')
            plt.plot(t, self.SP[str(j)],"b--",label='Set Point')
            plt.grid()
            plt.xlabel("Time")
            plt.ylabel(variable_names["x"][j])
            plt.legend()
            plt.show()
        
        # Plot control inputs
        len_d = 0
        if self.disturbance_active:
            len_d = len(self.disturbances)
        for k in range(self.Nu-len_d):
            plt.plot(t, u[:,k],"r--")
            plt.grid()
            plt.xlabel("Time")
            plt.ylabel(variable_names["u"][k])
            plt.show()

        # Plot disturbances
        if self.disturbance_active:
            variable_names["d"] = ["d_" + str(i) for i in range(len(self.disturbances))]
            for k in self.disturbances.keys():
                if self.disturbances[k].any() != None:
                    plt.plot(t, self.disturbances[k],"r")
                    plt.grid()
                    plt.xlabel("Time")
                    plt.ylabel(variable_names["d"][int(k)])
                    plt.show()

        # Plot constraint violation
        if self.constraint_active:
            for c_i in self.constraints.keys():
                plt.plot(t,cons[int(c_i)])
                plt.grid()
                plt.xlabel('Time')
                plt.ylabel(variable_names["x"][int(c_i)]+' Constraint Violation')
                plt.show()
        

    def discretise_model(self, casadi_func, delta_t):
        """
        Input:
            casadi_func to be discretised
        
        Output:
            discretised casadi func
        """
        x = SX.sym("x", self.Nx)
        u = SX.sym("u", self.Nu)
        xdot = casadi_func(x, u)

        dae = {'x':x, 'p':u, 'ode':xdot} 
        t0 = 0
        tf = delta_t
        discrete_model = integrator('discrete_model', 'cvodes', dae,t0,tf)

        return discrete_model
    


    def generate_inputs(self):
        """
        Generate num_steps steps between lower and upper bound for length of simulation
        
        Inputs:
            num_inputs - Number of input signals to generate - integer
            lb - Lower bound passed as list for each input
            ub - Upper bound passed as list for each input
            num_steps - Number of steps - integer
            tsim - Simulation time passed as numpy array with size indicating number of simulation steps
            
        Outputs:
            u - len(tsim) * num_inputs sized numpy array
        
        """
       

        lb = self.env_params['a_space']['low']
        ub = self.env_params['a_space']['high']

        num_steps = self.N
        tsim  = np.linspace(0,self.tsim,self.N)
        num_inputs = self.Nu - len(self.disturbances)
   
  
        N_sim = len(tsim)

        input_signal = np.ones((N_sim, num_inputs))

        for i in range(num_inputs):
            lower_bound = lb[i]
            upper_bound = ub[i]
            mean = (upper_bound + lower_bound)/2
            u = np.ones_like(tsim)*mean
            step_length = int(np.floor(N_sim/num_steps))
            for j in range(num_steps):
                u[j*step_length:(j+1)*step_length] = np.random.uniform(lower_bound,upper_bound)

            input_signal[:,i] = u

        return input_signal




