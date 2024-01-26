import numpy as np
from casadi import *
import numpy as np
import gymnasium as gym
from gymnasium import  spaces
import torch
import matplotlib.pyplot as plt
from case_studies import *
from Policy_Evaluation import policy_eval
from Integrator import integration_engine
    
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
        if env_params['normalise_a'] == True:
            self.action_space = spaces.Box(low = np.array([-1]*env_params['a_space']['low'].shape[0]), high = np.array([1]*env_params['a_space']['high'].shape[0]))
        else:
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
        self.normalise_a = env_params['normalise_a']
        self.normalise_o = env_params['normalise_o']
        self.integration_method = env_params['integration_method']
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
            self.info['cons_info'] = np.zeros((len(self.constraints),self.N,1))

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
        'four_tank_ode': four_tank_ode,
        'cstr_ode_jax': cstr_ode_jax,
        }   

        self.model = model_mapping.get(env_params['model'], None)
        
        # Handle the case where the model is not found (do this for all)
        if self.model is None:
            raise ValueError(f"Model '{env_params['model']}' not found in model_mapping.")
        
        
       
    
    def reset(self, seed=None):
        """
        Resets the state of the system and the noise generator

        Returns the state of the system
        """
        self.int_eng = integration_engine(Models_env,self.env_params)
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
        if self.normalise_a == True:
            action = (action + 1)*(self.env_params['a_space']['high'] - self.env_params['a_space']['low'])/2 + self.env_params['a_space']['low']
        
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
        if self.integration_method == 'casadi':
            Fk = self.int_eng.casadi_step(self.state,uk)
            self.state[:self.Nx] = np.array(Fk['xf'].full()).reshape(self.Nx)
        elif self.integration_method == 'jax':
            self.state[:self.Nx] = self.int_eng.jax_step(self.state,uk)

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
        if self.env_params['noise'] == True:
            noise_percentage = self.env_params['noise_percentage']
            self.state[:self.Nx] += np.random.normal(0,1,self.Nx) * self.state[:self.Nx] * noise_percentage


        if self.normalise_o == True:
            self.normstate = 2 * (self.state - self.observation_space.low) / (self.observation_space.high - self.observation_space.low) - 1
        return self.normstate, rew, self.done, False, self.info
    
    def reward_fn(self, state,c_violated):
        """
        Compute reward for one timestep and penalise constraint violation if requested by the user.

        Inputs:
            state - current state of the system
            c_violated - boolean indicating if constraint is violated
        Outputs:
            r - reward for current timestep

        """

        r = 0.

        for i in range(self.Nx):
            if str(i) in self.SP:
                r +=  (-((state[i] - np.array(self.SP[str(i)][self.t]))**2))*self.r_scale[str(i)]
                if self.r_penalty and c_violated:
                    r -= 1000
        return r
    
    def constraint_check(self,state):
        """
        Check if constraints are violated and update info array accordingly.

        Inputs: state - current state of the system

        Outputs: constraint_violated - boolean indicating if constraint is violated
        """

        constraint_violated = False
        for c_i in self.constraints:
            if self.constraints[c_i] is not None:
                if ((self.cons_type[c_i] == '>=' and state[int(c_i)] <= self.constraints[c_i]) or
                    (self.cons_type[c_i] == '<=' and state[int(c_i)] >= self.constraints[c_i])):
                    self.info['cons_info'][int(c_i), self.t, :] = abs(state[int(c_i)] - self.constraints[c_i])
                    constraint_violated = True
                    self.done = self.done_on_constraint
                        
      
        return constraint_violated
            
              

   

    def plot_rollout(self,policy,reps):
        policy_eval(Models_env,policy,reps,self.env_params).plot_rollout()
        

    





