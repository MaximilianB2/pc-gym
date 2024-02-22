import numpy as np
from casadi import *
import numpy as np
import gymnasium as gym
from gymnasium import  spaces
import torch
import matplotlib.pyplot as plt
from model_classes import *
from Policy_Evaluation import policy_eval
from Integrator import integration_engine
import copy
    
class Models_env(gym.Env):
    '''
    Class for Reactor RL-Gym Environment
    '''
    def __init__(self,env_params):
        '''
        Constructor for the class
        '''
        
        self.env_params = copy.deepcopy(env_params)
        
        # Define action and observation space
        if env_params['normalise_a'] is True:
            self.action_space = spaces.Box(low = np.array([-1]*env_params['a_space']['low'].shape[0]), high = np.array([1]*env_params['a_space']['high'].shape[0]))
        else:
            self.action_space = spaces.Box(low=env_params['a_space']['low'],high = env_params['a_space']['high'])
        
        self.observation_space = spaces.Box(low = env_params['o_space']['low'],high = env_params['o_space']['high'])  
        
        self.Nx = env_params['Nx']
        self.Nu = env_params['Nu']
        self.SP = env_params['SP']
        
        self.N = env_params['N']
        self.tsim = env_params['tsim']
        self.x0 = env_params['x0']
        self.r_scale = env_params['r_scale']
        self.normalise_a = env_params['normalise_a']
        self.normalise_o = env_params['normalise_o']
        self.integration_method = env_params['integration_method']
        self.dt =  self.tsim/self.N
        self.done = False

        

        # Constraints
        self.constraint_active = False
        self.r_penalty = False
        self.info = {}

        self.custom_constraint_active = False  # Initialize to False by default
        
        if env_params.get('constraints') is not None:
            self.constraints = env_params['constraints']
            self.done_on_constraint = env_params['done_on_cons_vio']
            self.r_penalty = env_params['r_penalty']
            self.cons_type = env_params['cons_type']
            self.constraint_active = True
            self.info['cons_info'] = np.zeros((len(self.constraints),self.N,1))

        if env_params.get('custom_con') is not None:
            self.done_on_constraint = env_params['done_on_cons_vio']
            self.r_penalty = env_params['r_penalty']
            self.custom_constraint_active = True

    

        #Select model 
        model_mapping = {
        'cstr_ode': cstr_ode,
        'first_order_system_ode': first_order_system_ode,
        # 'second_order_system_ode': second_order_system_ode,
        # 'large_scale_ode': large_scale_ode,
        # 'cstr_series_recycle_ode': cstr_series_recycle_ode,
        # 'cstr_series_recycle_two_ode': cstr_series_recycle_two_ode,
        # 'distillation_ode': distillation_ode,
         'multistage_extraction_ode': multistage_extraction_ode,
        # 'multistage_extraction_reactive_ode': multistage_extraction_reactive_ode,
        # 'heat_ex_ode': heat_ex_ode,
        # 'biofilm_reactor_ode': biofilm_reactor_ode,
        # 'polymerisation_ode': polymerisation_ode,
        # 'four_tank_ode': four_tank_ode,
        # 'cstr_ode_jax': cstr_ode_jax,
        }   

        # Load custom model if it is provide else load the selected standard model.
        if self.env_params.get('custom_model') is not None:
            m = self.env_params.get('custom_model')
        else:
            m = model_mapping.get(env_params['model'], None)
        self.model = m(int_method=self.integration_method) # Initialise the model with the selected integration method


        # Handle the case where the model is not found (do this for all)
        if self.model is None:
            raise ValueError(f"Model '{env_params['model']}' not found in model_mapping.")
    
        #Disturbances
        self.disturbance_active = False
        if env_params.get('disturbances') is not None:
            self.disturbance_active = True
            self.disturbances = env_params['disturbances']
            self.Nu += len(self.model.info()['disturbances'])
        
        
       
    
    def reset(self, seed=None):
        """
        Resets the state of the system and the noise generator

        Returns the state of the system
        """
        self.t = 0
        self.int_eng = integration_engine(Models_env,self.env_params)
        
        state = copy.deepcopy(self.env_params['x0'])
        r_init = self.reward_fn(state,False)
        
        self.done = False
        self.state = state
        if self.normalise_o is True:
            self.normstate = 2 * (self.state - self.observation_space.low) / (self.observation_space.high - self.observation_space.low) - 1
            return self.normstate, {'r_init':r_init}
        else:
            return self.state,{'r_init':r_init}
    
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
        if self.normalise_a is True:
            action = (action + 1)*(self.env_params['a_space']['high'] - self.env_params['a_space']['low'])/2 + self.env_params['a_space']['low']
        
        # Add disturbance to control vector
        if self.disturbance_active:
            uk[:self.Nu-len(self.model.info()['disturbances'])] = action # Add action to control vector
            for i, k in enumerate(self.model.info()['disturbances'], start=0):
                if k in self.disturbances:
                    uk[self.Nu-len(self.model.info()['disturbances'])+i] = self.disturbances[k][self.t] # Add disturbance to control vector
                else:
                    uk[self.Nu-len(self.model.info()['disturbances'])+i] = self.model.info()['parameters'][str(k)] # if there is no disturbance at this timestep, use the default value
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
        if self.constraint_active or self.custom_constraint_active:
            constraint_violated = self.constraint_check(self.state,uk)
        
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
        if self.env_params['noise'] is True:
            noise_percentage = self.env_params['noise_percentage']
            self.state[:self.Nx] += np.random.normal(0,1,self.Nx) * self.state[:self.Nx] * noise_percentage


        if self.normalise_o is True:
            self.normstate = 2 * (self.state - self.observation_space.low) / (self.observation_space.high - self.observation_space.low) - 1
            return self.normstate, rew, self.done, False, self.info
        else:
            return self.state, rew, self.done, False, self.info
    
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

        for k in self.SP:
            i = self.model.info()['states'].index(k)
            r +=  (-((state[i] - np.array(self.SP[k][self.t]))**2))*self.r_scale[k]
            if self.r_penalty and c_violated:
                r -= 1000
        return r
    
    def con_checker(self, items, item_values):
        """
        Check constraints for given items and their values.

        Parameters:
        items (list): A list of items (states or inputs).
        item_values (list): A list of corresponding item values.

        Returns:
        bool: True if any constraint is violated, False otherwise.
        """
        for i, item in enumerate(items):
            if item in self.constraints:
                constraint = self.constraints[item]
                cons_type = self.cons_type[item]
                item_value = item_values[i]

                is_greater_violated = cons_type == '>=' and item_value <= constraint
                is_less_violated = cons_type == '<=' and item_value >= constraint

                if is_greater_violated or is_less_violated:
                    self.info['cons_info'][self.con_i, self.t, :] = abs(item_value - constraint)
                    return True
                self.con_i += 1 

        return False
    

    def constraint_check(self,state,input):
        """
        Check if constraints are violated and update info array accordingly.

        Inputs: state - current state of the system

        Outputs: constraint_violated - boolean indicating if constraint is violated
        """
        self.con_i = 0
        constraint_violated = False
        states = self.model.info()['states']
        inputs = self.model.info()['inputs']
        if self.env_params.get('custom_con') is not None:
            custom_con_vio_f = self.env_params['custom_con']
            custom_con_vio = custom_con_vio_f(state,input) # User defined constraint return True if violated
            assert isinstance(custom_con_vio, bool), "Custom constraint must return a boolean (True == Violated)"
        else:
            custom_con_vio = False

        if self.constraint_active and self.custom_constraint_active:
            constraint_violated = self.con_checker(states, state) or self.con_checker(inputs, input) or False # Check both inputs and states
        elif self.constraint_active:
            constraint_violated = self.con_checker(states, state) or self.con_checker(inputs, input) or False # Check both inputs and states
        elif self.custom_constraint_active:
            constraint_violated = custom_con_vio

        self.done = self.done_on_constraint 
        return constraint_violated
            
              

   

    def plot_rollout(self,policy,reps,oracle = False,dist_reward = False,MPC_params = False):
        '''
        Plot the rollout of the given policy.

        Parameters:
        - policy: The policy to evaluate.
        - reps: The number of rollouts to perform.
        - oracle: Whether to use an oracle model for evaluation. Default is False.
        - dist_reward: Whether to use a distance-based reward. Default is False.
        - MPC_params: Whether to use MPC parameters. Default is False.
        '''
        policy_eval(Models_env,policy,reps,self.env_params,oracle,MPC_params).plot_rollout(dist_reward)
        

    





