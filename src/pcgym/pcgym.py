import numpy as np
from casadi import *
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .model_classes import *
from .Policy_Evaluation import policy_eval
from .Integrator import integration_engine
import copy 
    
class make_env(gym.Env):
    '''
    Class for RL-Gym Environment
    '''
    def __init__(self,env_params):
        '''
        Constructor for the class
        '''
        
        self.env_params = copy.deepcopy(env_params)
        try:
            self.normalise_a = env_params['normalise_a']
            self.normalise_o = env_params['normalise_o']
        except:
            self.normalise_a = True
            self.normalise_o = True
        
        # Define action and observation space
        if self.normalise_a is True:
            self.action_space = spaces.Box(low = np.array([-1]*env_params['a_space']['low'].shape[0]), high = np.array([1]*env_params['a_space']['high'].shape[0]))
        else:
            self.action_space = spaces.Box(low=env_params['a_space']['low'],high = env_params['a_space']['high'])
            
        self.SP = env_params['SP']
        self.N = env_params['N']
        self.tsim = env_params['tsim']
        self.x0 = env_params['x0']
        # Initial setup for observation space based on user-defined bounds
        base_obs_low = env_params['o_space']['low']
        base_obs_high = env_params['o_space']['high']
        self.observation_space = spaces.Box(low=base_obs_low, high=base_obs_high)

        try :
            self.integration_method = env_params['integration_method']
        except:
            self.integration_method = 'casadi'

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
            self.n_con = 0
            for _, con_list in self.constraints.items():
                self.n_con += len(con_list)
            self.info['cons_info'] = np.zeros((self.n_con,self.N,1))

        if env_params.get('custom_con') is not None:
            self.done_on_constraint = env_params['done_on_cons_vio']
            self.r_penalty = env_params['r_penalty']
            self.custom_constraint_active = True

    

        #Select model 
        model_mapping = {
        'cstr_ode': cstr_ode,
        'first_order_system_ode': first_order_system_ode,
        'bang_bang_control_ode': bang_bang_control_ode,
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
        
        # Import states and controls from model info
        self.Nx = len(self.model.info()['states'])
        self.Nu = len(self.model.info()['inputs'])

        # Disturbances
        self.disturbance_active = False
        self.Nd = 0
        if env_params.get('disturbances') is not None:
            self.disturbance_active = True
            self.disturbances = env_params['disturbances']
            self.Nd = len(self.model.info()['disturbances'])
            self.Nu += self.Nd
            # Extend the state size by the number of disturbances
            self.Nx += self.Nd
            # user has defined disturbance_bounds within env_params
            disturbance_low = env_params['disturbance_bounds']['low']
            disturbance_high = env_params['disturbance_bounds']['high']
            assert disturbance_low.shape[0] == self.Nd, "Mismatch in disturbance low bounds dimension"
            assert disturbance_high.shape[0] == self.Nd, "Mismatch in disturbance high bounds dimension"
            # Extend the observation space bounds to include disturbances
            extended_obs_low = np.concatenate((base_obs_low, disturbance_low))
            extended_obs_high = np.concatenate((base_obs_high, disturbance_high))
            # Define the extended observation space
            self.observation_space = spaces.Box(low=extended_obs_low, high=extended_obs_high, dtype=np.float32)
        
        
        
    def reset(self, seed=None, **kwargs):  # Accept arbitrary keyword arguments
        """
        Resets the state of the system 

        Returns the state of the system
        """
        self.t = 0
        self.int_eng = integration_engine(make_env,self.env_params)
        
        state = copy.deepcopy(self.env_params['x0'])
        
        # If disturbances are active, expand the initial state with disturbances
        if self.disturbance_active:
            initial_disturbances = [] 
            for k in self.model.info()['disturbances']:
                if k in self.disturbances:
                    initial_disturbances.append(self.disturbances[k][0])
                else:
                    initial_disturbances.append(self.model.info()['parameters'][str(k)])
            # Append initial disturbances to the state
            state = np.concatenate((state, initial_disturbances))

        self.state = state

        r_init = self.reward_fn(state,False)

        self.done = False

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
            disturbance_values = []
            for i, k in enumerate(self.model.info()['disturbances'], start=0):
                disturbance_index = self.Nx + i  # Index in state vector for this disturbance     
                if k in self.disturbances:
                    current_disturbance_value = self.disturbances[k][self.t]
                    uk[self.Nu-self.Nd+i] = self.disturbances[k][self.t] # Add disturbance to control vector
                    disturbance_values.append(current_disturbance_value)
                else:
                    default_value = self.model.info()['parameters'][str(k)]
                    uk[self.Nu-self.Nd+i] = self.model.info()['parameters'][str(k)] # if there is no disturbance at this timestep, use the default value
                    disturbance_values.append(default_value)
            # Update the state vector with current disturbance values
            self.state[self.Nx:(self.Nx + self.Nd)] = disturbance_values
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
        if self.env_params.get('noise', False):
            noise_percentage = self.env_params.get('noise_percentage', 0)
            self.state[:self.Nx] += np.random.normal(0, 1, self.Nx) * self.state[:self.Nx] * noise_percentage


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
            r_scale = self.env_params.get('r_scale', {})
            r +=  (-((state[i] - np.array(self.SP[k][self.t]))**2))*r_scale.get(k, 1)
            if self.r_penalty and c_violated:
                r -= 1000
        return r
    
    def con_checker(self, model_states, curr_state):
        """
        Check constraints for given model_states and their values.

        Parameters:
        model_states (list): A list of model_states (states or inputs).
        curr_state (list): A list of corresponding item values.

        Returns:
        bool: True if any constraint is violated, False otherwise.
        """
        for i, state in enumerate(model_states):
            if state in self.constraints:
                constraint = self.constraints[state] # List of constraints
                cons_type = self.cons_type[state] # List of cons type
                for j in range(len(constraint)):
                    curr_state_i = curr_state[i]
                    is_greater_violated = cons_type[j] == '>=' and curr_state_i <= constraint[j]
                    is_less_violated = cons_type[j] == '<=' and curr_state_i >= constraint[j]

                    if is_greater_violated or is_less_violated:
                        self.info['cons_info'][self.con_i, self.t, :] = abs(curr_state_i - constraint[j])
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
            
              
    def get_rollouts(self, policies, reps, oracle = False, dist_reward = False, MPC_params = False, cons_viol = False):
        '''
        Plot the rollout of the given policy.

        Parameters:
        - policies: dictionary of policies to evaluate
        - reps: The number of rollouts to perform.
        - oracle: Whether to use an oracle model for evaluation. Default is False.
        - dist_reward: Whether to use reward distribution for plotting. Default is False.
        - MPC_params: Whether to use MPC parameters. Default is False.
        '''
        # construct evaluator
        evaluator = policy_eval(make_env, policies, reps, self.env_params, oracle, MPC_params)
        # generate rollouts 
        data = evaluator.get_rollouts()
        # return evaluator and data
        return evaluator, data


    def plot_rollout(self, policies, reps, oracle = False, dist_reward = False, MPC_params = False, cons_viol = False):
        '''
        Plot the rollout of the given policy.

        Parameters:
        - policies: dictionary of policies to evaluate
        - reps: The number of rollouts to perform.
        - oracle: Whether to use an oracle model for evaluation. Default is False.
        - dist_reward: Whether to use reward distribution for plotting. Default is False.
        - MPC_params: Whether to use MPC parameters. Default is False.
        '''
        # construct evaluator
        evaluator = policy_eval(make_env, policies, reps, self.env_params, oracle, MPC_params, cons_viol)
        # generate rollouts 
        data = evaluator.get_rollouts()
        # plot data from rollouts via the evaluator method
        evaluator.plot_data(data, dist_reward)
        # return constructed evaluator and data
        return evaluator, data
        

    





