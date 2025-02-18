import numpy as np
import gymnasium as gym
from gymnasium import spaces
from pcgym.model_classes import (
    cstr,
    first_order_system,
    multistage_extraction,
    nonsmooth_control,
    cstr_series_recycle,
    distillation_column,
    multistage_extraction_reactive,
    four_tank,
    photo_production,
    heat_exchanger,
    biofilm_reactor,
    polymerisation_reactor,
    crystallization,
)
from pcgym.policy_evaluation import policy_eval
from pcgym.integrator import integration_engine
import copy


class make_env(gym.Env):
    def __init__(self, env_params: dict) -> None:
        """Initialize the environment with given parameters.

        Args:
            env_params (dict): Environment configuration parameters including model selection,
                                spaces, simulation parameters, constraints, and custom functions.

        """
        if not isinstance(env_params, dict):
            raise ValueError("env_params must be a dictionary")
        self.env_params = copy.deepcopy(env_params)
        self._initialize_action_config()
        self._setup_spaces()
        self._configure_reward()
        self._setup_simulation_params()
        self._setup_constraints()
        self._initialize_model()
        self._setup_state_dimensions()
        self._setup_disturbances()
        self._setup_custom_reward()
        self._setup_uncertainty()
        self._noise_percentage_setup()
        self._setup_partial_observations()

    def _initialize_action_config(self):
        self.a_delta = self.env_params.get("a_delta", False)
        if self.a_delta:
            self.a_0 = self.env_params["a_0"]
        self.normalise_a = self.env_params.get("normalise_a", True)
        self.normalise_o = self.env_params.get("normalise_o", True)

    def _noise_percentage_setup(self):
        self.noise_percentage = self.env_params.get("noise_percentage")
        if self.noise_percentage is not None:
            self.noise_percentage_float = isinstance(self.noise_percentage, float)
        
    def _setup_spaces(self):
        if self.normalise_a:
            dim = self.env_params["a_space"]["low"].shape[0]
            self.action_space = spaces.Box(
                low=np.array([-1] * dim),
                high=np.array([1] * dim)
            )
        else:
            self.action_space = spaces.Box(
                low=self.env_params["a_space"]["low"],
                high=self.env_params["a_space"]["high"]
            )
        
        base_obs_low = self.env_params["o_space"]["low"]
        base_obs_high = self.env_params["o_space"]["high"]
        self.observation_space_base = spaces.Box(low=base_obs_low, high=base_obs_high)
        
        if self.normalise_o:
            dim = base_obs_low.shape[0]
            self.observation_space = spaces.Box(
                low=np.array([-1] * dim),
                high=np.array([1] * dim, dtype=np.float32)
            )
        else:
            self.observation_space = self.observation_space_base

    def _configure_reward(self):
        self.maximise_reward = True
        self.SP = self.env_params.get("SP")
        
        if self.SP is not None and self.env_params.get("custom_reward") is None:
            self.reward = "SP_reward_fn"
        elif self.SP is None and self.env_params.get("custom_reward") is None:
            self.reward = "batch_reward_fn"
            self.reward_states = self.env_params["reward_states"]
            self.maximise_reward = self.env_params["maximise_reward"]

    def _setup_simulation_params(self):
        self.N = self.env_params["N"]
        self.tsim = self.env_params["tsim"]
        self.x0 = self.env_params["x0"]
        self.integration_method = self.env_params.get("integration_method", "casadi")
        self.dt = self.tsim / self.N
        self.done = False

    def _setup_constraints(self):
        self.constraint_active = False
        self.r_penalty = False
        self.custom_constraint_active = False
        self.info = {}

        if self.env_params.get("constraints") is not None:
            self.constraints = self.env_params["constraints"]
            self.done_on_constraint = self.env_params["done_on_cons_vio"]
            self.r_penalty = self.env_params["r_penalty"]
            self.constraint_active = True
            self.n_con = self.constraints(self.x0, self.action_space.sample()).shape[0]
            self.info["cons_info"] = np.zeros((self.n_con, self.N, 1))

    def _initialize_model(self):
        model_mapping = {
            "cstr": cstr,
            "first_order_system": first_order_system,
            "nonsmooth_control": nonsmooth_control,
            "multistage_extraction": multistage_extraction,
            "cstr_series_recycle": cstr_series_recycle,
            "distillation_column": distillation_column,
            "multistage_extraction_reactive": multistage_extraction_reactive,
            "four_tank": four_tank,
            "photo_production": photo_production,
            "heat_exchanger": heat_exchanger,
            "biofilm_reactor": biofilm_reactor,
            "polymerisation_reactor": polymerisation_reactor,
            "crystallization": crystallization,
        }

        if self.env_params.get("custom_model") is not None:
            m = self.env_params["custom_model"]
            m.int_method = self.integration_method
            self.model = m
        else:
            model_name = self.env_params.get("model")
            if model_name not in model_mapping:
                raise ValueError(f"Model '{model_name}' not found in model_mapping.")
            self.model = model_mapping[model_name](int_method=self.integration_method)

    def _setup_state_dimensions(self):
        self.Nx = len(self.model.info()["states"])
        if self.SP is not None:
            self.Nx += len(self.SP)
        self.Nx_oracle = len(self.model.info()["states"])
        self.Nu = len(self.model.info()["inputs"])

    def _setup_disturbances(self):
        self.disturbance_active = False
        self.Nd = self.Nd_model = 0
        
        if self.env_params.get("disturbances") is not None:
            self.disturbance_active = True
            self.disturbances = self.env_params["disturbances"]
            self.Nd = len(self.disturbances)
            self.Nd_model = len(self.model.info()["disturbances"])
            self.Nu += self.Nd_model
            self.Nx += self.Nd
            
            dist_low = self.env_params["disturbance_bounds"]["low"]
            dist_high = self.env_params["disturbance_bounds"]["high"]
            extended_obs_low = np.concatenate((self.observation_space_base.low, dist_low))
            self.observation_space_base.low = extended_obs_low
            extended_obs_high = np.concatenate((self.observation_space_base.high, dist_high))
            self.observation_space_base.high = extended_obs_high
            
            self.observation_space_base = spaces.Box(
                low=extended_obs_low,
                high=extended_obs_high,
                dtype=np.float32
            )
            
            if self.normalise_o:
                self.observation_space = spaces.Box(
                    low=np.array([-1] * extended_obs_low.shape[0]),
                    high=np.array([1] * extended_obs_high.shape[0]),
                    dtype=np.float32
                )
            else:
                self.observation_space = self.observation_space_base

    def _setup_custom_reward(self):
        self.custom_reward = False
        if self.env_params.get("custom_reward") is not None:
            self.custom_reward = True
            self.custom_reward_f = self.env_params["custom_reward"]


    def _setup_partial_observations(self):
        self.partial_observation = False
        if self.env_params.get("partial_observation") is not None:
            self.partial_observation = self.env_params["partial_observation"]
    def _setup_uncertainty(self):
        self.uncertainty = False
        self.NUn = 0
        self.uncertainty_percentages = None

        if self.env_params.get('uncertainty_percentages') is not None or self.env_params.get('empirical_distribution') is not None:
            self.uncertainty = True
            if self.env_params.get('uncertainty_percentages') is not None:
                self.uncertainty_percentages = self.env_params['uncertainty_percentages']
                self.original_param_values = {
                    param: getattr(self.model, param)
                    for param in self.uncertainty_percentages
                    if param != "x0"
                }
                self.distribution = self.env_params.get("distribution")
            else:
                self.empirical_distribution = self.env_params.get('empirical_distribution')
                self.original_param_values = {
                    param: getattr(self.model, param)
                    for param in self.empirical_distribution
                    if param != "x0"
                }
            
            uncertainty_low = self.env_params["uncertainty_bounds"]["low"]
            uncertainty_high = self.env_params["uncertainty_bounds"]["high"]
            
            # Extend the observation space bounds to include uncertainties
            extended_obs_low = np.concatenate((self.observation_space_base.low, uncertainty_low))
            self.observation_space_base.low = extended_obs_low
            extended_obs_high = np.concatenate((self.observation_space_base.high, uncertainty_high))
            self.observation_space_base.high = extended_obs_high
            # Define the extended observation space
            self.observation_space_base = spaces.Box(
                low=extended_obs_low, high=extended_obs_high
            )
            
            if self.normalise_o:
                self.observation_space = spaces.Box(low=np.array([-1]*extended_obs_low.shape[0]), high=np.array([1]*extended_obs_high.shape[0]))
            else:
                self.observation_space = spaces.Box(
                low=extended_obs_low, high=extended_obs_high)
            

    def apply_uncertainties(self, value, percentage, distribution):
        if distribution == "uniform":
            noise = np.random.uniform(-percentage, percentage)
            noisy_value = value * (1 + noise)
        elif distribution == "normal":
            noisy_value = np.random.normal(value, percentage * value)
        return noisy_value
    
    def reset(self, seed:int=0, **kwargs) -> tuple[np.array, dict]:  
        """
        Reset the environment to its initial state.

        This method resets the environment's state, time, and other relevant variables.
        It's called at the beginning of each episode.

        Args:
            seed (int, optional): Seed for random number generator.
            **kwargs: Additional keyword arguments.

        Returns:
            tuple: A tuple containing:
                - numpy.array: The initial state observation.
                - dict: Additional information (e.g., initial reward).
        """
        self.t = 0
        
        self.int_eng = integration_engine(make_env, self.env_params)
        
        # Initialize state with potential random uncertainties in x0
        state = copy.deepcopy(self.env_params["x0"])
        if self.uncertainty_percentages is not None and "x0" in self.uncertainty_percentages:
            x0_uncertainty = self.uncertainty_percentages["x0"]
            for idx, uncertainty in enumerate(x0_uncertainty):
                state[idx] = self.apply_uncertainties(state[idx], uncertainty, self.distribution)
        
        # If disturbances are active, expand the initial state with disturbances
        if self.disturbance_active:
            initial_disturbances = []
            for k in self.model.info()["disturbances"]:
                if k in self.disturbances:
                    initial_disturbances.append(self.disturbances[k][0])

            # Append initial disturbances to the state
            state = np.concatenate((state, initial_disturbances))

        # Handle initial uncertainties
        if self.uncertainty:
            uncertain_params = []
            if self.uncertainty_percentages is not None:
                for param, percentage in self.uncertainty_percentages.items():
                    if param != "x0":  # x0 handled separately
                        original_value = self.original_param_values[param]
                        new_value = self.apply_uncertainties(original_value, percentage, self.distribution)
                        setattr(self.model, param, new_value)
                        uncertain_params.append(new_value)
                state = np.concatenate((state, uncertain_params))
            elif self.empirical_distribution is not None:
                for param, _ in self.empirical_distribution.items():
                    sample = np.random.choice(self.empirical_distribution[param])
                    setattr(self.model, param, sample)
                    uncertain_params.append(sample)  
                state = np.concatenate((state, uncertain_params))

                
        
        if self.a_delta:
            self.a_save = self.a_0
        
        self.state = state
        self.obs = copy.deepcopy(self.state)

        if self.custom_reward:
            r_init = 0
        elif not self.custom_reward:
            r_init = 0
        self.done = False
        
        if self.normalise_o:
            self.normobs = (
            2 * (self.obs - self.observation_space_base.low)
            / (self.observation_space_base.high - self.observation_space_base.low)
            - 1
            )
            self.info['obs'] = copy.deepcopy(self.normobs)
            obs_to_return = self.normobs
        else:
            self.info['obs'] = copy.deepcopy(self.obs)
            obs_to_return = self.obs

        if self.partial_observation:
            for i in range(self.Nx_oracle):
                if self.model.info()["states"][i] not in self.partial_observation:
                    obs_to_return[i] = 0
        self.info['r_init'] = r_init    
        return obs_to_return, self.info
    def step(self, action: np.array) -> tuple[np.array, float, bool, bool, dict]:
        """
        Perform one time step in the environment.

        This method takes an action, applies it to the environment, and returns
        the next state, reward, and other information.

        Args:
            action (numpy.array): The action to be taken in the environment.

        Returns:
            tuple: A tuple containing:
                - numpy.array: The next state observation.
                - float: The reward for the current step.
                - bool: Whether the episode has ended.
                - bool: Whether the episode was truncated.
                - dict: Additional information about the step.
        """

        
        # Create control vector
        uk = np.zeros(self.Nu)
        if self.normalise_a is True:
            action = (action + 1) * (
                self.env_params["a_space"]["high"] - self.env_params["a_space"]["low"]
            ) / 2 + self.env_params["a_space"]["low"]
        if self.normalise_a and self.a_delta:
            action = (action + 1) * (
                self.env_params["a_space"]["high"] - self.env_params["a_space"]["low"]
            ) / 2 + self.env_params["a_space"]["low"]
            action = self.a_save + action
            self.a_save = action
            
            self.a_save = np.clip(self.a_save,self.env_params['a_space_act']['low'],self.env_params['a_space_act']['high'])
        
        # Add disturbance to control vector
        if self.disturbance_active:
            uk[: self.Nu - len(self.model.info()["disturbances"])] = (
                action  # Add action to control vector
            )
            disturbance_values = []
            disturbance_values_state = []
            for i, k in enumerate(self.model.info()["disturbances"]):
                if k in self.disturbances:
                    current_disturbance_value = self.disturbances[k][self.t+1]
                    uk[self.Nu - self.Nd_model + i] = self.disturbances[k][self.t+1]  # Add disturbance to control vector
                    
                    disturbance_values_state.append(current_disturbance_value)
                    disturbance_values.append(current_disturbance_value)
                else:
                    default_value = self.model.info()["parameters"][str(k)]
                    uk[self.Nu - self.Nd_model + i] = self.model.info()["parameters"][
                        str(k)
                    ]  
                    # if there is no disturbance at this timestep, use the default value
                    disturbance_values.append(default_value)
                    

            # Update the state vector with current disturbance values
            if self.uncertainty_percentages is not None:
                self.state[self.Nx_oracle + len(self.SP) + len(self.uncertainty_percentages):] = disturbance_values_state
            else:
                self.state[self.Nx_oracle + len(self.SP) :] = disturbance_values_state
        else:
            uk = action  # Add action to control vector

        if self.t == 0: 
            # Check if constraints are violated
            constraint_violated = False
            if self.constraint_active:
                constraint_violated = self.constraint_check(self.state, uk)
        
        # Simulate one timestep
        if self.integration_method == "casadi":
                Fk = self.int_eng.casadi_step(self.state, uk)
                self.state[: self.Nx_oracle] = np.array(Fk["xf"].full()).reshape(
                    self.Nx_oracle
                )
        elif self.integration_method == "jax":
            self.state[: self.Nx_oracle] = self.int_eng.jax_step(self.state, uk)

        # For each set point, if it exists, append its value at the current time step to the list
        if self.SP is not None:
            SP_t = []
            for k in self.SP.keys():
                if k in self.SP:
                    SP_t.append(self.SP[k][self.t])
            
            self.state[self.Nx_oracle:self.Nx_oracle+len(self.SP)] = np.array(SP_t)
            
        # Update timestep
        self.t += 1

        # Check if constraints are violated
        constraint_violated = False
        if self.constraint_active:
            constraint_violated = self.constraint_check(self.state, uk)

        if self.t == self.N-1:
            self.done = True


        # Copy the obs from the state and add noise if the user requests this
        self.obs = copy.deepcopy(self.state)
        if self.env_params.get("noise", False):
            if self.noise_percentage_float:
                noise_percentage = self.env_params.get("noise_percentage", 0)
                self.obs[: self.Nx_oracle] += (
                    np.random.normal(0, 1, self.Nx_oracle)
                    * self.state[: self.Nx_oracle] * noise_percentage )
            else:
                for i in range(self.Nx_oracle):
                    if self.model.info()["states"][i] in self.noise_percentage:
                        self.obs[i] += (
                            np.random.normal(0, 1, 1)
                            * self.state[i] * self.noise_percentage[str(self.model.info()["states"][i])]
                        )
        


        if self.custom_reward:
            rew = self.custom_reward_f(self, self.obs, uk, constraint_violated) 
            
        elif not self.custom_reward and self.reward == "SP_reward_fn":
            rew = self.SP_reward_fn(self.state, constraint_violated)
            
        elif not self.custom_reward and self.reward != "SP_reward_fn":
            rew = self.batch_reward_fn(self.state, constraint_violated)
            
        else:
            raise ValueError(
                "Reward not valid function"
            )
        if self.normalise_o:
            self.normobs = (
            2 * (self.obs - self.observation_space_base.low)
            / (self.observation_space_base.high - self.observation_space_base.low)
            - 1
            )
            self.info['obs'] = copy.deepcopy(self.normobs)
            obs_to_return = self.normobs
        else:
            self.info['obs'] = copy.deepcopy(self.obs)
            obs_to_return = self.obs

        if self.partial_observation:
            for i in range(self.Nx_oracle):
                if self.model.info()["states"][i] not in self.partial_observation:
                    obs_to_return[i] = 0

        return obs_to_return, rew, self.done, False, self.info

    def batch_reward_fn(self, state: np.array, c_violated: bool) -> float:
        """
        Compute the reward function for a batch 

        Args:
            states (np.array): Current State of the system
            c_violated (bool): Whether any constraints were violated

        Returns:
            float: the computed reward
        """
        
        r = 0.0
        if self.t == self.N-1:
            # Get the full list of states from the model
            all_states = self.model.info()["states"]
            # Find indices of reward states that actually exist in the model
            reward_state_indices = [all_states.index(state_name) for state_name in self.reward_states if str(state_name) in all_states]
            # Calculate reward based on those indices
            r_scale = self.env_params.get("r_scale", {})
            for state_index in reward_state_indices:
                state_name = all_states[state_index]
                if self.maximise_reward == True:
                    r += state[state_index] * r_scale.get(state_name, 1)
                elif self.maximise_reward == False:
                    r -= state[state_index] * r_scale.get(state_name, 1)
            
            if self.r_penalty and c_violated:
                r -= 1000
                    
        return r
        
        
    def SP_reward_fn(self, state:np.array, c_violated:bool) -> float:
        """
        Compute the reward for the current state and action.

        This method calculates the reward based on the current state and whether
        any constraints were violated.

        Args:
            state (numpy.array): The current state of the system.
            c_violated (bool): Whether any constraints were violated.

        Returns:
            float: The computed reward.
        """
        
        r = 0.0

        for k in self.SP:
            i = self.model.info()["states"].index(k)
            r_scale = self.env_params.get("r_scale", {})
            r += (-((state[i] - np.array(self.SP[k][self.t])) ** 2)) * r_scale.get(k, 1)
            if self.r_penalty and c_violated:
                r -= 1000
        return r

    def con_checker(self, curr_state:np.array, inputs:np.array) -> bool:
        """
        Check if any constraints are violated for the given states.

        Args:
            model_states (list): List of state or input names to check.
            curr_state (list): List of corresponding state or input values.

        Returns:
            bool: True if any constraint is violated, False otherwise.
        """
        constraint = self.constraints
        g = constraint(curr_state, inputs)
        self.info['cons_info'][:,self.t,:] = g.reshape(g.shape[0],1)
        if np.any(self.info["cons_info"][:,self.t,:] > 0):
            return True
        else:
            return False
        

    def constraint_check(self, state: np.array, input:np.array) -> bool:
        """
        Check if any constraints are violated in the current step.

        This method checks both state and input constraints, as well as any
        custom constraints defined by the user.

        Args:
            state (numpy.array): The current state of the system.
            input (numpy.array): The current input (action) applied to the system.

        Returns:
            bool: True if any constraint is violated, False otherwise.
        """

        self.con_i = 0
        
        if self.normalise_a is True:
            input = (input + 1) * (
                self.env_params["a_space"]["high"] - self.env_params["a_space"]["low"]
            ) / 2 + self.env_params["a_space"]["low"]
        
        if self.normalise_o is True:
            state = (
                (state + 1)
                * (self.observation_space_base.high - self.observation_space_base.low)
                / 2
                + self.observation_space_base.low
            )
        constraint_violated = (
            self.con_checker(state,  input)
        )  # Check both inputs and states
        
        if constraint_violated and self.done_on_constraint:
            self.done = True
        return constraint_violated

    def get_rollouts(
        self,
        policies: dict,
        reps: int,
        oracle: bool=False,
        dist_reward: bool=False,
        MPC_params: bool=False,
        cons_viol: bool=False,
    ) -> tuple[policy_eval, dict]:
        """
        Generate rollouts for the given policies.

        This method simulates the environment for multiple episodes using the provided policies.

        Args:
            policies (dict): Dictionary of policies to evaluate.
            reps (int): Number of rollouts to perform.
            oracle (bool, optional): Whether to use an oracle model for evaluation. Defaults to False.
            dist_reward (bool, optional): Whether to use reward distribution. Defaults to False.
            MPC_params (bool, optional): Whether to use MPC parameters. Defaults to False.
            cons_viol (bool, optional): Whether to track constraint violations. Defaults to False.

        Returns:
            tuple: A tuple containing:
                - policy_eval: The policy evaluator object.
                - dict: Data from the rollouts.
        """
        # construct evaluator
        evaluator = policy_eval(
            make_env, policies, reps, self.env_params, oracle, MPC_params
        )
        # generate rollouts
        data = evaluator.get_rollouts()
        # return evaluator and data
        return evaluator, data

    def plot_rollout(
        self,
        policies: dict,
        reps: int,
        oracle: bool=False,
        dist_reward: bool=False,
        MPC_params: bool=False,
        cons_viol: bool=False,
        save_fig: bool=False,
    ) -> tuple[policy_eval, dict]:
        """
        Generate and plot rollouts for the given policies.

        This method simulates the environment for multiple episodes using the provided policies
        and plots the results.

        Args:
            policies (dict): Dictionary of policies to evaluate.
            reps (int): Number of rollouts to perform.
            oracle (bool, optional): Whether to use an oracle model for evaluation. Defaults to False.
            dist_reward (bool, optional): Whether to use reward distribution for plotting. Defaults to False.
            MPC_params (bool, optional): Whether to use MPC parameters. Defaults to False.
            cons_viol (bool, optional): Whether to track constraint violations. Defaults to False.
            save_fig (bool, optional): Whether to save the generated figures. Defaults to False.

        Returns:
            tuple: A tuple containing:
                - policy_eval: The policy evaluator object.
                - dict: Data from the rollouts.
        """
        # construct evaluator
        evaluator = policy_eval(
            make_env, policies, reps, self.env_params, oracle, MPC_params, cons_viol,save_fig
        )
        # generate rollouts
        data = evaluator.get_rollouts()
        # plot data from rollouts via the evaluator method
        evaluator.plot_data(data, dist_reward)
        # return constructed evaluator and data
        return evaluator, data