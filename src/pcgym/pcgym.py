import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .model_classes import (
    cstr,
    first_order_system,
    multistage_extraction,
    nonsmooth_control,
    cstr_series_recycle,
    distillation_column,
    multistage_extraction_reactive,
    four_tank,
    heat_exchanger,
    biofilm_reactor,
    polymerisation_reactor,
    crystallization
)
from .policy_evaluation import policy_eval
from .integrator import integration_engine
import copy
import typing

class make_env(gym.Env):
    """
    Class for RL-Gym Environment.

    This class creates a customizable environment for reinforcement learning tasks,
    compatible with the OpenAI Gym interface. It supports various chemical engineering
    models, custom constraints, disturbances, and reward functions.

    Attributes:
        env_params (dict): Parameters for environment configuration.
        action_space (gym.spaces.Box): The action space of the environment.
        observation_space (gym.spaces.Box): The observation space of the environment.
        model: The selected chemical engineering model.
        integration_method (str): Method used for numerical integration.
        constraint_active (bool): Whether constraints are active.
        disturbance_active (bool): Whether disturbances are active.
        uncertainty_active (bool): Whether uncertainties are active.
        custom_reward (bool): Whether a custom reward function is used.
    """

    def __init__(self, env_params: dict) -> None:
        """
        Initialize the environment with given parameters.

        Args:
            env_params (dict): Dictionary containing environment parameters including
                model selection, action and observation space definitions, simulation
                parameters, constraints, disturbances, and custom functions.
        """

        self.env_params = copy.deepcopy(env_params)
        self.a_delta = False
        if env_params.get("a_delta") is not None:
            self.a_delta = env_params["a_delta"]
            self.a_0 = env_params["a_0"]
            
        try:
            self.normalise_a = env_params["normalise_a"]
            self.normalise_o = env_params["normalise_o"]
        except Exception:
            self.normalise_a = True
            self.normalise_o = True

        # Define action and observation space
        if self.normalise_a is True:
            self.action_space = spaces.Box(
                low=np.array([-1] * env_params["a_space"]["low"].shape[0]),
                high=np.array([1] * env_params["a_space"]["high"].shape[0]),
            )
        else:
            self.action_space = spaces.Box(
                low=env_params["a_space"]["low"], high=env_params["a_space"]["high"]
            )

        self.SP = env_params["SP"]
        self.N = env_params["N"]
        self.tsim = env_params["tsim"]
        self.x0 = env_params["x0"]
        # Initial setup for observation space based on user-defined bounds
        base_obs_low = env_params["o_space"]["low"]
        base_obs_high = env_params["o_space"]["high"]
        self.observation_space_base = spaces.Box(low=base_obs_low, high=base_obs_high)

        if self.normalise_o: 
            self.observation_space = spaces.Box(low=np.array([-1]*base_obs_low.shape[0]), high=np.array([1]*base_obs_high.shape[0]))
        else:
            self.observation_space = spaces.Box(low=base_obs_low, high=base_obs_high)
        
        try:
            self.integration_method = env_params["integration_method"]
        except Exception:
            self.integration_method = "casadi"

        self.dt = self.tsim / self.N
        self.done = False

        # Constraints
        self.constraint_active = False
        self.r_penalty = False
        self.info = {}

        self.custom_constraint_active = False  # Initialize to False by default

        if env_params.get("constraints") is not None:
            self.constraints = env_params["constraints"]
            self.done_on_constraint = env_params["done_on_cons_vio"]
            self.r_penalty = env_params["r_penalty"]
            self.cons_type = env_params["cons_type"]
            self.constraint_active = True
            self.n_con = 0
            for _, con_list in self.constraints.items():
                self.n_con += len(con_list)
            self.info["cons_info"] = np.zeros((self.n_con, self.N, 1))

        if env_params.get("custom_con") is not None:
            self.done_on_constraint = env_params["done_on_cons_vio"]
            self.r_penalty = env_params["r_penalty"]
            self.custom_constraint_active = True
        
        # Select model
        model_mapping = {
            "cstr": cstr,
            "first_order_system": first_order_system,
            "nonsmooth_control": nonsmooth_control,
            "multistage_extraction": multistage_extraction,
            "cstr_series_recycle": cstr_series_recycle,
            "distillation_column": distillation_column,
            "multistage_extraction_reactive": multistage_extraction_reactive,
            "four_tank": four_tank,
            "heat_exchanger": heat_exchanger,
            "biofilm_reactor": biofilm_reactor,
            "polymerisation_reactor": polymerisation_reactor,
            "crystallization": crystallization 
        }

        # Load custom model if it is provided else load the selected standard model.
        if self.env_params.get("custom_model") is not None:
            m = self.env_params.get("custom_model")
            m.int_method  = self.integration_method
            self.model = m
        else:
            m = model_mapping.get(env_params["model"], None)
            self.model = m(
                int_method=self.integration_method
            )  # Initialise the model with the selected integration method

        # Handle the case where the model is not found (do this for all)
        if self.model is None:
            raise ValueError(
                f"Model '{env_params['model']}' not found in model_mapping."
            )

        # Import states and controls from model info
        self.Nx = len(self.model.info()["states"]) + len(self.SP)
        self.Nx_oracle = len(self.model.info()["states"])
        self.Nu = len(self.model.info()["inputs"])

        # Disturbances
        self.disturbance_active = False
        self.Nd = 0
        self.Nd_model = 0
        if env_params.get("disturbances") is not None:
            self.disturbance_active = True
            self.disturbances = env_params["disturbances"]
            self.Nd = len(self.disturbances)
            self.Nd_model = len(self.model.info()["disturbances"])
            self.Nu += len(self.model.info()["disturbances"])

            # Extend the state size by the number of disturbances
            self.Nx += self.Nd
            # user has defined disturbance_bounds within env_params
            disturbance_low = env_params["disturbance_bounds"]["low"]
            disturbance_high = env_params["disturbance_bounds"]["high"]
            # Extend the observation space bounds to include disturbances
            extended_obs_low = np.concatenate((base_obs_low, disturbance_low))
            extended_obs_high = np.concatenate((base_obs_high, disturbance_high))
            # Define the extended observation space
            self.observation_space_base = spaces.Box(
                low=extended_obs_low, high=extended_obs_high, dtype=np.float32
            )
            if self.normalise_o:
                self.observation_space = spaces.Box(low=np.array([-1]*extended_obs_low.shape[0]), high=np.array([1]*extended_obs_high.shape[0]))
            else:
                self.observation_space = spaces.Box(
                low=extended_obs_low, high=extended_obs_high, dtype=np.float32
            )
                
        # Custom reward function
        self.custom_reward = False # Set custom_reward to False by default
        if env_params.get("custom_reward") is not None:
            self.custom_reward = True
            self.custom_reward_f = env_params["custom_reward"]
        pass
        
        # Initial state and/or parametric uncertainty
        self.uncertainty = False 
        if env_params.get('uncertainty_percentages') is not None:
            self.uncertainty = True
            self.uncertainty_percentages = env_params['uncertainty_percentages']
            self.original_param_values = {}

            for param in self.uncertainty_percentages:
                if param != "x0":
                    self.original_param_values[param] = getattr(self.model, param)

            # User has defined uncertainty bounds within env_params
            uncertainty_low = env_params["uncertainty_bounds"]["low"]
            uncertainty_high = env_params["uncertainty_bounds"]["high"]

            # Extend the observation space bounds to include uncertainties
            extended_obs_low = np.concatenate((base_obs_low, uncertainty_low))
            extended_obs_high = np.concatenate((base_obs_high, uncertainty_high))

            # Define the extended observation space
            self.observation_space_base = spaces.Box(
                low=extended_obs_low, high=extended_obs_high, dtype=np.float32
            )
            
            if self.normalise_o:
                self.observation_space = spaces.Box(low=np.array([-1]*extended_obs_low.shape[0]), high=np.array([1]*extended_obs_high.shape[0]))
            else:
                self.observation_space = spaces.Box(
                low=extended_obs_low, high=extended_obs_high, dtype=np.float32)
        

    def apply_uncertainties(self, value, percentage):
        noise = np.random.uniform(-percentage, percentage)
        return value * (1 + noise)

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
        if self.uncertainty and "x0" in self.uncertainty_percentages:
            x0_uncertainty = self.uncertainty_percentages["x0"]
            for idx, percentage in x0_uncertainty.items():
                state[idx] = self.apply_uncertainties(state[idx], percentage)

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
            for param, percentage in self.uncertainty_percentages.items():
                if param != "x0":  # x0 handled separately
                    original_value = self.original_param_values[param]
                    new_value = self.apply_uncertainties(original_value, percentage)
                    setattr(self.model, param, new_value)
                    uncertain_params.append(new_value)
            state = np.concatenate((state, uncertain_params))
        
        if self.a_delta:
            self.a_save = self.a_0
        self.state = state
        
        if self.custom_reward:
            r_init = 0
        elif not self.custom_reward:
            r_init = 0
        self.done = False
        
        if self.normalise_o is True:
        
            self.normstate = (
                2
                * (self.state - self.observation_space_base.low)
                / (self.observation_space_base.high - self.observation_space_base.low)
                - 1
            )
            return self.normstate, {"r_init": r_init}
        else:
            return self.state, {"r_init": r_init}

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
                    current_disturbance_value = self.disturbances[k][self.t]
                    uk[self.Nu - self.Nd_model + i] = self.disturbances[k][self.t]  # Add disturbance to control vector
                    
                    disturbance_values_state.append(current_disturbance_value)
                    disturbance_values.append(current_disturbance_value)
                else:
                    default_value = self.model.info()["parameters"][str(k)]
                    uk[self.Nu - self.Nd_model + i] = self.model.info()["parameters"][
                        str(k)
                    ]  # if there is no disturbance at this timestep, use the default value

                    disturbance_values.append(default_value)
                    

            # Update the state vector with current disturbance values
            
            self.state[self.Nx_oracle + len(self.SP) :] = disturbance_values_state
        else:
            uk = action  # Add action to control vector
        # Simulate one timestep
        if self.integration_method == "casadi":
            
                Fk = self.int_eng.casadi_step(self.state, uk)
                self.state[: self.Nx_oracle] = np.array(Fk["xf"].full()).reshape(
                    self.Nx_oracle
                )
        elif self.integration_method == "jax":
            self.state[: self.Nx_oracle] = self.int_eng.jax_step(self.state, uk)
        # Check if constraints are violated
        constraint_violated = False
        if self.constraint_active or self.custom_constraint_active:
            constraint_violated = self.constraint_check(self.state, uk)

        # Compute reward
        if self.custom_reward:
            rew = self.custom_reward_f(self, self.state, uk, constraint_violated) 
        elif not self.custom_reward:
            rew = self.reward_fn(self.state, constraint_violated)

        # For each set point, if it exists, append its value at the current time step to the list
        SP_t = []
        for k in self.SP.keys():
            if k in self.SP:
                SP_t.append(self.SP[k][self.t])
        
        self.state[self.Nx_oracle:self.Nx_oracle+len(self.SP)] = np.array(SP_t)
        # Update timestep
        self.t += 1

        if self.t == self.N-1:
            self.done = True

        # add noise to state
        if self.env_params.get("noise", False):
            noise_percentage = self.env_params.get("noise_percentage", 0)
            self.state[: self.Nx_oracle] += (
                np.random.normal(0, 1, self.Nx_oracle)
                * self.state[: self.Nx_oracle]
                * noise_percentage
            )
        if self.normalise_o is True:
            self.normstate = (
                2
                * (self.state - self.observation_space_base.low)
                / (self.observation_space_base.high - self.observation_space_base.low)
                - 1
            )
            return self.normstate, rew, self.done, False, self.info
        else:
            return self.state, rew, self.done, False, self.info

    def reward_fn(self, state:np.array, c_violated:bool) -> float:
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

    def con_checker(self, model_states:list, curr_state:np.array) -> bool:
        """
        Check if any constraints are violated for the given states.

        Args:
            model_states (list): List of state or input names to check.
            curr_state (list): List of corresponding state or input values.

        Returns:
            bool: True if any constraint is violated, False otherwise.
        """
        for i, state in enumerate(model_states):
            if state in self.constraints:
                constraint = self.constraints[state]  # List of constraints
                cons_type = self.cons_type[state]  # List of cons type
                for j in range(len(constraint)):
                    curr_state_i = curr_state[i]
                    is_greater_violated = (
                        cons_type[j] == ">=" and curr_state_i <= constraint[j]
                    )
                    is_less_violated = (
                        cons_type[j] == "<=" and curr_state_i >= constraint[j]
                    )

                    if is_greater_violated or is_less_violated:
                        self.info["cons_info"][self.con_i, self.t, :] = abs(
                            curr_state_i - constraint[j]
                        )
                        return True
                    self.con_i += 1
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
        constraint_violated = False
        states = self.model.info()["states"]
        inputs = self.model.info()["inputs"]
        if self.env_params.get("custom_con") is not None:
            custom_con_vio_f = self.env_params["custom_con"]
            custom_con_vio = custom_con_vio_f(
                state, input
            )  # User defined constraint return True if violated
            assert isinstance(
                custom_con_vio, bool
            ), "Custom constraint must return a boolean (True == Violated)"
        else:
            custom_con_vio = False

        if self.constraint_active and self.custom_constraint_active:
            constraint_violated = (
                self.con_checker(states, state)
                or self.con_checker(inputs, input)
                or False
            )  # Check both inputs and states
        elif self.constraint_active:
            constraint_violated = (
                self.con_checker(states, state)
                or self.con_checker(inputs, input)
                or False
            )  # Check both inputs and states
        elif self.custom_constraint_active:
            constraint_violated = custom_con_vio

        self.done = self.done_on_constraint
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
