import sys
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import gymnasium as gym
import torch
import torch as th
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy
import optuna
from torch.nn import Mish 
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
import warnings

# Clear the content of log files
save_dir = '/rds/general/user/jrn23/home/saferl-pcgym/optuna-safe-rl/'
# save_dir = '/rds/general/user/jrn23/home/optuna_mpc-2/'
open(os.path.join(save_dir, "job_error.txt"), 'w').close()
open(os.path.join(save_dir, "job_output.txt"), 'w').close()

# Suppress specific TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF warnings
import tensorflow as tf
tf.get_logger().setLevel('ERROR')  # Suppress TF warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Global seed for reproducibility
seed = 1990

def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Set the seed for NumPy and PyTorch
setup_seed(seed)

# Setting the backend for matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent GUI issues

# Add the 'pc-gym/src' directory to the Python path
sys.path.append(os.path.join(os.getcwd(), 'pc-gym', 'src'))

import pcgym
from pcgym.pcgym import make_env

# Directory configurations
server_dir = '/rds/general/user/jrn23/home/'
save_dir = os.path.join(server_dir, "saferl-pcgym/optuna-safe-rl/")
os.makedirs(save_dir, exist_ok=True)  # Ensure the save directory exists

##################################################################################
# Environment and RL Definition
##################################################################################

# Global params
T = 26
nsteps = 100

#Enter required setpoints for each state. Enter None for states without setpoints.
SP = {
    'T': [324.475443431599 for _ in range(5)] + [340.0 for _ in range(nsteps - 5)],
}

#Continuous box action space
action_space = {
    'low': np.array([250]),
    'high': np.array([350])
}

#Continuous box observation space ([CA, T, CA_Setpoint, T_Setpoint])
observation_space = {
    'low' : np.array([0.0,200,300]),
    'high' : np.array([1,600,400])
}

r_scale ={
    'T': 1e-6 #Reward scale for each state,
}

# Define disturbance bounds
disturbance_bounds = {
    'low': np.array([350]),
    'high': np.array([450])
}

# Environment parameters
env_params_template = {
    'Nx': 2,
    'N': 100,
    'tsim': 26,
    'Nu': 1,
    'SP': SP,
    'o_space': observation_space,
    'a_space': action_space,
    'x0': np.array([0.87725294608097, 324.475443431599, 324.475443431599]),
    'model': 'cstr_ode',
    'r_scale': r_scale,
    'normalise_a': True,
    'normalise_o': True,
    'noise': True,
    'integration_method': 'casadi',
    'noise_percentage': 0.001,
    'disturbance_bounds': disturbance_bounds 
}

# Function to create random disturbances
def create_random_disturbances(seed, nsteps, low=350, high=450):
    np.random.seed(seed)
    values = np.random.uniform(low, high, 3) # Generate three random disturbance values within the specified range
    times = np.sort(np.random.choice(range(1, nsteps - 1), 2, replace=False))  # Select two unique time steps for disturbances
    times = np.append(times, nsteps)  # Append the total number of steps to get three periods
    times = np.diff([0] + times.tolist())  # Calculate the duration of each disturbance period
    disturbances = {'Ti': np.repeat(values, times)}  # Repeat the disturbance values according to the calculated durations
    return disturbances

# Create multiple environments with different disturbances
def create_multiple_envs(n_envs, seed):
    envs = []
    for i in range(n_envs):
        env_params = env_params_template.copy()
        disturbances = create_random_disturbances(seed + i, nsteps)
        env_params.update({'disturbances': disturbances})
        envs.append(lambda: make_env(env_params))
    return envs

# Learning rate decay schedule
def cosine_annealing_schedule(progress_remaining: float, num_cycles=1, min_lr=0.005, max_lr=0.01):
    progress = 1.0 - progress_remaining
    lr = min_lr + (max_lr - min_lr)/2 * (1 + np.cos(np.pi * num_cycles * progress))
    return lr

# Configuration for reinforcement learning model
config = {
    "policy": 'MlpPolicy', # default: MlpPolicy
    "learning_rate": "cosine", # default: 0.01
    "gamma": 0.99, #default: 0.99, the discount factor
    "total_timesteps": 5.0e5, # base: 1.0e5
    "gae_lambda": 0.99, # default: 0.95, Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    "ent_coef": 0.005, # default: 0.0, Entropy coefficient for the loss calculation
    "batch_size": 64, # default: 64 # base: 64
    "n_steps": 128, # default: 2048 # base: 120
    "n_epochs": 10, # default: 10, number of times the optimizer will go over the collected batch of experiences to update the model
    # rollout buffer size is (n_steps * n_envs) where n_envs is number of environment copies running in parallel
    "clip_range": 0.2, # default: 0.2
    "clip_range_vf": None, # default: None # base: 0.2
    "seed": 1990,
    "check_freq": 100, # base: 12000 (~100 episodes)
    "n_eval_episodes": 10, # evaluate the agent over 100 episodes in the evaluation environment
    "n_trials": 500,  # Number of trials for Optuna optimization (base: 100)
    "positive_definiteness_penalty_weight": 0,  # Set to 0
    "derivative_penalty_weight": 0,  # Set to 0
}

# Add noise_percentage from env_params to config
config['noise_percentage'] = env_params_template['noise_percentage']

# Custom NN Architecture
policy_kwargs = dict(
    activation_fn=th.nn.Tanh,
    net_arch=dict(pi=[32, 32], vf=[32, 32])
)

# Instantiate the environment with multiple parallel environments
def create_parallel_envs(n_envs, seed):
    envs = create_multiple_envs(n_envs, seed)
    return SubprocVecEnv(envs)

# Create parallel environments for evaluation
def create_eval_env(seed):
    env_params = env_params_template.copy()
    disturbances = create_random_disturbances(seed, nsteps)
    env_params.update({'disturbances': disturbances})
    env_params.update({'disturbance_bounds': disturbance_bounds})  # Ensure bounds are included
    return make_env(env_params)

# Calculate valid batch sizes for different n_steps values
def get_valid_batch_sizes(n_steps, n_envs):
    return [i for i in range(16, 129) if n_steps * n_envs % i == 0]

# Store valid batch sizes for different n_steps values
n_steps_values = list(range(32, 257))
valid_batch_sizes_dict = {n: get_valid_batch_sizes(n, 10) for n in n_steps_values}

# Instantiate the evaluation environment
eval_env = create_eval_env(seed)  # Evaluation environment

##################################################################################
# Optuna - RL Agent Hyperparameter Tuning
##################################################################################

def run_pc_gym_paper_tuning_mult_disturb_study(env_params_template, save_dir, n_trials, n_envs, rollout_number, seed):
    # Adjust the make_env function to accept env_params as an argument
    def create_env(env_params):
        env = make_env(env_params)
        return env

    def objective(trial):
        # Use the global seed directly inside function
        torch.manual_seed(seed)  # Re-seed if needed for each trial

        # Hyperparameters to tune
        # positive_definiteness_penalty_weight = trial.suggest_float('positive_definiteness_penalty_weight', 5, 20)
        # derivative_penalty_weight = trial.suggest_float('derivative_penalty_weight', 1, 15)
        ent_coef = trial.suggest_float('ent_coef', 0.001, 0.01)
        min_lr = trial.suggest_float('min_lr', 0.002, 0.01)
        max_lr = trial.suggest_float('max_lr', 0.01, 0.02)
        n_steps = trial.suggest_int('n_steps', 32, 256)

        # Get valid batch sizes for the current n_steps value
        valid_batch_sizes = valid_batch_sizes_dict.get(n_steps, [])
        if not valid_batch_sizes:
            valid_batch_sizes = [i for i in range(16, 129) if (n_steps * n_envs) % i == 0]
            if not valid_batch_sizes:
                valid_batch_sizes = [16, 32, 64, 128]  # Common fallback batch sizes
                batch_size = next((bs for bs in valid_batch_sizes if (n_steps * n_envs) % bs == 0), None)
                if batch_size is None:
                    # Adjust n_steps to ensure we have a valid batch size
                    n_steps = next(ns for ns in n_steps_values if any((ns * n_envs) % bs == 0 for bs in valid_batch_sizes))
                    valid_batch_sizes = valid_batch_sizes_dict.get(n_steps, [])
                    batch_size = valid_batch_sizes[np.random.randint(len(valid_batch_sizes))]

        # Ensure batch_size is compatible with n_steps
        batch_size = valid_batch_sizes[np.random.randint(len(valid_batch_sizes))]

        # Ensure the selected batch_size is valid
        assert (n_steps * n_envs) % batch_size == 0, f"Invalid batch size {batch_size} for n_steps {n_steps}"

        # Tune network architecture
        pi_units = [2 ** trial.suggest_int(f'pi_units_{i}', 3, 5) for i in range(2)]
        vf_units = [2 ** trial.suggest_int(f'vf_units_{i}', 3, 5) for i in range(2)]
        
        policy_kwargs = dict(
            activation_fn=th.nn.Tanh,
            net_arch=dict(pi=pi_units, vf=vf_units)
        )

        # Update config with the trial's suggestions
        config.update({
            # 'positive_definiteness_penalty_weight': positive_definiteness_penalty_weight,
            # 'derivative_penalty_weight': derivative_penalty_weight,
            'ent_coef': ent_coef,
            'batch_size': batch_size,
            'n_steps': n_steps,
            'learning_rate': lambda progress_remaining: cosine_annealing_schedule(progress_remaining, min_lr=min_lr, max_lr=max_lr)
        })

        # Create and train the model with parallel environments
        env = create_parallel_envs(n_envs, seed)
        model = PPO(
            config['policy'],
            env,
            learning_rate=config['learning_rate'],
            clip_range=config['clip_range'],
            clip_range_vf=config['clip_range_vf'],
            batch_size=config['batch_size'],
            n_steps=config['n_steps'],
            seed=seed,
            gamma=config['gamma'],
            gae_lambda=config['gae_lambda'],
            ent_coef=config['ent_coef'],
            n_epochs=config['n_epochs'],
            policy_kwargs=policy_kwargs,
            tensorboard_log=f"{save_dir}/runs/ppo",
            positive_definiteness_penalty_weight=config['positive_definiteness_penalty_weight'],  # Set to 0
            derivative_penalty_weight=config['derivative_penalty_weight'],  # Set to 0
        )
        eval_env = create_eval_env(seed)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=save_dir,
            log_path=save_dir,
            eval_freq=config['check_freq'],
            n_eval_episodes=config['n_eval_episodes'],
            deterministic=True,
            render=False
        )
        model.learn(total_timesteps=config['total_timesteps'], callback=eval_callback)

        # Evaluate the model
        mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=config['n_eval_episodes'])

        # Cleanup
        env.close()
        eval_env.close()

        return mean_reward

    # Initialize and run the Optuna study
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    # Save final best parameters and best trial info
    results_file_path = os.path.join(save_dir, "optimization_pc_gym_paper_mult_disturb_results.txt")
    with open(results_file_path, "a") as file:
        best_trial = study.best_trial
        file.write(f"Best trial number: {best_trial.number}\n")
        file.write(f"Best parameters: {best_trial.params}\n")
        file.write(f"Best observed reward: {best_trial.value}\n")

    return study
    
##################################################################################
# Main Execution
##################################################################################

def main():
    setup_seed(seed)  # Setup seed at the start of main execution
    
    # Number of trial evaluations
    n_trials = config['n_trials']

    # Number of parallel environments
    n_envs = 10

    # Number of rollouts for evaluation
    rollout_number = 5
    
    # Suppress output for cleaner execution
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = open(os.devnull, 'w')
    sys.stderr = open(os.devnull, 'w')

    # Run RL agent tuning
    rl_study = run_pc_gym_paper_tuning_mult_disturb_study(env_params_template, save_dir, n_trials, n_envs, rollout_number, seed)

    # Re-enable output
    sys.stdout = original_stdout
    sys.stderr = original_stderr

if __name__ == "__main__":
    main()
