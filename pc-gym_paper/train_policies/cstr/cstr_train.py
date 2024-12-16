import sys
sys.path.append("..")  # Adds higher directory to python modules path for callback class. 
sys.path.append("..\..\..\src\pcgym") # Add local pc-gym files to path.

from pcgym import make_env
from callback import LearningCurveCallback
import numpy as np
from stable_baselines3 import PPO, DDPG, SAC
from custom_reward import sp_track_reward

# Define environment
T = 26
nsteps = 60
training_seed = 1
SP = {
    'Ca': [0.85 for i in range(int(nsteps/3))] + [0.9 for i in range(int(nsteps/3))]+ [0.87 for i in range(int(nsteps/3))],
}

action_space = {
    'low': np.array([295]),
    'high':np.array([302]) 
}

observation_space = {
    'low' : np.array([0.7,300,0.8]),
    'high' : np.array([1,350,0.9])  
}

r_scale = {'Ca':1e3}

# Define reward to be equal to the OCP (i.e the same as the oracle)
env_params = {
    'N': nsteps, 
    'tsim':T, 
    'SP':SP, 
    'o_space' : observation_space, 
    'a_space' : action_space,
    'x0': np.array([0.8,330,0.8]),
    'r_scale': r_scale,
    'model': 'cstr', 
    'normalise_a': True, 
    'normalise_o':True, 
    'noise':True, 
    'integration_method': 'casadi', 
    'noise_percentage':0.001, 
    'custom_reward': sp_track_reward
}

env = make_env(env_params)

# Global timesteps
nsteps_train = 5e4
training_reps = 1
for r_i in range(training_reps):
    print(f'Training repition:{r_i+1}')
    # Train SAC 
    log_file = f"learning_curves\SAC_CSTR_LC_rep_{r_i}.csv"
    SAC_CSTR =  SAC("MlpPolicy", env, verbose=1, learning_rate=0.01, seed=training_seed)
    callback = LearningCurveCallback(log_file=log_file)
    SAC_CSTR.learn(nsteps_train,callback=callback)

    # Save SAC Policy 
    SAC_CSTR.save(f'policies\SAC_CSTR_rep_{r_i}.zip')

    # Train PPO 
    log_file = f"learning_curves\PPO_CSTR_LC_rep_{r_i}.csv"
    PPO_CSTR =  PPO("MlpPolicy", env, verbose=1, learning_rate=0.001, seed = training_seed)
    callback = LearningCurveCallback(log_file=log_file)
    PPO_CSTR.learn(nsteps_train,callback=callback)

    # Save SAC Policy 
    PPO_CSTR.save(f'policies\PPO_CSTR_rep_{r_i}.zip')

    # Train SAC 
    log_file = f'learning_curves\DDPG_CSTR_LC_rep_{r_i}.csv'
    DDPG_CSTR =  DDPG("MlpPolicy", env, verbose=1, learning_rate=0.001, seed=training_seed)
    callback = LearningCurveCallback(log_file=log_file)
    DDPG_CSTR.learn(nsteps_train,callback=callback)

    # Save DDPG Policy 
    DDPG_CSTR.save(f'policies\DDPG_CSTR_rep_{r_i}.zip')