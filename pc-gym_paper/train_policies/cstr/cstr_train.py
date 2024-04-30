import sys
sys.path.append("..")  # Adds higher directory to python modules path. 

from pcgym import make_env
from callback import LearningCurveCallback
import numpy as np

from stable_baselines3 import PPO, DDPG, SAC

# Define environment
T = 26
nsteps = 120

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

env_params = {
    'N': nsteps, 
    'tsim':T, 
    'SP':SP, 
    'o_space' : observation_space, 
    'a_space' : action_space,
    'x0': np.array([0.8,330,0.8]),
    'r_scale': {'Ca':1e3},
    'model': 'cstr_ode', 
    'normalise_a': True, 
    'normalise_o':True, 
    'noise':True, 
    'integration_method': 'casadi', 
    'noise_percentage':0.001, 
}

env = make_env(env_params)

# Global timesteps
nsteps_train = 1e3

# Train SAC 
log_file = "learning_curves\SAC_CSTR_LC.csv"
SAC_CSTR =  SAC("MlpPolicy", env, verbose=1, learning_rate=0.01)
callback = LearningCurveCallback(log_file=log_file)
SAC_CSTR.learn(nsteps_train,callback=callback)

# Save SAC Policy 
SAC_CSTR.save('policies\SAC_CSTR.zip')

# Train PPO 
log_file = "PPO_CSTR_LC.csv"
PPO_CSTR =  PPO("MlpPolicy", env, verbose=1, learning_rate=0.01)
callback = LearningCurveCallback(log_file=log_file)
PPO_CSTR.learn(nsteps_train,callback=callback)

# Save SAC Policy 
SAC_CSTR.save('policies\PPO_CSTR.zip')

# Train SAC 
log_file = "DDPG_CSTR_LC.csv"
DDPG_CSTR =  DDPG("MlpPolicy", env, verbose=1, learning_rate=0.01)
callback = LearningCurveCallback(log_file=log_file)
DDPG_CSTR.learn(nsteps_train,callback=callback)

# Save DDPG Policy 
DDPG_CSTR.save('policies\DDPG_CSTR.zip')