import sys
import os
import pickle
import glob
sys.path.append("../train_policies")  # Adds higher directory to python modules path for callback class. 
sys.path.append("..\..\..\src\pcgym") # Add local pc-gym files to path.


from pcgym import make_env
from stable_baselines3 import PPO, DDPG, SAC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

r_scale = {'Ca':1e3}

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
    'noise_percentage':0.0005, 

}

env = make_env(env_params)


# Collect r-squared data
DDPG_squared = []
for model_path in glob.glob('../policies_inter/DDPG_CSTR_r_squared_*'):
    model = DDPG.load(model_path)
    DDPG_squared.append(model)

data_DDPG_square = []
for pol_i in DDPG_squared:
  _, data = env.get_rollouts({'pol_i':pol_i}, reps=1, )
  data_DDPG_square.append(data)


# Collect r-sparse data
DDPG_sparse = []
for model_path in glob.glob('../policies_inter/DDPG_CSTR_r_sparse_*'):
    model = DDPG.load(model_path)
    DDPG_sparse.append(model)

data_DDPG_sparse = []
for pol_i in DDPG_sparse:
  _, data = env.get_rollouts({'pol_i':pol_i}, reps=1, )
  data_DDPG_sparse.append(data)


# Collect r-squared data
DDPG_abs = []
for model_path in glob.glob('../policies_inter/DDPG_CSTR_r_abs_*'):
    model = DDPG.load(model_path)
    DDPG_abs.append(model)

data_DDPG_abs = []
for pol_i in DDPG_abs:
  _, data = env.get_rollouts({'pol_i':pol_i}, reps=1, )
  data_DDPG_abs.append(data)


# Save data from all three reward functions
print('Saving data')
# Save data_DDPG_square
with open('data_DDPG_square.pkl', 'wb') as f:
    pickle.dump(data_DDPG_square, f)

# Save data_DDPG_sparse
with open('data_DDPG_sparse.pkl', 'wb') as f:
    pickle.dump(data_DDPG_sparse, f)

# Save data_DDPG_abs
with open('data_DDPG_abs.pkl', 'wb') as f:
    pickle.dump(data_DDPG_abs, f)