import sys

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
    'noise_percentage':0.001, 

}

env = make_env(env_params)

# Load trained policies (l2 norm reward)
# SAC_cstr_l2 = SAC.load('../policies/SAC_CSTR_r_squared')
# PPO_cstr_l2 = PPO.load('../policies/PPO_CSTR_r_squared')
DDPG_cstr_l2 = DDPG.load('../policies/DDPG_CSTRr_squared')

# Visualise policies with the oracle
# evaluator, data = env.plot_rollout({'SAC':SAC_cstr_l2,'PPO':PPO_cstr_l2,'DDPG':DDPG_cstr_l2}, reps=3, oracle=False, MPC_params={'N':20,'R':0.001},save_fig=False)


# Load trained policies (sparse)
# SAC_cstr_sparse = SAC.load('../policies/SAC_CSTR_r_sparse')
# PPO_cstr_sparse = PPO.load('../policies/PPO_CSTR_r_sparse')
DDPG_cstr_sparse = DDPG.load('../policies/DDPG_CSTRr_sparse')

# Visualise policies with the oracle
# evaluator, data = env.plot_rollout({'SAC':SAC_cstr_sparse,'PPO':PPO_cstr_sparse,'DDPG':DDPG_cstr_sparse}, reps=3, oracle=False, MPC_params={'N':20,'R':0.001},save_fig=False)

# Load trained policies (abs)
# SAC_cstr_abs = SAC.load('../policies/SAC_CSTR_r_abs')
# PPO_cstr_abs = PPO.load('../policies/PPO_CSTR_r_abs')
DDPG_cstr_abs = DDPG.load('../policies/DDPG_CSTRr_abs')

# Visualise policies with the oracle
# evaluator, data = env.plot_rollout({'SAC':SAC_cstr_sparse,'PPO':PPO_cstr_sparse,'DDPG':DDPG_cstr_sparse}, reps=3, oracle=False, MPC_params={'N':20,'R':0.001},save_fig=False)
evaluator, data = env.plot_rollout({'DDPG (r_squared)':DDPG_cstr_l2, 'DDPG (r_abs)': DDPG_cstr_abs, 'DDPG (r_sparse)':DDPG_cstr_sparse}, reps=3, oracle=False, MPC_params={'N':20,'R':0.001},save_fig=False)