from dataclasses import dataclass
import numpy as np 
import torch
import matplotlib.pyplot as plt 
import gymnasium as gym
import torch.nn.functional as F
import copy
from stable_baselines3 import PPO,SAC
from pcgym.pcgym import make_env
from pcgym.evaluation_metrics import reproducibility_metric

 
import jax.numpy as jnp
#Global params
T = 26
nsteps = 100


#Enter required setpoints for each state. Enter None for states without setpoints.
SP = {
    'Ca': [0.85 for i in range(int(nsteps/2))] + [0.9 for i in range(int(nsteps/2))],
}

#Continuous box action space
action_space = {
    'low': np.array([295]),
    'high':np.array([302]) 
}
#Continuous box observation space
observation_space = {
    'low' : np.array([0.7,300,0.8]),
    'high' : np.array([1,350,0.9])  
}

r_scale ={
    'Ca': 5 #Reward scale for each state
}
env_params = {
    'N': nsteps, # Number of time steps
    'tsim':T, # Simulation Time
    'SP':SP, #Setpoint
    'o_space' : observation_space, #Observation space
    'a_space' : action_space, # Action space
    'x0': np.array([0.8,330,0.8]), # Initial conditions (torch.tensor)
    'model': 'cstr_ode', #Select the model
    'r_scale': r_scale, #Scale the L1 norm used for reward (|x-x_sp|*r_scale)
    'normalise_a': True, #Normalise the actions
    'normalise_o':True, #Normalise the states,
    'noise':True, #Add noise to the states
    'integration_method': 'casadi', #Select the integration method
    'noise_percentage':0.001 #Noise percentage
}
env = make_env(env_params)

SAC_init = SAC.load('SAC_cstr.zip') # NOTE This will fail unless people run the file from the correct directory
PPO_init = PPO.load('PPO_cstr.zip') # NOTE This will fail unless people run the file from the correct directory

# NOTE this is how we should roll out and assess the performance of the policies this is untested due to issues with import of reoproducibility_metric
evaluator, data = env.plot_rollout({'SAC':SAC_init,'PPO':PPO_init}, reps=100, oracle = False,dist_reward=True)
policy_measure = reproducibility_metric(dispersion='mad', performance='mean', scalarised_weight=0.3)
scalarised_performance = policy_measure.evaluate(evaluator, component='r')

print('scalarised_performance', scalarised_performance)