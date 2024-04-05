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

nsteps = 150
T = 3
SP = {
    'X1': [0 for i in range(int(nsteps))],
    #'X2': [0 for i in range(int(nsteps))] 
}

#Continuous box action space
action_space = {
    'low': np.array([-1]),
    'high':np.array([1]) 
}
#Continuous box observation space
observation_space = {
    'low' : np.array([-1,-1,-1]),
    'high' : np.array([1,1,1])  
}

r_scale ={
    'X1': 100,
}
env_params = {
    'N': nsteps, # Number of time steps
    'tsim':T, # Simulation Time
    'SP':SP, #Setpoint
    'o_space' : observation_space, #Observation space
    'a_space' : action_space, # Action space
    'x0': np.array([1,-1,0.]), # Initial conditions (torch.tensor)
    'model': 'bang_bang_control_ode', #Select the model
    'r_scale': r_scale, #Scale the L1 norm used for reward (|x-x_sp|*r_scale)
    'normalise_a': False, #Normalise the actions
    'normalise_o':False, #Normalise the states,
    'noise':False, #Add noise to the states
    'integration_method': 'casadi', #Select the integration method
    'noise_percentage':0 #Noise percentage
}
env = make_env(env_params)
# Load the saved policy
bang_pol = SAC.load('bang_sac.zip')

# Evaluate the policy and plot the rollout
evaluator, data = env.plot_rollout({'SAC': bang_pol}, reps=1, oracle=True, dist_reward=True, MPC_params={'N': 10, 'R': 0})

# Calculate ISE and IAE
ISE_1 = np.sum((data['oracle']['x'][0]) ** 2)
ISE_2 = np.sum((data['oracle']['x'][1]) ** 2)
IAE_1 = np.sum(np.abs(data['oracle']['x'][0]))
IAE_2 = np.sum(np.abs(data['oracle']['x'][1]))


ISE_1_SAC = np.sum((data['SAC']['x'][0]) ** 2)
ISE_2_SAC = np.sum((data['SAC']['x'][1]) ** 2)
IAE_1_SAC = np.sum(np.abs(data['SAC']['x'][0]))
IAE_2_SAC = np.sum(np.abs(data['SAC']['x'][1]))

# Print the results
print('X1 ISE', ISE_1)
print('X2 ISE', ISE_2)
print('Total ISE', ISE_1 + ISE_2)
print('X1 IAE', IAE_1)
print('X2 IAE', IAE_2)
print('Total IAE', IAE_1 + IAE_2)

print('X1 ISE', ISE_1_SAC)
print('X2 ISE', ISE_2_SAC)
print('Total ISE', ISE_1_SAC + ISE_2_SAC)
print('X1 IAE', IAE_1_SAC)
print('X2 IAE', IAE_2_SAC)
print('Total IAE', IAE_1_SAC + IAE_2_SAC)