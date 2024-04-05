from pcgym.pcgym import make_env
import numpy as np
import random
from stable_baselines3 import SAC,PPO

T = 26
nsteps = 120
SP = {
    'Ca': [0.85 for i in range(int(nsteps/4))] + [0.9 for i in range(int(3*nsteps/4))],
}

disturbance = {'Caf': np.repeat([1, 1.05, 1], [nsteps//3, nsteps//3, nsteps//3])}

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
disturbance_space ={
  'low': np.array([1]),
  'high': np.array([1.05])
}
env_params = {
    'N': nsteps, # Number of time steps
    'tsim':T, # Simulation Time
    'SP':SP, #Setpoint
    'o_space' : observation_space, #Observation space
    'a_space' : action_space, # Action space
    'x0': np.array([0.8,330,0.8]), # Initial conditions (torch.tensor)
    'model': 'cstr_ode', #Select the model
    'normalise_a': True, #Normalise the actions
    'normalise_o':True, #Normalise the states,
    'r_scale': {'Ca':100},
    'noise':True, #Add noise to the states
    'integration_method': 'casadi', #Select the integration method
    'noise_percentage':0, #Noise percentage
    'disturbance_bounds':disturbance_space,
    'disturbances': disturbance
}

env = make_env(env_params)
model = PPO('MlpPolicy', env, verbose=1,learning_rate=1e-3)
model.learn(total_timesteps=10e4)
model.save('pse_track_ppo.zip')
model.load('pse_track_ppo.zip')
env.plot_rollout({'Random policy':model},1,oracle = True,dist_reward=True,MPC_params={'N':10,'R':0.001})