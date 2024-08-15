from pcgym import make_env
import numpy as np 
from stable_baselines3 import PPO

# Enter required setpoints for each state.
T = 26
nsteps = 60
SP = {
    'Ca': [0.85 for i in range(int(nsteps/2))] + [0.9 for i in range(int(nsteps/2))],
}


# Continuous box action space
action_space = {
    'low': np.array([295]),
    'high':np.array([302]) 
}

# Continuous box observation space
observation_space = {
    'low' : np.array([0.7,300,0.8]),
    'high' : np.array([1,350,0.9])  
}

r_scale ={
    'Ca': 1e3 #Reward scale for each state
}
env_params = {
    'N': nsteps, # Number of time steps
    'tsim':T, # Simulation Time
    'SP':SP, # Setpoint
    'o_space' : observation_space, # Observation space
    'a_space' : action_space, # Action space
    'x0': np.array([0.8,330,0.8]), # Initial conditions 
    'model': 'cstr', # Select the model
    'r_scale': r_scale, # Scale the L1 norm used for reward (|x-x_sp|*r_scale)
    'normalise_a': True, # Normalise the actions
    'normalise_o':True, # Normalise the states,
    'noise':True, # Add noise to the states
    'integration_method': 'casadi', # Select the integration method
    'noise_percentage':0.001, # Noise percentage

    
}
env = make_env(env_params)


disturbance = {'Ti': np.repeat([350, 345, 350], [nsteps//4, nsteps//2, nsteps//4])}
disturbance_space ={
  'low': np.array([320]),
  'high': np.array([350])
}
env = make_env({**env_params,'disturbance_bounds':disturbance_space, 'disturbances': disturbance})

# print(env.Nu, env.Nd_model)
# print(env.model.info[])
nsteps_learning = 3e3
# PPO_policy = PPO('MlpPolicy', env, verbose=1,learning_rate=0.001).learn(nsteps_learning)
# PPO.save(PPO_policy, 'test_pol')
PPO_policy = PPO.load('test_pol')



evaluator, data = env.plot_rollout({'PPO': PPO_policy}, oracle = True, reps = 1, MPC_params={'N':30,})
# print(data['oracle']['x'])