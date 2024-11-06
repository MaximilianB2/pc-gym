import sys
sys.path.append("..\..\src\pcgym") # Add local pc-gym files to path.
import numpy as np
from stable_baselines3 import SAC, DDPG
from pcgym import make_env
from custom_reward import con_reward


# Define environment
T = 26
nsteps = 60

SP = {
    'Ca': [0.86,0.9] * (nsteps // 2)}

action_space = {
    'low': np.array([290]),
    'high':np.array([310]) 
}

observation_space = {
    'low' : np.array([0.7,300,0.8]),
    'high' : np.array([1,350,0.9])  
}

r_scale = {'Ca':1}


# Define reward to be equal to the OCP (i.e the same as the oracle)



cons = {'T':[327,321]}

cons_type = {'T':['<=','>=']}

env_params_con = {
    'N': nsteps, 
    'tsim':T, 
    'SP':SP, 
    'o_space' : observation_space, 
    'a_space' : action_space,
    'x0': np.array([0.8,325,0.86]),
    'r_scale': r_scale,
    'model': 'cstr', 
    'normalise_a': True, 
    'normalise_o':True, 
    'noise':True, 
    'integration_method': 'casadi', 
    'noise_percentage':0.001, 
    'custom_reward': con_reward,
    'done_on_cons_vio': False,
    'constraints': cons, 
    'r_penalty': False,
    'cons_type': cons_type,
}
env_con = make_env(env_params_con)


env_params = env_params_con
env_params.pop('done_on_cons_vio')
env_params.pop('constraints')
env_params.pop('r_penalty')
env_params.pop('cons_type')

env = make_env(env_params)

DDPG_constraint = DDPG("MlpPolicy", env_con, verbose=1, learning_rate=0.001).learn(2.5e4)
DDPG_norm = DDPG("MlpPolicy", env, verbose=1, learning_rate=0.001).learn(2.5e4)


DDPG_constraint.save('DDPG_constraint.zip')
DDPG_norm.save('DDPG_norm.zip')

DDPG_constraint = DDPG.load('DDPG_constraint.zip')
DDPG_norm = DDPG.load('DDPG_norm.zip')
_, con_data = env_con.get_rollouts({'DDPG':DDPG_constraint,}, reps=50, oracle=True, MPC_params={'N':20,})

np.save('constraint_rollout_data.npy', con_data, allow_pickle=True)

_, norm_data = env_con.get_rollouts({'DDPG':DDPG_norm,}, reps=50)

np.save('norm_rollout_data.npy', norm_data, allow_pickle=True)
