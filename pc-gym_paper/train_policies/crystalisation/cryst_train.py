import sys
sys.path.append("..")  # Adds higher directory to python modules path for callback class. 
sys.path.append("..\..\..\src\pcgym") # Add local pc-gym files to path.

from pcgym import make_env
from callback import LearningCurveCallback
import numpy as np
from stable_baselines3 import PPO, DDPG, SAC
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Define environment
T = 55
nsteps = 55


# Define reward to be equal to the OCP (i.e the same as the oracle)
def oracle_reward(self,x,u,con):

    
    SP = self.SP
    
    CV = (x[2]*x[0]/(x[1]**2) - 1)**0.5
    ln = x[1]/x[0]
    r = -1*(abs(SP['CV'][self.t] - CV) + abs((SP['Ln'][self.t]-ln)/10))
    return r

SP = {
        'CV': [1 for i in range(int(nsteps))],
        'Ln': [15 for i in range(int(nsteps))]
    }

action_space = {
    'low': np.array([-1]),
    'high':np.array([1])
}
action_space_act = {
    'low': np.array([10]),
    'high':np.array([40])
}
lbMu0 = 0  # 0.1
ubMu0 = 1e20
lbMu1 = 0  # 0.1
ubMu1 = 1e20
lbMu2 = 0
ubMu2 = 1e20
lbMu3 = 0
ubMu3 = 1e20
lbC = 0
ubC = 0.5
lbT = 0
ubT = 40
observation_space = {
    'low' : np.array([lbMu0, lbMu1, lbMu2, lbMu3, lbC, 0, 0,  0.9, 14]),
    'high' : np.array([ubMu0, ubMu1, ubMu2, ubMu3, ubC, 2, 20, 1.1, 16])  
}
CV_0 = np.sqrt(1800863.24079725 * 1478.00986666666/ (22995.8230590611**2) - 1)
Ln_0 =  22995.8230590611 / ( 1478.00986666666 + 1e-6)
env_params_cryst = {
    'N': nsteps,
    'tsim':T,
    'SP':SP,
    'o_space' : observation_space,
    'a_space' : action_space,
    'x0': np.array([1478.00986666666, 22995.8230590611, 1800863.24079725, 248516167.940593, 0.15861523304,CV_0, Ln_0 , 1, 15]),
    'model': 'crystallization', 
    'normalise_a': True, #Normalise the actions
    'normalise_o':True, #Normalise the states,
    'noise':False, #Add noise to the states
    'noise_percentage':0.001,
    'integration_method': 'casadi',
    'a_0':39,
    'a_delta':True,
    'a_space_act':action_space_act,
}
env = make_env(env_params_cryst)

# Global timesteps
nsteps_train = 0.5e4
training_reps = 1
for r_i in range(training_reps):
    print(f'Training repition: {r_i+1}')
    # Train SAC 
    print('Training using SAC...')
    log_file = f"learning_curves\SAC_cryst_LC_rep_{r_i}.csv"
    SAC_cryst =  SAC("MlpPolicy", env, verbose=1, learning_rate=0.01)
    callback = LearningCurveCallback(log_file=log_file)
    
    SAC_cryst.learn(nsteps_train,callback=callback)

    # Save SAC Policy 
    SAC_cryst.save(f'policies\SAC_cryst_rep_{r_i}.zip')
    env.plot_rollout({'SAC':SAC_cryst},oracle=False, reps=1)
    # Train PPO 
    # print('Training using PPO...')
    # log_file = f"learning_curves\PPO_cryst_LC_rep_{r_i}.csv"
    # PPO_cryst =  PPO("MlpPolicy", env, verbose=1, learning_rate=0.001)
    # callback = LearningCurveCallback(log_file=log_file)
    # PPO_cryst.learn(nsteps_train,callback=callback)

    # # Save PPO Policy 
    # PPO_cryst.save(f'policies\PPO_cryst_rep_{r_i}.zip')

    # Train DDPG
    # print('Training using DDPG...')
    # log_file = f'learning_curves\DDPG_cryst_LC_rep_{r_i}.csv'
    # DDPG_cryst =  DDPG("MlpPolicy", env, verbose=1, learning_rate=0.001)
    # callback = LearningCurveCallback(log_file=log_file)
    # DDPG_cryst.learn(nsteps_train,callback=callback)

    # Save DDPG Policy 
    # DDPG_cryst.save(f'policies\DDPG_cryst_rep_{r_i}.zip')