import sys
sys.path.append("..")  # Adds higher directory to python modules path for callback class. 
sys.path.append("..\..\..\src\pcgym") # Add local pc-gym files to path.

from pcgym import make_env
from callback import LearningCurveCallback
import numpy as np
from stable_baselines3 import PPO, DDPG, SAC

# Define environment
T = 2e4
nsteps = 10000


# Define reward to be equal to the OCP (i.e the same as the oracle)
def oracle_reward(self,x,u,con):
    Sp_i = 0
    cost = 0 
    R = 4
    for k in self.env_params["SP"]:
        i = self.model.info()["states"].index(k)
        SP = self.SP[k]
     
        o_space_low = self.env_params["o_space"]["low"][i] 
        o_space_high = self.env_params["o_space"]["high"][i] 

        x_normalized = (x[i] - o_space_low) / (o_space_high - o_space_low)
        setpoint_normalized = (SP - o_space_low) / (o_space_high - o_space_low)

        r_scale = self.env_params.get("r_scale", {})

        cost += (np.sum(x_normalized - setpoint_normalized[self.t]) ** 2) * r_scale.get(k, 1)

        Sp_i += 1
    u_normalized = (u - self.env_params["a_space"]["low"]) / (
        self.env_params["a_space"]["high"] - self.env_params["a_space"]["low"]
    )

    # Add the control cost
    cost += R * u_normalized**2
    r = -cost
    return r

SP = {
      'H_R':[20 for i in range(int(nsteps/2))] + [22 for i in range(int(nsteps/2))],
      'H_M':[20 for i in range(int(nsteps/2))] + [20 for i in range(int(nsteps/2))],
      'H_B':[20 for i in range(int(nsteps/2))] + [20 for i in range(int(nsteps/2))],
  }

action_space = {
    'low': np.array([0.667,8,8,0.67,8]),
    'high':np.array([1,40,40,3,40])
}

observation_space = {
    'low' : np.array([10, 0, 0, 0, 10, 0, 0, 0, 10, 0, 0.75, 0, 15, 15, 15]),
    'high' : np.array([30, 1, 0.15, 0.02, 30, 1, 0.15, 0.02, 30, 0.15, 1, 0.15, 25, 25, 25])  
}



env_params_RSR = {
    'N': nsteps,
    'tsim':T,
    'SP':SP,
    'o_space' : observation_space,
    'a_space' : action_space,
    'x0': np.array([20, 0.8861, 0.1082, 0.0058, 20, 0.8861, 0.1082, 0.0058, 20, 0.1139, 0.7779, 0.1082, 20,20,20]),
    'model': 'RSR', 
    'normalise_a': True, #Normalise the actions
    'normalise_o':True, #Normalise the states,
    'noise':False, #Add noise to the states
    'integration_method': 'casadi'
}
env = make_env(env_params_RSR)


# Global timesteps
nsteps_train = 1e4
training_reps = 1
for r_i in range(training_reps):
    print(f'Training repition: {r_i+1}')
    # Train SAC 
    print('Training using SAC...')
    log_file = f"learning_curves\SAC_RSR_LC_rep_{r_i}.csv"
    SAC_RSR =  SAC("MlpPolicy", env, verbose=1, learning_rate=0.01)
    callback = LearningCurveCallback(log_file=log_file)
    SAC_RSR.learn(nsteps_train,callback=callback)

    # Save SAC Policy 
    SAC_RSR.save(f'policies\SAC_RSR_rep_{r_i}.zip')

    # Train PPO 
    # print('Training using PPO...')
    # log_file = f"learning_curves\PPO_RSR_LC_rep_{r_i}.csv"
    # PPO_RSR =  PPO("MlpPolicy", env, verbose=1, learning_rate=0.001)
    # callback = LearningCurveCallback(log_file=log_file)
    # PPO_RSR.learn(nsteps_train,callback=callback)

    # # Save PPO Policy 
    # PPO_RSR.save(f'policies\PPO_RSR_rep_{r_i}.zip')

    # Train DDPG
    # print('Training using DDPG...')
    # log_file = f'learning_curves\DDPG_RSR_LC_rep_{r_i}.csv'
    # DDPG_RSR =  DDPG("MlpPolicy", env, verbose=1, learning_rate=0.001)
    # callback = LearningCurveCallback(log_file=log_file)
    # DDPG_RSR.learn(nsteps_train,callback=callback)

    # # Save DDPG Policy 
    # DDPG_RSR.save(f'policies\DDPG_RSR_rep_{r_i}.zip')