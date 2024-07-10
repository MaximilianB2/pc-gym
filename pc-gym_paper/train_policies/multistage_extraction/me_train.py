import sys
sys.path.append("..")  # Adds higher directory to python modules path for callback class. 
sys.path.append("..\..\..\src\pcgym") # Add local pc-gym files to path.

from pcgym import make_env
from callback import LearningCurveCallback
import numpy as np
from stable_baselines3 import PPO, DDPG, SAC

# Define environment
T = 26
nsteps = 120


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
      'X5': [0.3 for i in range(int(nsteps/4))] + [0.4 for i in range(int(nsteps/2))]+ [0.3 for i in range(int(nsteps/4))],
  }

action_space = {
    'low': np.array([5,10]),
    'high':np.array([500,1000])
}

observation_space = {
    'low' : np.array([0]*10+[0.3]),
    'high' : np.array([1]*10+[0.4])  
}


r_scale = {
    'X5': 1
}

env_params_ms = {
    'N': nsteps,
    'tsim':T,
    'SP':SP,
    'o_space' : observation_space,
    'a_space' : action_space,
    'dt': 1,
    'x0': np.array([0.55, 0.3, 0.45, 0.25, 0.4, 0.20, 0.35, 0.15, 0.25, 0.1,0.3]),
    'model': 'multistage_extraction', 
    'r_scale': r_scale,
    'normalise_a': True, #Normalise the actions
    'normalise_o':True, #Normalise the states,
    'noise':True, #Add noise to the states
    'noise_percentage':0.001,
    'integration_method': 'casadi'
}
env = make_env(env_params_ms)


# Global timesteps
nsteps_train = 1e5
training_reps = 3
for r_i in range(training_reps):
    print(f'Training repition: {r_i+1}')
    # Train SAC 
    # print('Training using SAC...')
    # log_file = f"learning_curves\SAC_ME_LC_rep_{r_i}.csv"
    # SAC_ME =  SAC("MlpPolicy", env, verbose=1, learning_rate=0.01)
    # callback = LearningCurveCallback(log_file=log_file)
    # SAC_ME.learn(nsteps_train,callback=callback)

    # # Save SAC Policy 
    # SAC_ME.save(f'policies\SAC_ME_rep_{r_i}.zip')

    # Train PPO 
    print('Training using PPO...')
    log_file = f"learning_curves\PPO_ME_LC_rep_{r_i}.csv"
    PPO_ME =  PPO("MlpPolicy", env, verbose=1, learning_rate=0.001)
    callback = LearningCurveCallback(log_file=log_file)
    PPO_ME.learn(nsteps_train,callback=callback)

    # Save PPO Policy 
    PPO_ME.save(f'policies\PPO_ME_rep_{r_i}.zip')

    # Train DDPG
    # print('Training using DDPG...')
    # log_file = f'learning_curves\DDPG_ME_LC_rep_{r_i}.csv'
    # DDPG_ME =  DDPG("MlpPolicy", env, verbose=1, learning_rate=0.001)
    # callback = LearningCurveCallback(log_file=log_file)
    # DDPG_ME.learn(nsteps_train,callback=callback)

    # Save DDPG Policy 
    # DDPG_ME.save(f'policies\DDPG_ME_rep_{r_i}.zip')