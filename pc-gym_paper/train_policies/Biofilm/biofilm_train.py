import sys
sys.path.append("..")  # Adds higher directory to python modules path for callback class. 
sys.path.append("..\..\..\src\pcgym") # Add local pc-gym files to path.

from pcgym import make_env
from callback import LearningCurveCallback
import numpy as np
from stable_baselines3 import PPO, DDPG, SAC

# Define environment
T = 100
nsteps = 100


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
      'S2_A':[1.5 for i in range(int(nsteps/2))] + [2 for i in range(int(nsteps/2))]
      
  }

action_space = {
    'low': np.array([0, 1, 0.05, 0.05, 0.05]),
    'high':np.array([10, 30, 1, 1, 1])
}

observation_space = {
    'low' : np.array([0,0,0,0]*4+[0.9]),
    'high' : np.array([10,10,10,500]*4+[1.1])  
}



env_params_RSR = {
    'N': nsteps,
    'tsim':T,
    'SP':SP,
    'o_space' : observation_space,
    'a_space' : action_space,
    'x0': np.array([2,0.1,10,0.1]*4+[1]),
    'model': 'biofilm_reactor', 
    'normalise_a': True, #Normalise the actions
    'normalise_o':True, #Normalise the states,
    'noise':False, #Add noise to the states
    'integration_method': 'casadi'
}
env = make_env(env_params_RSR)


# Global timesteps
nsteps_train = 5e3
training_reps = 1
for r_i in range(training_reps):
    print(f'Training repition: {r_i+1}')
    # Train SAC 
    print('Training using SAC...')
    log_file = f"learning_curves\SAC_biofilm_LC_rep_{r_i}.csv"
    SAC_biofilm =  SAC("MlpPolicy", env, verbose=1, learning_rate=0.01)
    callback = LearningCurveCallback(log_file=log_file)
    SAC_biofilm.learn(nsteps_train,callback=callback)

    # Save SAC Policy 
    SAC_biofilm.save(f'policies\SAC_biofilm_rep_{r_i}.zip')
    env.plot_rollout({'SAC':SAC_biofilm},oracle=False, reps=1)
    # Train PPO 
    # print('Training using PPO...')
    # log_file = f"learning_curves\PPO_biofilm_LC_rep_{r_i}.csv"
    # PPO_biofilm =  PPO("MlpPolicy", env, verbose=1, learning_rate=0.001)
    # callback = LearningCurveCallback(log_file=log_file)
    # PPO_biofilm.learn(nsteps_train,callback=callback)

    # # Save PPO Policy 
    # PPO_biofilm.save(f'policies\PPO_biofilm_rep_{r_i}.zip')

    # Train DDPG
    # print('Training using DDPG...')
    # log_file = f'learning_curves\DDPG_biofilm_LC_rep_{r_i}.csv'
    # DDPG_biofilm =  DDPG("MlpPolicy", env, verbose=1, learning_rate=0.001)
    # callback = LearningCurveCallback(log_file=log_file)
    # DDPG_biofilm.learn(nsteps_train,callback=callback)

    # # Save DDPG Policy 
    # DDPG_biofilm.save(f'policies\DDPG_biofilm_rep_{r_i}.zip')