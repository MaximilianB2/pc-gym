import sys
sys.path.append("..")  # Adds higher directory to python modules path for callback class. 
sys.path.append("..\..\..\src\pcgym") # Add local pc-gym files to path.

from pcgym import make_env
from callback import LearningCurveCallback
import numpy as np
from stable_baselines3 import PPO, DDPG, SAC

# Define environment
T = 1000
nsteps = 60


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
    try:
        return r[0]
    except Exception:
        return r

SP = {
        'h3': [0.5 for i in range(int(nsteps/2))] + [0.1 for i in range(int(nsteps/2))],
        'h4': [0.2 for i in range(int(nsteps/2))] + [0.3 for i in range(int(nsteps/2))],
    }

action_space = {
    'low': np.array([0,0]),
    'high':np.array([10,10])
}

observation_space = {
    'low' : np.array([0,]*6),
    'high' : np.array([0.5]*6)  
}


env_params_4tank = {
    'N': nsteps,
    'tsim':T,
    'SP':SP,
    'o_space' : observation_space,
    'a_space' : action_space,
    'x0': np.array([0.141, 0.112, 0.072, 0.42,SP['h3'][0],SP['h4'][0]]),
    'model': 'four_tank', 
    'normalise_a': True, #Normalise the actions
    'normalise_o':True, #Normalise the states,
    'noise':True, #Add noise to the states
    'noise_percentage':0.001,
    'custom_reward': oracle_reward,
    'integration_method': 'casadi'
}
env = make_env(env_params_4tank)




# Global timesteps
nsteps_train = 1e4
training_reps = 3
for r_i in range(training_reps):
    print(f'Training repition: {r_i+1}')
    # Train SAC 
    print('Training using SAC...')
    log_file = f"learning_curves\SAC_4tank_LC_rep_{r_i}.csv"
    SAC_4tank =  SAC("MlpPolicy", env, verbose=1, learning_rate=0.01)
    callback = LearningCurveCallback(log_file=log_file)
    SAC_4tank.learn(nsteps_train,callback=callback)

    # Save SAC Policy 
    SAC_4tank.save(f'policies\SAC_4tank_rep_{r_i}.zip')

    # # # Train PPO 
    # print('Training using PPO...')
    # log_file = f"learning_curves\PPO_4tank_LC_rep_{r_i}.csv"
    # PPO_4tank =  PPO("MlpPolicy", env, verbose=1, learning_rate=0.001)
    # callback = LearningCurveCallback(log_file=log_file)
    # PPO_4tank.learn(nsteps_train,callback=callback)

    # # # Save PPO Policy 
    # PPO_4tank.save(f'policies\PPO_4tank_rep_{r_i}.zip')

    # # # Train DDPG
    # print('Training using DDPG...')
    # log_file = f'learning_curves\DDPG_4tank_LC_rep_{r_i}.csv'
    # DDPG_4tank =  DDPG("MlpPolicy", env, verbose=1, learning_rate=0.001)
    # callback = LearningCurveCallback(log_file=log_file)
    # DDPG_4tank.learn(nsteps_train,callback=callback)

    # # Save DDPG Policy 
    # DDPG_4tank.save(f'policies\DDPG_4tank_rep_{r_i}.zip')