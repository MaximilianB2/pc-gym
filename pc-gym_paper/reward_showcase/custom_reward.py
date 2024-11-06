import sys
import os
sys.path.append("../train_policies")  # Adds higher directory to python modules path for callback class. 
sys.path.append("..\..\src\pcgym") # Add local pc-gym files to path.

from pcgym import make_env
from callback import LearningCurveCallback
import numpy as np
from stable_baselines3 import PPO, DDPG, SAC
from stable_baselines3.common.callbacks import BaseCallback

class SaveModelCallback(BaseCallback):
    def __init__(self, check_freq: int, save_path: str, name_prefix: str):
        super(SaveModelCallback, self).__init__()
        self.check_freq = check_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.steps_since_last_save = 0

    def _on_step(self) -> bool:
        self.steps_since_last_save += 1
        if self.steps_since_last_save >= self.check_freq:
            self.steps_since_last_save = 0  # reset counter
            # Do the saving
            save_file = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}")
            self.model.save(save_file)
            print(f"Saved model to {save_file}")
        return True
T = 26
nsteps = 120
training_seed = 1
SP = {
    'Ca': [0.85 for i in range(int(nsteps/3))] + [0.9 for i in range(int(nsteps/3))]+ [0.87 for i in range(int(nsteps/3))],
}

action_space = {
    'low': np.array([295]),
    'high':np.array([302]) 
}

observation_space = {
    'low' : np.array([0.7,300,0.8]),
    'high' : np.array([1,350,0.9])  
}

r_scale = {'Ca':1e3}

def r_squared(self,x,u,con):
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

def r_sparse(self,x,u,con):
    Sp_i = 0
    r = 0

    for k in self.env_params["SP"]:
      e = 0
      i = self.model.info()["states"].index(k)
      SP = self.SP[k]
    
      o_space_low = self.env_params["o_space"]["low"][i] 
      o_space_high = self.env_params["o_space"]["high"][i] 

      x_normalized = (x[i] - o_space_low) / (o_space_high - o_space_low)
      setpoint_normalized = (SP - o_space_low) / (o_space_high - o_space_low)

      e += (np.sum(x_normalized - setpoint_normalized[self.t]) ** 2) 

      Sp_i += 1

      if e < 0.003:
        r += 1 

      
    return r

def r_abs(self,x,u,con):
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

        cost += (np.sum(np.abs(x_normalized - setpoint_normalized[self.t]))) * r_scale.get(k, 1)

        Sp_i += 1
    u_normalized = (u - self.env_params["a_space"]["low"]) / (
        self.env_params["a_space"]["high"] - self.env_params["a_space"]["low"]
    )

    # Add the control cost
    cost += R * u_normalized**2
    r = -cost
    
    return r
r_func = [r_squared,r_sparse, r_abs]
# r_func = [r_sparse]
for r_func_i in r_func:
    # Define environment
    # print(r_func_i.__name__)
    
    env_params = {
        'N': nsteps, 
        'tsim':T, 
        'SP':SP, 
        'o_space' : observation_space, 
        'a_space' : action_space,
        'x0': np.array([0.8,330,0.8]),
        'r_scale': r_scale,
        'model': 'cstr', 
        'normalise_a': True, 
        'normalise_o':True, 
        'noise':True, 
        'integration_method': 'casadi', 
        'noise_percentage':0.001, 
        'custom_reward': r_func_i # loop through the custom reward functions
    }

    env = make_env(env_params)
    
    # Global timesteps
    nsteps_train = 1e4

    # # Train SAC 
    # log_file = "learning_curves\SAC_CSTR_LC_"+r_func_i.__name__+".csv"
    # SAC_CSTR =  SAC("MlpPolicy", env, verbose=1, learning_rate=0.01)
    # callback = LearningCurveCallback(log_file=log_file)
    # SAC_CSTR.learn(nsteps_train,callback=callback)

    # # Save SAC Policy 
    # SAC_CSTR.save('policies\SAC_CSTR_'+r_func_i.__name__+'.zip')

    # # Train PPO 
    # log_file = 'learning_curves\PPO_CSTR_LC_'+r_func_i.__name__+'.csv'
    # PPO_CSTR =  PPO("MlpPolicy", env, verbose=1, learning_rate=0.001)
    # callback = LearningCurveCallback(log_file=log_file)
    # PPO_CSTR.learn(nsteps_train*3,callback=callback)

    # # Save SAC Policy 
    # PPO_CSTR.save('policies\PPO_CSTR_'+r_func_i.__name__+'.zip')

    # Train DDPG

    log_file = "learning_curves\DDPG_CSTR_LC_"+r_func_i.__name__+".csv"
    save_path = 'policies_inter'
    name_prefix = f'DDPG_CSTR_{r_func_i.__name__}'
    DDPG_CSTR =  DDPG("MlpPolicy", env, verbose=1, learning_rate=0.001, seed = training_seed )
    callback = SaveModelCallback(check_freq=100, save_path=save_path, name_prefix=name_prefix)
    DDPG_CSTR.learn(nsteps_train*3,callback=callback)
    # log_file = "learning_curves\DDPG_CSTR_LC_"+r_func_i.__name__+".csv"
    # DDPG_CSTR =  DDPG("MlpPolicy", env, verbose=1, learning_rate=0.001)
    # callback = LearningCurveCallback(log_file=log_file)

    # DDPG_CSTR.learn(nsteps_train*3,callback=callback)

    # # Save DDPG Policy 
    # DDPG_CSTR.save('policies\DDPG_CSTR'+r_func_i.__name__+'.zip')