import sys
sys.path.append("..\..\..\src\pcgym") # Add local pc-gym files to path.
import numpy as np
from stable_baselines3 import SAC
import gymnasium as gym
import pickle
from pcgym import make_env


# Define environment
T = 26
nsteps = 120

SP = {
    'Ca': [0.85 for i in range(int(nsteps))],
}

action_space = {
    'low': np.array([295]),
    'high':np.array([310]) 
}

observation_space = {
    'low' : np.array([0.7,300,0.8]),
    'high' : np.array([1,350,0.9])  
}

r_scale = {'Ca':1e3}


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
disturbance_space = {
    'low': np.array([0.9]),
    'high': np.array([1.5])
}

env_params_temp = {
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
    'noise_percentage':0.0025, 
    'custom_reward': oracle_reward,
    'disturbance_bounds':disturbance_space
}