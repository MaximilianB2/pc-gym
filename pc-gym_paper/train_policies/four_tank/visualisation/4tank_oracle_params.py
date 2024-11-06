import sys
import os
# Get the path two directories up
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)


sys.path.append(grandparent_dir)
from oracle_params import OptimizationStudy


sys.path.append("..\..\..\..\src\pcgym") # Add local pc-gym files to path.
from pcgym import make_env
from stable_baselines3 import PPO, DDPG, SAC
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from multiprocessing import freeze_support
# Define environment
T = 600
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
    try:
        return r[0]
    except Exception:
        return r

SP = {
        'h1': [0.125 for i in range(int(nsteps/2))] + [0.125 for i in range(int(nsteps/2))],
        'h2': [0.25 for i in range(int(nsteps/2))] + [0.25 for i in range(int(nsteps/2))],
        'h3': [0.6 for i in range(int(nsteps/2))] + [0.1 for i in range(int(nsteps/2))],
        'h4': [0.2 for i in range(int(nsteps/2))] + [0.3 for i in range(int(nsteps/2))],
    }

action_space = {
    'low': np.array([0,0]),
    'high':np.array([10,10])
}

observation_space = {
    'low' : np.array([0,]*8),
    'high' : np.array([0.5]*8)  
}


env_params_4tank = {
    'N': nsteps,
    'tsim':T,
    'SP':SP,
    'o_space' : observation_space,
    'a_space' : action_space,
    'dt': 15,
    'x0': np.array([0.141, 0.112, 0.072, 0.42,SP['h1'][0],SP['h2'][0],SP['h3'][0],SP['h4'][0]]),
    'model': 'four_tank', 
    'normalise_a': True, #Normalise the actions
    'normalise_o':True, #Normalise the states,
    'noise':True, #Add noise to the states
    'noise_percentage':0.001,
    'custom_reward': oracle_reward,
    'integration_method': 'casadi'
}
env = make_env(env_params_4tank)

if __name__ == '__main__':
    freeze_support()

    # Your environment setup
    env = make_env(env_params_4tank)
    bounds = np.array([[5,40],[1e-5,1e-2]])
    optimization = OptimizationStudy(env, '..//policies/SAC_4tank_rep_0',)
    optimization.run_optimization(300)
    optimization.print_results()
    optimization.visualize_results()
    best_params = optimization.get_best_params()