
import sys
sys.path.append("..\..\..\src\pcgym") # Add local pc-gym files to path.
from pcgym import make_env
from stable_baselines3 import PPO, DDPG, SAC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define environment
T = 26
nsteps = 120
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

r_scale = {'Ca':1e4}

# Define reward to be equal to the OCP (i.e the same as the oracle)
def oracle_reward(self,x,u,con):
    Sp_i = 0
    cost = 0 
    R = 0.001
    for k in self.env_params["SP"]:
        i = self.model.info()["states"].index(k)
        SP = self.SP[k]
     
        o_space_low = self.env_params["o_space"]["low"][i] 
        o_space_high = self.env_params["o_space"]["high"][i] 

        x_normalized = (x[i] - o_space_low) / (o_space_high - o_space_low)
        setpoint_normalized = (SP - o_space_low) / (o_space_high - o_space_low)

        

        cost += (np.sum(np.abs(x_normalized - setpoint_normalized[self.t]))) # Analyse with IAE otherwise too much bias for small errors 
         
        Sp_i += 1
    u_normalized = (u - self.env_params["a_space"]["low"]) / (
        self.env_params["a_space"]["high"] - self.env_params["a_space"]["low"]
    )

    # Add the control cost
    cost += R * u_normalized**2
    r = -cost
    return r

env_params = {
    'N': nsteps, 
    'tsim':T, 
    'SP':SP, 
    'o_space' : observation_space, 
    'a_space' : action_space,
    'x0': np.array([0.85,330,0.8]),
    'r_scale': r_scale,
    'model': 'cstr', 
    'normalise_a': True, 
    'normalise_o':True, 
    'noise':True, 
    'integration_method': 'casadi', 
    'noise_percentage':0.001, 
    'custom_reward': oracle_reward
}

env = make_env(env_params)

# Load trained policies
SAC_cstr = SAC.load('./policies/SAC_CSTR')
PPO_cstr = PPO.load('./policies/PPO_CSTR')
DDPG_cstr = DDPG.load('./policies/DDPG_CSTR')

# Visualise policies with the oracle
evaluator, data = env.plot_rollout({'SAC':SAC_cstr,'PPO':PPO_cstr,'DDPG':DDPG_cstr}, reps=3, oracle=True, MPC_params={'N':20,'R':0.001},save_fig=True)


# Visualise the learning curves
SAC_lc = pd.read_csv('./learning_curves/SAC_CSTR_LC.csv')
DDPG_lc = pd.read_csv('./learning_curves/DDPG_CSTR_LC.csv')
PPO_lc = pd.read_csv('./learning_curves/PPO_CSTR_LC.csv')

# Calculate the rolling mean and standard deviation
window_size = 400
SAC_lc['Reward_mean'] = SAC_lc['Reward'].rolling(window_size).mean()
SAC_lc['Reward_std'] = SAC_lc['Reward'].rolling(window_size).std()

DDPG_lc['Reward_mean'] = DDPG_lc['Reward'].rolling(window_size).mean()
DDPG_lc['Reward_std'] = DDPG_lc['Reward'].rolling(window_size).std()

PPO_lc['Reward_mean'] = PPO_lc['Reward'].rolling(window_size).mean()
PPO_lc['Reward_std'] = PPO_lc['Reward'].rolling(window_size).std()

episode_min = min(SAC_lc['Episode'].min(), DDPG_lc['Episode'].min(), PPO_lc['Episode'].min())
episode_max = max(SAC_lc['Episode'].max(), DDPG_lc['Episode'].max(), PPO_lc['Episode'].max())
# Plot the data with standard deviation
plt.figure()
plt.plot(SAC_lc['Episode'], SAC_lc['Reward_mean'], color = 'tab:red', label = 'SAC')
plt.fill_between(SAC_lc['Episode'], SAC_lc['Reward_mean'] - SAC_lc['Reward_std'], SAC_lc['Reward_mean'] + SAC_lc['Reward_std'], color='tab:red',edgecolor ='None', alpha=0.2)

plt.plot(DDPG_lc['Episode'], DDPG_lc['Reward_mean'], color = 'tab:olive', label = 'DDPG')
plt.fill_between(DDPG_lc['Episode'], DDPG_lc['Reward_mean'] - DDPG_lc['Reward_std'], DDPG_lc['Reward_mean'] + DDPG_lc['Reward_std'], color='tab:olive', edgecolor ='None', alpha=0.2)

plt.plot(PPO_lc['Episode'], PPO_lc['Reward_mean'],color = 'tab:purple', label = 'PPO')
plt.fill_between(PPO_lc['Episode'], PPO_lc['Reward_mean'] - PPO_lc['Reward_std'], PPO_lc['Reward_mean'] + PPO_lc['Reward_std'], color='tab:purple', edgecolor ='None', alpha=0.2)

plt.xlabel('Timestep')
plt.ylabel('Reward')
plt.legend(loc = 'lower right')
plt.grid(True)
plt.xlim(window_size, episode_max)
plt.savefig('cstr_lc.pdf')
plt.show()

# Visualise the optimality gap 

SAC_opt_gap = (data['oracle']['r'][0,:,:] - data['SAC']['r'][0,:,:])
PPO_opt_gap = (data['oracle']['r'][0,:,:] - data['PPO']['r'][0,:,:])
DDPG_opt_gap =(data['oracle']['r'][0,:,:] - data['DDPG']['r'][0,:,:])
x = np.arange(len(SAC_opt_gap))
plt.figure()
plt.plot(np.median(SAC_opt_gap, axis=1), label='SAC', color='tab:red')
plt.fill_between(x, np.min(SAC_opt_gap, axis=1), np.max(SAC_opt_gap, axis=1), color='tab:red', alpha=0.2)

plt.plot(np.median(PPO_opt_gap, axis=1), label='PPO', color='tab:purple')
plt.fill_between(x, np.min(PPO_opt_gap, axis=1), np.max(PPO_opt_gap, axis=1), color='tab:purple', alpha=0.2)

plt.plot(np.median(DDPG_opt_gap, axis=1), label='DDPG', color='tab:olive')
plt.fill_between(x, np.min(DDPG_opt_gap, axis=1), np.max(DDPG_opt_gap, axis=1), color='tab:olive', alpha=0.2)
plt.grid('True')
plt.legend()
plt.xlim(0,26)
plt.ylabel('Optimality gap')
plt.xlabel('Time (min)')
plt.tight_layout
plt.show()