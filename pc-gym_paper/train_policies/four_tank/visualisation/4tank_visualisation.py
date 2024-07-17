
import sys
sys.path.append("..\..\..\..\src\pcgym") # Add local pc-gym files to path.
from pcgym import make_env
from stable_baselines3 import PPO, DDPG, SAC
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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

# Load trained policies
SAC_4tank = SAC.load('..//policies/SAC_4tank_rep_0')
# PPO_4tank = PPO.load('./policies/PPO_4tank')
# DDPG_4tank = DDPG.load('./policies/DDPG_4tank')

# Visualise policies with the oracle
# 'PPO':PPO_4tank,'DDPG':DDPG_4tank
evaluator, data = env.plot_rollout({'SAC':SAC_4tank,}, reps=1, oracle=True, MPC_params={'N':10,'R':0.01},save_fig=True)


# # Visualise the learning curves
# reps = 3
# algorithms = ['SAC', 'DDPG', 'PPO']
# data = {alg: [] for alg in algorithms}

# # Load data
# for r_i in range(1,reps):
#     for alg in algorithms:
#         lc = pd.read_csv(f'./learning_curves/{alg}_CSTR_LC_rep_{r_i}.csv')
#         data[alg].append(lc['Reward'])  # Assuming 'reward' is the column name

# # Combine and calculate median reward
# median_rewards = {}
# min_rewards = {}
# max_rewards = {}
# std_rewards = {}
# for alg in algorithms:
#     combined_rewards = pd.concat(data[alg], axis=1)
#     median_rewards[alg] = combined_rewards.median(axis=1)
#     min_rewards[alg] = combined_rewards.min(axis=1)
#     max_rewards[alg] = combined_rewards.max(axis=1)
#     std_rewards[alg] = combined_rewards.std(axis=1)
# window_size = 1000

# # Calculate rolling mean for each algorithm
# rolling_means = {}
# rolling_stds = {}
# for alg, rewards in median_rewards.items():
#     rolling_means[alg] = rewards.rolling(window=window_size).mean()
#     rolling_stds[alg] = std_rewards[alg].rolling(window=window_size).mean()
# # Plotting
# plt.figure(figsize=(10, 6))

# colors = {'SAC': 'tab:red', 'DDPG': 'tab:olive', 'PPO': 'tab:blue'}
# for alg, rolling_mean in rolling_means.items():
#     plt.plot(rolling_mean.index, rolling_mean, label=alg, color=colors[alg])
#     rolling_min = min_rewards[alg].rolling(window=window_size).mean()  
#     rolling_max = max_rewards[alg].rolling(window=window_size).mean()
#     upper_bound = rolling_mean + rolling_stds[alg]
#     lower_bound = rolling_mean - rolling_stds[alg]
    
#     plt.fill_between(rolling_mean.index, lower_bound, upper_bound, color=colors[alg], alpha=0.1)
#     # plt.fill_between(rolling_min.index, rolling_min, rolling_max, color=colors[alg], alpha=0.1)

# plt.xlabel('Timestep')
# plt.ylabel('Reward')
# plt.legend(loc = 'lower right')
# plt.xlim(1000,30000)
# plt.grid(True)
# plt.tight_layout()
# plt.show()
# # # Calculate the rolling mean and standard deviation
# # window_size = 400
# # SAC_lc['Reward_mean'] = SAC_lc['Reward'].rolling(window_size).mean()
# # SAC_lc['Reward_std'] = SAC_lc['Reward'].rolling(window_size).std()

# # DDPG_lc['Reward_mean'] = DDPG_lc['Reward'].rolling(window_size).mean()
# # DDPG_lc['Reward_std'] = DDPG_lc['Reward'].rolling(window_size).std()

# # PPO_lc['Reward_mean'] = PPO_lc['Reward'].rolling(window_size).mean()
# # PPO_lc['Reward_std'] = PPO_lc['Reward'].rolling(window_size).std()

# # episode_min = min(SAC_lc['Episode'].min(), DDPG_lc['Episode'].min(), PPO_lc['Episode'].min())
# # episode_max = max(SAC_lc['Episode'].max(), DDPG_lc['Episode'].max(), PPO_lc['Episode'].max())
# # # Plot the data with standard deviation
# # plt.figure()
# # plt.plot(SAC_lc['Episode'], SAC_lc['Reward_mean'], color = 'tab:red', label = 'SAC')
# # plt.fill_between(SAC_lc['Episode'], SAC_lc['Reward_mean'] - SAC_lc['Reward_std'], SAC_lc['Reward_mean'] + SAC_lc['Reward_std'], color='tab:red',edgecolor ='None', alpha=0.2)

# # plt.plot(DDPG_lc['Episode'], DDPG_lc['Reward_mean'], color = 'tab:olive', label = 'DDPG')
# # plt.fill_between(DDPG_lc['Episode'], DDPG_lc['Reward_mean'] - DDPG_lc['Reward_std'], DDPG_lc['Reward_mean'] + DDPG_lc['Reward_std'], color='tab:olive', edgecolor ='None', alpha=0.2)

# # plt.plot(PPO_lc['Episode'], PPO_lc['Reward_mean'],color = 'tab:purple', label = 'PPO')
# # plt.fill_between(PPO_lc['Episode'], PPO_lc['Reward_mean'] - PPO_lc['Reward_std'], PPO_lc['Reward_mean'] + PPO_lc['Reward_std'], color='tab:purple', edgecolor ='None', alpha=0.2)

# # plt.xlabel('Timestep')
# # plt.ylabel('Reward')
# # plt.legend(loc = 'lower right')
# # plt.grid(True)
# # plt.xlim(window_size, episode_max)
# # plt.savefig('cstr_lc.pdf')
# # plt.show()

