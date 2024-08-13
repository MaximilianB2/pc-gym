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
T = 200
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
        'S2_A':[10 for i in range(int(nsteps/2))] + [15 for i in range(int(nsteps/2))]
    }

action_space = {
    'low': np.array([0, 1, 0.05, 0.05, 0.05]),
    'high':np.array([10, 30, 1, 1, 1])
}

observation_space = {
    'low' : np.array([0,0,0,0]*4+[0]),
    'high' : np.array([10,10,10,500]*4+[20])  
}



env_params_bio = {
    'N': nsteps,
    'tsim':T,
    'SP':SP,
    'o_space' : observation_space,
    'a_space' : action_space,
    'x0': np.array([10.,5.,5.,300.]*4+[10]),
    'model': 'biofilm_reactor', 
    'normalise_a': True, #Normalise the actions
    'normalise_o':True, #Normalise the states,
    'noise':True, #Add noise to the states
    'noise_percentage':0.001,
    'integration_method': 'casadi'
}
env = make_env(env_params_bio)


# Load trained policies
SAC_bio = SAC.load('./policies/SAC_biofilm_rep_0')
# PPO_RSR = PPO.load('./policies/PPO_RSR')
# DDPG_RSR = DDPG.load('./policies/DDPG_RSR')

# Visualise policies with the oracle
# 'PPO':PPO_RSR,'DDPG':DDPG_RSR
evaluator, data = env.get_rollouts({'SAC':SAC_bio,}, reps=1, oracle=True, MPC_params={'N':10,'R':0})
np.save('data.npy', data)
data = np.load('data.npy', allow_pickle=True).item()

def paper_plot(data):
    # Set up LaTeX rendering
    rcParams['text.usetex'] = True
    rcParams['font.family'] = 'serif'
    rcParams['axes.labelsize'] = 10
    rcParams['xtick.labelsize'] = 10
    rcParams['ytick.labelsize'] = 10
    rcParams['legend.fontsize'] = 10

    t = np.linspace(0, T, nsteps)
    
    # A4 width in inches
    a4_width_inches = 8.27
    
    # Calculate height to maintain aspect ratio
    height = a4_width_inches * 0.4  # Adjusted for more subplots
    
    fig, axs = plt.subplots(2, 3, figsize=(a4_width_inches, height))
    plt.subplots_adjust(wspace=0.5, hspace=0.4, top=0.85, bottom=0.1, left=0.08, right=0.98)
    policies = ['oracle','SAC']
    cols = ['tab:orange', 'tab:red', 'tab:blue', 'tab:green', ]
    labels = ['Oracle','SAC']

    # Create lines for the legend
    lines = []

    for i, policy in enumerate(policies):
        line, = axs[0,0].plot([], [], color=cols[i], label=labels[i])
        lines.append(line)
    ref_line, = axs[0,0].plot([], [], color='black', linestyle='--',  label='Reference')
    lines.append(ref_line)

    # Create legend above the plots
    fig.legend(handles=lines, loc='upper center', bbox_to_anchor=(0.5, 0.98),
                ncol=5, frameon=False, columnspacing=1)

    y_labels = [r'$S_A$ [kg/m$^3$]']
    u_labels = [r'F', r'Fr', r'S1_F', r'S2_F', r'S3_F']
    

    ax = axs[0,0]
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    for i, policy in enumerate(policies):

        ax.plot(t, np.median(data[policy]['x'][13,:,:], axis=1), color=cols[i], linewidth=1.25)
        if policy == 'SAC':
                ax.step(t, np.median(data[policy]['x'][16,:,:], axis=1), color='black', linestyle = '--') 
        
        ax.fill_between(t, np.max(data[policy]['x'][0,:,:], axis=1), 
                        np.min(data[policy]['x'][0,:,:], axis=1), 
                        alpha=0.2, linewidth=0, color=cols[i])
        
    ax.set_ylabel(y_labels[0])
    ax.set_xlabel(r'Time (hr)')
    ax.set_xlim(0, T)
        

    # Plot for 2 controls
    for idx in range(5):
        if idx < 2:
            ax = axs[0,idx+1]
        else:
            ax = axs[1,idx-2]
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        
        for i, policy in enumerate(policies):
            ax.step(t, np.median(data[policy]['u'][idx,:,:], axis=1), color=cols[i], where='post', linewidth=1.25)
            ax.fill_between(t, np.max(data[policy]['u'][idx,:,:], axis=1), 
                            np.min(data[policy]['u'][idx,:,:], axis=1),
                            step="post", alpha=0.2, linewidth=0, color=cols[i])
        
        ax.set_ylabel(u_labels[idx])
        ax.set_xlabel(r'Time (hr)')
        ax.set_xlim(0, T)

    # Histogram plot
    # ax = axs[1, 2]
    # all_rewards = np.concatenate([data[policy]["r"].sum(axis=1).flatten() for policy in policies])
    # min_reward, max_reward = np.min(all_rewards), np.max(all_rewards)
    # bins = np.linspace(min_reward, max_reward, 11)

    # for i, policy in enumerate(policies):
    #     ax.hist(
    #         data[policy]["r"].sum(axis=1).flatten(),
    #         bins=bins,
    #         color=cols[i],
    #         alpha=0.5,
    #         label=labels[i],
    #         edgecolor='None',
    #     )

    # ax.set_ylabel('Frequency')
    # ax.set_xlabel('Cumulative Reward')
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol=2, frameon=False)

    # Adjust the plots to be square and the same size
    for ax in axs.flatten():
        ax.set_box_aspect(1)
    
    plt.savefig('biofilm_vis.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.show()

paper_plot(data)
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


# %%
