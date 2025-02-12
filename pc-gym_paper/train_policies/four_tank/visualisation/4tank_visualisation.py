
import sys
sys.path.append("..\..\..\..\src\pcgym") # Add local pc-gym files to path.
from pcgym import make_env
from stable_baselines3 import PPO, DDPG, SAC
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Define environment
T = 1000
nsteps = 60


# Define reward to be equal to the OCP (i.e the same as the oracle)
def oracle_reward(self,x,u,con):
    Sp_i = 0
    cost = 0 
    R = 0
    if not hasattr(self, 'u_prev'):
        self.u_prev = u

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
    u_prev_norm =  (self.u_prev - self.env_params["a_space"]["low"]) / (
        self.env_params["a_space"]["high"] - self.env_params["a_space"]["low"]
    )
    self.u_prev = u

    # Add the control cost
    cost += np.sum(R * (u_normalized-u_prev_norm)**2)
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
    'low': np.array([0.1, 0.1]),
    'high':np.array([10, 10])
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
    'dt': 15,
    'x0': np.array([0.141, 0.112, 0.072, 0.42,SP['h3'][0],SP['h4'][0]]),
    'model': 'four_tank', 
    'normalise_a': True, #Normalise the actions
    'normalise_o':True, #Normalise the states,
    'noise':True, #Add noise to the states
    'noise_percentage':0.05,
    'custom_reward': oracle_reward,
    'integration_method': 'casadi'
}
env = make_env(env_params_4tank)

# Load trained policies
SAC_4tank = SAC.load('..//policies/SAC_4tank_rep_0')
PPO_4tank = PPO.load('..//policies/PPO_4tank_rep_0')
DDPG_4tank = DDPG.load('..//policies/DDPG_4tank_rep_0')

# Visualise policies with the oracle
# 'PPO':PPO_4tank,'DDPG':DDPG_4tank
# evaluator, data = env.get_rollouts({'SAC':SAC_4tank,'PPO':PPO_4tank,'DDPG':DDPG_4tank}, reps=50, oracle=True, MPC_params={'N': 19,})
# np.save('data.npy', data)
data = np.load('data.npy', allow_pickle=True).item()


def paper_plot(data):
    # Set up LaTeX rendering
    rcParams['text.usetex'] = True
    rcParams['font.family'] = 'serif'
    rcParams['axes.labelsize'] = 10
    rcParams['xtick.labelsize'] = 10
    rcParams['ytick.labelsize'] = 10
    rcParams['legend.fontsize'] = 10

    t = np.linspace(0, 1000, 60)
    
    # A4 width in inches
    a4_width_inches = 8.27
    
    # Calculate height to maintain aspect ratio
    height = a4_width_inches * 0.8  # Adjusted for more subplots
    
    fig, axs = plt.subplots(1, 4, figsize=(a4_width_inches, height))
    plt.subplots_adjust(wspace=0.4, hspace=0.4, top=0.85, bottom=0.1, left=0.08, right=0.98)

    policies = ['oracle', 'SAC', 'PPO', 'DDPG']
    cols = ['tab:orange', 'tab:red', 'tab:blue', 'tab:green', ]
    labels = ['Oracle','SAC', 'PPO', 'DDPG']


    # Create lines for the legend
    lines = []
    for i, policy in enumerate(policies):
        line, = axs[0].plot([], [], color=cols[i], label=labels[i])
        lines.append(line)
    ref_line, = axs[0].plot([], [], color='black', linestyle='--',  label='Reference')
    lines.append(ref_line)

    # Create legend above the plots
    fig.legend(handles=lines, loc='upper center', bbox_to_anchor=(0.5, 0.65),
                ncol=5, frameon=False, columnspacing=1)

    y_labels = [r'$h_1$ [m]', r'$h_2$ [m]', r'$h_3$ [m]', r'$h_4$ [m]']
    u_labels = [r'$v_1$ [V]', r'$v_2$ [V]']
    
    for idx in range(2):  # Loop for 4 states

        ax = axs[idx]
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        
        for i, policy in enumerate(policies):
            ax.plot(t, np.median(data[policy]['x'][idx+2,:,:], axis=1), color=cols[i], linewidth=1.25)
            if policy == 'SAC' and idx == 0:
                    ax.step(t, np.median(data[policy]['x'][idx+4,:,:], axis=1), color='black', linestyle = '--')
            if policy == 'SAC' and idx == 1:
                ax.step(t, np.median(data[policy]['x'][idx+4,:,:], axis=1), color='black', linestyle = '--') 
            
            ax.fill_between(t, np.max(data[policy]['x'][idx+2,:,:], axis=1), 
                            np.min(data[policy]['x'][idx+2,:,:], axis=1), 
                            alpha=0.2, linewidth=0, color=cols[i])
            
        ax.set_ylabel(y_labels[idx])
        ax.set_xlabel(r'Time [s]')
        ax.set_xlim(0, T)
        

    # Plot for 2 controls
    for idx in range(2):
        ax = axs[idx+2]
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        
        for i, policy in enumerate(policies):
            ax.step(t, np.median(data[policy]['u'][idx,:,:], axis=1), color=cols[i], where='post', linewidth=1.25)
            ax.fill_between(t, np.max(data[policy]['u'][idx,:,:], axis=1), 
                            np.min(data[policy]['u'][idx,:,:], axis=1),
                            step="post", alpha=0.2, linewidth=0, color=cols[i])
        
        ax.set_ylabel(u_labels[idx])
        ax.set_xlabel(r'Time [s]')
        ax.set_xlim(0, T)
    for ax in axs.flatten():
        ax.set_box_aspect(1)
    
    plt.savefig('4tank_vis.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.show()


def paper_r_distribution(data):
    # Set up LaTeX rendering
    rcParams['text.usetex'] = True
    rcParams['font.family'] = 'serif'
    rcParams['axes.labelsize'] = 10
    rcParams['xtick.labelsize'] = 10
    rcParams['ytick.labelsize'] = 10
    rcParams['legend.fontsize'] = 10

    # A4 width in inches
    a4_width_inches = 8.27*0.5
    
    # Calculate height to maintain aspect ratio
    height = a4_width_inches   # Adjust this factor as needed
    
    fig, ax = plt.subplots(1, 1, figsize=(a4_width_inches, height))
    plt.subplots_adjust(wspace=0.3, top=0.85, bottom=0.15, left=0.12, right=0.98)
    policies = ['oracle', 'SAC', 'PPO', 'DDPG']
    cols = ['tab:orange', 'tab:red', 'tab:blue', 'tab:green']
    labels = ['Oracle', 'SAC', 'PPO', 'DDPG']

    # Calculate the global min and max across all datasets
    all_rewards = np.concatenate([data[policy]["r"].sum(axis=1).flatten() for policy in policies])
    global_min, global_max = np.min(all_rewards), np.max(all_rewards)

    # Create a single set of bins for all datasets
    num_bins = 20 # Increase this for more granularity
    bins = np.linspace(global_min, global_max, num_bins)

    for i, policy in enumerate(policies[1:], start=1):
            rewards = data[policy]["r"].sum(axis=1).flatten()
            
            # Plot histogram
            n, _, patches = ax.hist(
                rewards,
                bins=bins,
                color=cols[i],
                alpha=0.5,
                label=labels[i],
                edgecolor='None',
            )

    # Add vertical line for oracle
    oracle_reward = np.median(data['oracle']["r"].sum(axis=1).flatten())
    ax.axvline(x=oracle_reward, color='tab:orange', linestyle='--', linewidth=2, label='Oracle')

    ax.set_ylabel('Frequency')
    ax.set_xlabel('Cumulative Reward')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.18), ncol=2, frameon=False)
    ax.set_box_aspect(1)
    

    plt.savefig('4tank_vis_r_dist.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.show()




oracle_r = np.median(data['oracle']["r"].sum(axis=1).flatten())
policies = ['SAC', 'PPO', 'DDPG']
for policy in policies:
    rewards = data[policy]["r"].sum(axis=1).flatten()
    rewards = np.median(rewards)
    print(policy,oracle_r, rewards)
    normalized_gap = (oracle_r/ nsteps*2 - rewards/ nsteps*2) 


    mad = np.median(np.abs( np.median(data[policy]["r"].sum(axis=1).flatten()) - data[policy]["r"].sum(axis=1).flatten()))
    
    print(f'{policy}:')
    print(f'  Normalized Optimality Gap: {normalized_gap:.4f}')
    print(f'  Median Absolute Deviation (MAD): {mad}')
paper_plot(data)
paper_r_distribution(data)
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

