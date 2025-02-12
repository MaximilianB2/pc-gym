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
T = 30
nsteps = 30

# Define reward to be equal to the OCP (i.e the same as the oracle)
def oracle_reward(self,x,u,con):

    R = 0.01
    SP = self.SP
    if not hasattr(self, 'u_prev'):
        self.u_prev = u

    CV = (x[2]*x[0]/(x[1]**2) - 1)**0.5
    ln = x[1]/x[0]

    o_space_low = self.env_params["o_space"]["low"][[5,6]] 
    o_space_high = self.env_params["o_space"]["high"][[5,6]] 

    CV_normalized = (CV - o_space_low[0]) / (o_space_high[0] - o_space_low[0])
    Ln_normalized = (ln - o_space_low[1]) / (o_space_high[1] - o_space_low[1])
    SP_CV = (SP['CV'][self.t] - o_space_low[0]) / (o_space_high[0] - o_space_low[0])
    SP_Ln = (SP['Ln'][self.t] - o_space_low[1]) / (o_space_high[1] - o_space_low[1])

    r = -1*((SP_CV - CV_normalized)**2 + (SP_Ln - Ln_normalized)**2)


    u_normalized = (u - self.env_params["a_space"]["low"]) / (
        self.env_params["a_space"]["high"] - self.env_params["a_space"]["low"]
    )
    u_prev_norm =  (self.u_prev - self.env_params["a_space"]["low"]) / (
        self.env_params["a_space"]["high"] - self.env_params["a_space"]["low"]
    )

    r -= np.sum(R * (u_normalized-u_prev_norm)**2)
    self.u_prev = u
    print(r)
    
    return r

SP = {
      'CV': [1 for i in range(int(nsteps))],
      'Ln': [15 for i in range(int(nsteps))]
  }

action_space = {
    'low': np.array([-1]),
    'high':np.array([1])
}
action_space_act = {
    'low': np.array([10]),
    'high':np.array([40])
}
lbMu0 = 0  # 0.1
ubMu0 = 1e20
lbMu1 = 0  # 0.1
ubMu1 = 1e20
lbMu2 = 0
ubMu2 = 1e20
lbMu3 = 0
ubMu3 = 1e20
lbC = 0
ubC = 0.5
lbT = 0
ubT = 40
observation_space = {
    'low' : np.array([lbMu0, lbMu1, lbMu2, lbMu3, lbC,0, 0,  0.9, 14]),
    'high' : np.array([ubMu0, ubMu1, ubMu2, ubMu3, ubC,2, 20, 1.1, 16])  
}
CV_0 = np.sqrt(1800863.24079725 * 1478.00986666666/ (22995.8230590611**2) - 1)
Ln_0 =  22995.8230590611 / ( 1478.00986666666 + 1e-6)
print(CV_0, Ln_0)
env_params_cryst = {
    'N': nsteps,
    'tsim':T,
    'SP':SP,
    'o_space' : observation_space,
    'a_space' : action_space,
    'x0': np.array([1478.00986666666, 22995.8230590611, 1800863.24079725, 248516167.940593, 0.15861523304,CV_0, Ln_0 , 1, 15]),
    'model': 'crystallization', 
    'normalise_a': True, #Normalise the actions
    'normalise_o':True, #Normalise the states,
    'noise':True, #Add noise to the states
    'noise_percentage':0.001,
    'integration_method': 'casadi',
    'a_0':39,
    'a_delta':True,
    'a_space_act':action_space_act,
    'custom_reward':oracle_reward
}
env = make_env(env_params_cryst)

SAC_cryst = SAC.load('./policies/SAC_cryst_rep_0')
PPO_cryst = PPO.load('./policies/PPO_cryst_rep_0')
DDPG_cryst = DDPG.load('./policies/DDPG_cryst_rep_0')
evaluator, data = env.get_rollouts({'SAC':SAC_cryst, 'PPO':PPO_cryst,'DDPG':DDPG_cryst}, reps=50, oracle=True, MPC_params={'N':10,})
# np.save('data.npy', data)
# data = np.load('data.npy', allow_pickle=True).item()
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
    
    fig, axs = plt.subplots(1, 4, figsize=(a4_width_inches+0.4, height))
    plt.subplots_adjust(wspace=0.35, hspace=0.4, top=0.85, bottom=0.1, left=0.08, right=0.98)
    policies = ['oracle','SAC' , 'PPO', 'DDPG']
    cols = ['tab:orange', 'tab:red', 'tab:blue', 'tab:green', ]
    labels = ['Oracle','SAC','PPO','DDPG']

    # Create lines for the legend
    lines = []

    for i, policy in enumerate(policies):
        line, = axs[0].plot([], [], color=cols[i], label=labels[i])
        lines.append(line)
    ref_line, = axs[0].plot([], [], color='black', linestyle='--',  label='Reference')
    lines.append(ref_line)

    # Create legend above the plots
    fig.legend(handles=lines, loc='upper center', bbox_to_anchor=(0.4, 0.85),
                ncol=5, frameon=False, columnspacing=1)

    y_labels = [r'$CV$', r'$\bar{L}_n$ [$\mu$m]']
    u_labels = [r'$T_c$ [$^\circ$C]']
    
    for idx in range(2):
      ax = axs[idx]
      ax.grid(True, linestyle='--', alpha=0.7)
      ax.set_axisbelow(True)
      
      for i, policy in enumerate(policies):

          ax.plot(t, np.median(data[policy]['x'][idx+5,:,:], axis=1), color=cols[i], linewidth=1.25)
          if policy == 'SAC':
                  ax.step(t, np.median(data[policy]['x'][idx+7,:,:], axis=1), color='black', linestyle = '--') 
          
          ax.fill_between(t, np.max(data[policy]['x'][idx+5,:,:], axis=1), 
                          np.min(data[policy]['x'][idx+5,:,:], axis=1), 
                          alpha=0.2, linewidth=0, color=cols[i])
          
      ax.set_ylabel(y_labels[idx])
      ax.set_xlabel(r'Time [hr]')
      ax.set_xlim(0, T)
        

    # Plot for 2 controls

    ax = axs[2]
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    for i, policy in enumerate(policies):
        if policy in ['SAC', 'PPO', 'DDPG']:
          u_0 = 39  # Initial u value
          delta_u = data[policy]['u'][0,:,:]  # Assuming this contains delta u values
          actual_u = np.cumsum(delta_u, axis=0) + u_0  # Convert to actual u
          data[policy]['u'][0,:,:] = actual_u  # Replace delta u with actual u in the data structure
          ax.step(t, np.median(data[policy]['u'][0,:,:], axis=1), color=cols[i], where='post', linewidth=1.25)
          ax.fill_between(t, np.max(data[policy]['u'][0,:,:], axis=1), 
                          np.min(data[policy]['u'][0,:,:], axis=1),
                          step="post", alpha=0.2, linewidth=0, color=cols[i])
        else:
          ax.step(t, np.median(data[policy]['u'][0,:,:], axis=1), color=cols[i], where='post', linewidth=1.25)
          ax.fill_between(t, np.max(data[policy]['u'][0,:,:], axis=1), 
                          np.min(data[policy]['u'][0,:,:], axis=1),
                          step="post", alpha=0.2, linewidth=0, color=cols[i])
      
    ax.set_ylabel(u_labels[0])
    ax.set_xlabel(r'Time [hr]')
    ax.set_xlim(0, T)

    # Histogram plot
    ax = axs[3]
    all_rewards = np.concatenate([data[policy]["r"].sum(axis=1).flatten() for policy in policies])
    min_reward, max_reward = np.min(all_rewards), np.max(all_rewards)
    bins = np.linspace(min_reward, max_reward, 25)

    for i, policy in enumerate(policies[1:], start=1):  # Start from 1 to skip oracle
        ax.hist(
            data[policy]["r"].sum(axis=1).flatten(),
            bins=bins,
            color=cols[i],
            alpha=0.5,
            label=labels[i],
            edgecolor='None',
        )

    # Add vertical line for oracle
    oracle_reward = np.median(data['oracle']["r"].sum(axis=1).flatten())
    ax.axvline(x=oracle_reward, color=cols[0], linestyle='--', linewidth=2, label='Oracle')

    ax.set_ylabel('Frequency')
    ax.set_xlabel('Cumulative Reward')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol=2, frameon=False)

    # Adjust the plots to be square and the same size
    for ax in axs.flatten():
        ax.set_box_aspect(1)
    
    plt.savefig('cryst_vis.pdf', bbox_inches='tight', pad_inches=0.2)
    plt.show()
data = np.load('data.npy', allow_pickle=True).item()
oracle_r = np.median(data['oracle']["r"].sum(axis=1).flatten())
policies = ['SAC', 'PPO', 'DDPG']
for policy in policies:
    rewards = data[policy]["r"].sum(axis=1).flatten()
    rewards = np.median(rewards)
    print(policy,oracle_r, rewards)
    normalized_gap = (oracle_r - rewards) / nsteps*2

    mad = np.median(np.abs( np.median(data[policy]["r"].sum(axis=1).flatten()) - data[policy]["r"].sum(axis=1).flatten()))
    
    print(f'{policy}:')
    print(f'  Normalized Optimality Gap: {normalized_gap:.4f}')
    print(f'  Median Absolute Deviation (MAD): {mad}')
paper_plot(data)