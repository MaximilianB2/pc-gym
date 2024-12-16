import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# Load data
norm_data = np.load('norm_rollout_data.npy', allow_pickle=True)
cons_data = np.load('constraint_rollout_data.npy', allow_pickle=True)


norm_data = np.load('norm_rollout_data.npy', allow_pickle=True).item()
cons_data = np.load('constraint_rollout_data.npy', allow_pickle=True).item()


norm_x = norm_data['DDPG']['x']
norm_u = norm_data['DDPG']['u']

cons_x = cons_data['DDPG']['x']
cons_u = cons_data['DDPG']['u']

oracle_cons_x = cons_data['oracle']['x']
oracle_cons_u = cons_data['oracle']['u']


oracle_r = np.median(cons_data['oracle']["r"].sum(axis=1).flatten())
policies = ['DDPG']
for i, policy in enumerate(policies):
    print(f'{policy} optimality gap: {oracle_r - np.median(cons_data[policy]["r"].sum(axis=1).flatten())}')
def cstr_comparison_plot(norm_data, cons_data, cons, cons_type):
    # Set up LaTeX rendering
    rcParams['text.usetex'] = True
    rcParams['font.family'] = 'serif'
    rcParams['axes.labelsize'] = 10
    rcParams['xtick.labelsize'] = 10
    rcParams['ytick.labelsize'] = 10
    rcParams['legend.fontsize'] = 10

    t = np.linspace(0, 25, 60)
    
    # A4 width in inches
    a4_width_inches = 8.27
    
    # Calculate height to maintain aspect ratio
    height = a4_width_inches * 0.4  # Adjust this factor as needed
    
    fig, axs = plt.subplots(1, 3, figsize=(a4_width_inches, height))
    plt.subplots_adjust(wspace=0.3, top=0.85, bottom=0.15, left=0.08, right=0.98)

    policies = ['Normal', 'Constrained', 'Oracle']
    cols = ['tab:blue', 'tab:orange', 'tab:green']
    labels = ['Normal', 'Constrained', 'Oracle']

    # Create lines for the legend
    lines = []
    for i, policy in enumerate(policies):
        line, = axs[0].plot([], [], color=cols[i], label=labels[i])
        lines.append(line)
    ref_line, = axs[0].plot([], [], color='black', linestyle='--',  label='Reference')
    lines.append(ref_line)
    constraint_line, = axs[0].plot([], [], color='red', linestyle=':', label='Constraint')
    lines.append(constraint_line)

    # Create legend above the plots
    fig.legend(handles=lines, loc='upper center', bbox_to_anchor=(0.37, 0.94),
                ncol=5, frameon=False, columnspacing=1)

    y_labels = [r'$C_A$ [mol/m$^3$]', r'$T$ [K]', r'$T_c$ [K]']
    
    for idx, ax in enumerate(axs):
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        
        if idx < 2:  # For Ca and T plots
            ax.plot(t, np.median(norm_data['DDPG']['x'][idx,:,:], axis=1), color=cols[0], linewidth=1.25)
            ax.fill_between(t, np.max(norm_data['DDPG']['x'][idx,:,:], axis=1), 
                            np.min(norm_data['DDPG']['x'][idx,:,:], axis=1), 
                            alpha=0.2, linewidth=0, color=cols[0])
            
            ax.plot(t, np.median(cons_data['DDPG']['x'][idx,:,:], axis=1), color=cols[1], linewidth=1.25)
            ax.fill_between(t, np.max(cons_data['DDPG']['x'][idx,:,:], axis=1), 
                            np.min(cons_data['DDPG']['x'][idx,:,:], axis=1), 
                            alpha=0.2, linewidth=0, color=cols[1])
            
            ax.plot(t, np.median(cons_data['oracle']['x'][idx,:,:], axis=1), color=cols[2], linewidth=1.25)
            ax.fill_between(t, np.max(cons_data['oracle']['x'][idx,:,:], axis=1), 
                            np.min(cons_data['oracle']['x'][idx,:,:], axis=1), 
                            alpha=0.2, linewidth=0, color=cols[2])
            
            ax.set_ylabel(y_labels[idx])
            ax.set_xlabel(r'Time [min]')
            ax.set_xlim(0, 25)
            
            
            if idx == 0:
                ax.step(t, cons_data['DDPG']['x'][2,:,0], color='black', linestyle='--')
            
            # Add constraint visualization for temperature plot
            if idx == 1:  # Temperature plot
                ax.set_ylim(318,328)
                for i, (constraint_value, constraint_type) in enumerate(zip(cons['T'], cons_type['T'])):
                    ax.axhline(y=constraint_value, color='red', linestyle=':', linewidth=1.5)
                    if constraint_type == '<=':
                        ax.fill_between(t, constraint_value, ax.get_ylim()[1], alpha=0.1, color='red')
                    elif constraint_type == '>=':
                        ax.fill_between(t, ax.get_ylim()[0], constraint_value, alpha=0.1, color='red')
        
        elif idx == 2:  # For Tc plot
            ax.step(t, np.median(norm_data['DDPG']['u'][0,:,:], axis=1), color=cols[0], where='post', linewidth=1.25)
            ax.fill_between(t, np.max(norm_data['DDPG']['u'][0,:,:], axis=1), 
                            np.min(norm_data['DDPG']['u'][0,:,:], axis=1),
                            step="post", alpha=0.2, linewidth=0, color=cols[0])
            
            ax.step(t, np.median(cons_data['DDPG']['u'][0,:,:], axis=1), color=cols[1], where='post', linewidth=1.25)
            ax.fill_between(t, np.max(cons_data['DDPG']['u'][0,:,:], axis=1), 
                            np.min(cons_data['DDPG']['u'][0,:,:], axis=1),
                            step="post", alpha=0.2, linewidth=0, color=cols[1])
            
            ax.step(t, np.median(cons_data['oracle']['u'][0,:,:], axis=1), color=cols[2], where='post', linewidth=1.25)
            ax.fill_between(t, np.max(cons_data['oracle']['u'][0,:,:], axis=1), 
                            np.min(cons_data['oracle']['u'][0,:,:], axis=1),
                            step="post", alpha=0.2, linewidth=0, color=cols[2])
            
            ax.set_ylabel(y_labels[idx])
            ax.set_xlabel(r'Time [min]')
            ax.set_xlim(0, 25)

    # Adjust the plots to be square and the same size
    for ax in axs:
        ax.set_box_aspect(1)
    
    plt.savefig('cstr_comparison_with_constraints.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.show()


cons = {'T':[327,321]}
cons_type = {'T': ['<=', '>=']}


cstr_comparison_plot(norm_data, cons_data, cons, cons_type)

def calculate_violation_metric(data, cons):
    def violation_rate(x):
        violations = (x > cons['T'][0]) | (x < cons['T'][1])
        return np.mean(violations)

    N = data['DDPG']['x'].shape[2]  # Number of time steps
    
    metrics = {}
    
    # Normal policy (from norm_data)
    metrics['Normal'] = violation_rate(data['DDPG']['x'][1, :, :])
    
    # Constrained policy (from cons_data['DDPG'])
    metrics['Constrained'] = violation_rate(data['DDPG']['x'][1, :, :])
    
    # Oracle policy (from cons_data['oracle'])
    try:
        metrics['Oracle'] = violation_rate(data['oracle']['x'][1, :, :])
    except:
        pass
    return metrics

# Calculate metrics for cons_data
cons_metrics = calculate_violation_metric(cons_data, cons)
# Calculate metrics for norm_data
norm_metrics = calculate_violation_metric(norm_data, cons)
# Print results
print("Violation Metrics:")
print(f"Normal Policy: {norm_metrics['Normal']:.4f}")
print(f"Constrained Policy: {cons_metrics['Constrained']:.4f}")
print(f"Oracle Policy: {cons_metrics['Oracle']:.4f}")