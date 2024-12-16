import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

# Load in the learning data
def load_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

data_DDPG_square = load_data('data_DDPG_square.pkl')
data_DDPG_sparse = load_data('data_DDPG_sparse.pkl')
data_DDPG_abs = load_data('data_DDPG_abs.pkl')

def paper_plot(data_DDPG_square, data_DDPG_sparse, data_DDPG_abs):
    # Set up LaTeX rendering
    rcParams['text.usetex'] = True
    rcParams['font.family'] = 'serif'
    rcParams['axes.labelsize'] = 10
    rcParams['xtick.labelsize'] = 10
    rcParams['ytick.labelsize'] = 10
    rcParams['legend.fontsize'] = 10

    t = np.linspace(0, 25, 120)
    
    # A4 width in inches
    a4_width_inches = 8.27
    
    # Calculate height to maintain aspect ratio
    height = a4_width_inches * 0.8  # Increased height for three subplots
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(a4_width_inches, height), sharex=True)
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.98, hspace=0.2)

    alphas = np.linspace(0.1, 1, int(301))
    colors = ['tab:blue', 'tab:red', 'tab:green']
    labels = ['Square error', 'Sparse reward', 'Absolute error']
    data_sets = [data_DDPG_square, data_DDPG_sparse, data_DDPG_abs]
    axes = [ax1, ax2, ax3]

    # Create lines for the legend
    lines = []
    for i, label in enumerate(labels):
        line, = ax1.plot([], [], color=colors[i], label=label)
        lines.append(line)
    ref_line, = ax1.plot([], [], color='black', linestyle='--', label='Reference')
    lines.append(ref_line)

    # Create legend above the plot
    fig.legend(handles=lines, loc='upper center', bbox_to_anchor=(0.5, 0.98),
               ncol=4, frameon=False, columnspacing=1)

    for i, alpha in enumerate(alphas):
        
        if i % 5 == 0:
            # Adjust alpha based on iteration number
            if i <= 250:
                # Scale from 0.2 to 0.8 for iterations 0-250
                plot_alpha = 0.2 + (0.2 * i / 280)
            else:
                # Use alpha >= 0.8 for iterations > 250
                plot_alpha = max(0.8, alpha)
                
            for j, (data, ax) in enumerate(zip(data_sets, axes)):
                ax.plot(t, data[i]['pol_i']['x'][0,:,0], color=colors[j], alpha=plot_alpha)

    for ax in axes:
        ax.step(t, data_DDPG_abs[0]['pol_i']['x'][2,:,0], '--', color='black')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        ax.set_ylabel(r'$C_A$ [mol/m$^3$]')
        ax.set_xlim(0, 25)

    ax3.set_xlabel(r'Time [min]')

    plt.savefig('r_showcase_learning_split.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.show()

paper_plot(data_DDPG_square, data_DDPG_sparse, data_DDPG_abs)