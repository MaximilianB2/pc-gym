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
    height = a4_width_inches * 0.4  # Adjust this factor as needed
    
    fig, ax = plt.subplots(figsize=(a4_width_inches, height))
    plt.subplots_adjust(top=0.85, bottom=0.15, left=0.08, right=0.98)

    alphas = np.linspace(0.1, 1, 30)
    colors = ['tab:blue', 'tab:red', 'tab:green']
    labels = ['Square error', 'Sparse reward', 'Absolute error']
    data_sets = [data_DDPG_square, data_DDPG_sparse, data_DDPG_abs]

    # Create lines for the legend
    lines = []
    for i, label in enumerate(labels):
        line, = ax.plot([], [], color=colors[i], label=label)
        lines.append(line)
    ref_line, = ax.plot([], [], color='black', linestyle='--', label='Reference')
    lines.append(ref_line)

    # Create legend above the plot
    fig.legend(handles=lines, loc='upper center', bbox_to_anchor=(0.5, 0.98),
               ncol=4, frameon=False, columnspacing=1)

    for i, alpha in enumerate(alphas):
        iterations = (i + 1) * 500
        if i % 2 == 0 or i == 29:
            for j, data in enumerate(data_sets):
                ax.plot(t, data[i]['pol_i']['x'][0,:,0], color=colors[j], alpha=alpha)

    ax.step(t, data_DDPG_abs[0]['pol_i']['x'][2,:,0], '--', color='black')
    
    ax.set_xlabel(r'Time [min]')
    ax.set_ylabel(r'$C_A$ [mol/m$^3$]')
    ax.set_xlim(0, 25)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    plt.savefig('r_showcase_learning.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.show()

paper_plot(data_DDPG_square, data_DDPG_sparse, data_DDPG_abs)