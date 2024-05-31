
import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load in the learning data
# Load data_DDPG_square
with open('data_DDPG_square.pkl', 'rb') as f:
    data_DDPG_square = pickle.load(f)

# Load data_DDPG_sparse
with open('data_DDPG_sparse.pkl', 'rb') as f:
    data_DDPG_sparse = pickle.load(f)

# Load data_DDPG_abs
with open('data_DDPG_abs.pkl', 'rb') as f:
    data_DDPG_abs = pickle.load(f)


t = np.linspace(0,25,120)
plt.figure(figsize=(20,8))
alphas = np.linspace(0.1,1,30)
plt.rcParams['text.usetex'] = 'True'
plt.rcParams['font.family'] = 'serif'
for i, alpha in enumerate(alphas):

    iterations = (i + 1) * 500  # calculate the number of iterations

    if i in [0, len(alphas)//2, len(alphas)-1]:
        label_square = f'Square error ({iterations} timesteps)'
        label_sparse = f'Sparse reward ({iterations} timesteps)'
        label_abs = f'Absolute error ({iterations} timesteps)'
    else:
        label_square = label_sparse = label_abs = None

    plt.plot(t, data_DDPG_square[i]['pol_i']['x'][0,:,0], color='tab:blue', alpha=alpha, label=label_square)
    plt.plot(t, data_DDPG_sparse[i]['pol_i']['x'][0,:,0], color='tab:red', alpha=alpha, label=label_sparse)
    plt.plot(t, data_DDPG_abs[i]['pol_i']['x'][0,:,0], color='tab:green', alpha=alpha, label=label_abs)

plt.step(t, data_DDPG_abs[0]['pol_i']['x'][2,:,0],'--',color = 'black')
plt.xlabel('Time (min)')
plt.ylabel('$C_A$ (mol/m$^3$)')
plt.xlim(0,25)
plt.grid('True')
plt.legend(bbox_to_anchor=(0.5, 1.01), loc='lower center', ncol=3)
plt.savefig('r_showcase_learning.pdf')
plt.show()

