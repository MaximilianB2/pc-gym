import matplotlib.pyplot as plt
import numpy as np

# Load data
norm_data = np.load('norm_rollout_data.npy', allow_pickle=True)
cons_data = np.load('constraint_rollout_data.npy', allow_pickle=True)


norm_data = np.load('norm_rollout_data.npy', allow_pickle=True).item()
cons_data = np.load('constraint_rollout_data.npy', allow_pickle=True).item()


norm_x = norm_data['SAC']['x']
norm_u = norm_data['SAC']['u']

cons_x = cons_data['SAC']['x']
cons_u = cons_data['SAC']['u']

oracle_cons_x = cons_data['oracle']['x']
oracle_cons_u = cons_data['oracle']['u']


t = np.linspace(0,26,120)
plt.figure()
fig, axs = plt.subplots(3, 1)
# Ca
axs[0].plot(t,np.median(norm_x[0,:,:],axis=1))
axs[0].plot(t,np.median(cons_x[0,:,:],axis=1), label = 'Constraint Penalty')

# T
axs[1].plot(t,np.median(norm_x[1,:,:], axis=1))
axs[1].plot(t,np.median(cons_x[1,:,:], axis=1), label = 'Constraint Penalty')
# Tc
axs[2].step(t,np.median(norm_u[0,:,:],axis=1))
axs[2].step(t,np.median(cons_u[0,:,:],axis=1), label = 'Constraint Penalty')
plt.show()