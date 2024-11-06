import numpy as np
import matplotlib.pyplot as plt
data = np.load('data.npy', allow_pickle=True).item()

print(data['SAC']['r'].shape)
print(data['SAC']['r'].sum(axis = 1))

plt.figure()
plt.hist(data['SAC']['r'].sum(axis = 1))
plt.show()