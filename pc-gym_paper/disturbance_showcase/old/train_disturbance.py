import sys
sys.path.append("..\..\..\src\pcgym") # Add local pc-gym files to path.
import numpy as np
from stable_baselines3 import SAC
import gymnasium as gym
import pickle
from pcgym import make_env

# Script to train an agent on samples from a distribution of disturbances to the feed concentration of the cstr environment. 
# Then test on an unseen sample from the same distribution.


# Define environment
T = 26
nsteps = 120

SP = {
    'Ca': [0.85 for i in range(int(nsteps))],
}

action_space = {
    'low': np.array([295]),
    'high':np.array([310]) 
}

observation_space = {
    'low' : np.array([0.7,300,0.8]),
    'high' : np.array([1,350,0.9])  
}

r_scale = {'Ca':1e3}


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
disturbance_space = {
    'low': np.array([0.9]),
    'high': np.array([1.5])
}

env_params_temp = {
    'N': nsteps, 
    'tsim':T, 
    'SP':SP, 
    'o_space' : observation_space, 
    'a_space' : action_space,
    'x0': np.array([0.8,330,0.8]),
    'r_scale': r_scale,
    'model': 'cstr', 
    'normalise_a': True, 
    'normalise_o':True, 
    'noise':True, 
    'integration_method': 'casadi', 
    'noise_percentage':0.0025, 
    'custom_reward': oracle_reward,
    'disturbance_bounds':disturbance_space
}

# env = make_env(env_params_temp)
def create_random_disturbances(seed, nsteps, low=0.9, high=1.5):
    np.random.seed(seed)
    values = np.append(np.array([1]),np.random.uniform(low, high, 2))
   # Generate three random disturbance values within the specified range
    times = np.sort(np.random.choice(range(20, nsteps - 20), 2, replace=False))  # Select two unique time steps for disturbances
    times = np.append(times, nsteps)  # Append the total number of steps to get three periods
    times = np.diff([0] + times.tolist())  # Calculate the duration of each disturbance period
    disturbances = {'Caf': np.repeat(values, times)}  # Repeat the disturbance values according to the calculated durations
    return disturbances


def create_multiple_disturbances(num_disturbances, seed, nsteps, low=0.9, high=1.5):
    disturbances_list = []
    for i in range(num_disturbances):
        disturbances_list.append(create_random_disturbances(seed + i, nsteps, low, high))
    return disturbances_list



class CyclingDisturbancesEnv(gym.Wrapper):
    def __init__(self, env, disturbances_list, unseen_test = False, training_dist = {'disturbances_list':0}):
        super().__init__(env)
        self.disturbances_list = disturbances_list
        self.current_disturbance_index = 0
        self.episode_disturbances = []  # To keep track of disturbances used in each episode
        self.unseen_test = unseen_test
        
        self.training_dist = training_dist['disturbances_list']
        self.env.env_params.update({
          'disturbances': disturbances_list[0],
          'disturbance_bounds': disturbance_space,
          })
        self.env = make_env(self.env.env_params)

    def reset(self,seed = 0):
        # Set the next disturbance
        if self.unseen_test:
            print('here')
            same = True
            while same:
              test_disturbance  = create_random_disturbances(seed+5, nsteps, low=1, high=1.1)
              if not any(np.array_equal(test_disturbance['Caf'], d['Caf']) for d in self.training_dist):
                  same = False
            
            self.env.env_params.update({
                                  'disturbances': test_disturbance,
                                  'disturbance_bounds': disturbance_space,
                                  })
            self.env = make_env(self.env.env_params)
            return self.env.reset()
        else:
          next_disturbance = self.disturbances_list[self.current_disturbance_index]
          self.env.env_params.update({
          'disturbances': next_disturbance,
          'disturbance_bounds': disturbance_space,
          })
          # print(self.env.env_params['disturbances'])
          # Record which disturbance was used
          
          self.episode_disturbances.append(self.current_disturbance_index)
          
          # Move to the next disturbance for the next reset
          self.current_disturbance_index = (self.current_disturbance_index + 1) % len(self.disturbances_list)
          self.env = make_env(self.env.env_params)
          return self.env.reset()

    def save_disturbances(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump({
                'disturbances_list': self.disturbances_list,
                'episode_disturbances': self.episode_disturbances
            }, f)

    @classmethod
    def load_disturbances(cls, env, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        instance = cls(env, data['disturbances_list'])
        instance.episode_disturbances = data['episode_disturbances']
        return instance

def create_cycling_env(env_params, num_disturbances, seed, unseen = False, training_dist = {'disturbances_list':0}):
    base_env = make_env(env_params)
    disturbances_list = create_multiple_disturbances(num_disturbances, seed, nsteps,)
    return CyclingDisturbancesEnv(base_env, disturbances_list, unseen_test=unseen, training_dist = training_dist)



def load_disturbances(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

if __name__ == '__main__':
  # Env which cycles through N samples of the disturbance space 
  N_disturbance = 5
  cycling_env = create_cycling_env(env_params_temp, N_disturbance, 1990)
  cycling_env.save_disturbances('disturbance_used.pkl')
  
  # Train your agent on this environment
  policy = SAC('MlpPolicy', cycling_env, verbose=1).learn(1e4)
  policy.save('SAC_cstr_CaF_1107.zip')

  # Test on an unseen disturbance
  train_disturbance = load_disturbances('disturbance_used.pkl')
  cycling_env = create_cycling_env(env_params_temp, N_disturbance, 1990, unseen=True, training_dist=train_disturbance)
  cycling_env.plot_rollout({'SAC': policy}, oracle=True, reps = 2, MPC_params={'N':5, 'R':0.005})