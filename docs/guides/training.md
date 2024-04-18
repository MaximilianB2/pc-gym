This is a user guide to setup a training algorithm using the pc-gym environment. We use  <a href="https://stable-baselines3.readthedocs.io/en/master/#">Stable Baselines 3</a> to implement the reinforcement learning algorithm  <a href ="https://arxiv.org/abs/1707.06347"> Proximal Policy Optimization (PPO)</a>.

### Environment Definition
Firstly import the pc-gym library, numpy and stable baselines 3.

```py
from pcgym import make_env
import numpy as np 
from stable_baselines3 import PPO
```
In control systems, a setpoint is the target or goal for a system's output. It's the value that the control system aims to achieve.

In this code snippet, a dictionary named `SP` is created to store the setpoints for each state. The keys in the dictionary represent the state (concentration of species A), and the values are lists that represent the setpoints for each step in the state.
```py
T = 26
nsteps = 100
# Enter required setpoints for each state.
SP = {
    'Ca': [0.85 for i in range(int(nsteps/2))] + [0.9 for i in range(int(nsteps/2))],
}

```
In reinforcement learning, the `action_space` and `observation_space` are two important concepts that define the range of possible actions that an agent can take and the range of possible observations that an agent can perceive, respectively.

In this code snippet, both the `action_space` and `observation_space` are defined as continuous spaces, represented by a dictionary with 'low' and 'high' keys. 
```py
# Continuous box action space
action_space = {
    'low': np.array([295]),
    'high':np.array([302]) 
}

# Continuous box observation space
observation_space = {
    'low' : np.array([0.7,300,0.8]),
    'high' : np.array([1,350,0.9])  
}
```

In this code snippet, a dictionary named `env_params` is created to store the parameters of the environment. An instance of the `Models_env` OpenAI gym class is created with the parameters defined in the `env_params` dictionary.

```py
r_scale ={
    'Ca': 1e3 #Reward scale for each state
}
env_params = {
    'N': nsteps, # Number of time steps
    'tsim':T, # Simulation Time
    'SP':SP, # Setpoint
    'o_space' : observation_space, # Observation space
    'a_space' : action_space, # Action space
    'x0': np.array([0.8,330,0.8]), # Initial conditions 
    'model': 'cstr_ode', # Select the model
    'r_scale': r_scale, # Scale the L1 norm used for reward (|x-x_sp|*r_scale)
    'normalise_a': True, # Normalise the actions
    'normalise_o':True, # Normalise the states,
    'noise':True, # Add noise to the states
    'integration_method': 'casadi', # Select the integration method
    'noise_percentage':0.001 # Noise percentage
}
env = make_env(env_params)
```

### Policy Training
Next the policy can be trained using the previously defined environment and the PPO algorithm from stable baselines 3.
```py
nsteps_learning = 3e4
PPO_policy = PPO('MlpPolicy', env, verbose=1,learning_rate=0.001).learn(nsteps_learning)
```

### Policy Rollout and Plotting
With the trained policy, the `plot_rollout` method can be called to rollout and plot the resulting state and control values. The method returns an instance of the evaluator class and the data used for the plots

```py
evaluator, data = env.plot_rollout({'PPO': PPO_policy}, reps = 10)
```


<figure>
  <img src="../../img/init-policy.png" alt="Image title" style="width:100%">
</figure>
