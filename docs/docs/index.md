
<figure>
  <img src="img/pc-gym%20dark.png#only-dark" alt="Image title" style="width:100%">
  <img src="img/pc-gym.png#only-light" alt="Image title" style="width:100%">
</figure>
## Welcome to pc-gym 💪

pc-gym is a suite of environments designed to aid the development of safe-rl algorithms used for chemical process control.


### Overview


| Environment                              | Category  | Source                                                                                           | Description                                                            |
|------------------------------------------|-----------|--------------------------------------------------------------------------------------------------|------------------------------------------------------------------------|
| CSTR                              | Reactor           |                              [code]                                                             | [doc]  |
| First Order System                              | Simple Model                                           |                              [code]                                           | [doc]  |
| Second Order System                              | Simple Model           |                              [code]                                         | [doc]  |
| Reactor-Separator-Recycle                              | Plantwide Control           |                              [code]                                 | [doc]  |
| Distillation Column                              | Column           |                              [code]                                                  | [doc]  |
| Multistage Extration                               | Column          |                              [code]                                                 | [doc]  |
| Multistage Extraction                              | Column/Reactor           |                              [code]                                        | [doc]  |
| Heat Exchanger                              | Heat transfer        |                              [code]                                                   | [doc]  |
| Poltmerisation Reactor                           | Reactor          |                              [code]                                                  | [doc]  |
| Biofilm Reactor                           | Bioprocess         |           [code]                                                                          | [doc]  |
| Four Tank                          | Settling Tank         |                              [code]                                                           | [doc]  |


### Installation
You can install the latest version of pc-gym from PyPI:
```bash
pip install pc-gym
```

### Getting Started

Those from the RL community will recognise pc-gym's interface as it is built upon the [OpenAI Gym](https://github.com/openai/gym). Below is an example of how to define an environment.

```py
import numpy as np 
#Global params
T = 26
nsteps = 100
#Enter required setpoints for each state. Enter None for states without setpoints.
SP = {
    '0': [0.8 for i in range(int(nsteps/2))] + [0.9 for i in range(int(nsteps/2))],
}

#Continuous box action space
action_space = {
    'low': np.array([-1]),
    'high':np.array([1]) 
}
#Continuous box observation space
observation_space = {
    'low' : np.array([0.7,300,0.8]),
    'high' : np.array([1,350,0.9])  
}

env_params = {
    'Nx': 2, # Number of states
    'N': nsteps, # Number of time steps
    'tsim':T, # Simulation Time
    'Nu':1, # Number of control/actions
    'SP':SP, #Setpoint
    'o_space' : observation_space, #Observation space
    'a_space' : action_space, # Action space
    'dt': 1., # Time step
    'x0': np.array([0.8,330,0.8]), # Initial conditions (torch.tensor)
    'model': 'cstr_ode', #Select the model
    'r_scale': np.array([5]), #Scale the L1 norm used for reward (|x-x_sp|*r_scale)
    'normalise': True, #Normalise the states 
}
env = Models_env(env_params)
```
Once the environment is defined it can then be used in off-the-shelf RL packages such as [Stable Baselines 3](https://stable-baselines3.readthedocs.io/en/master/).
```py
from stable_baselines3 import PPO
n_steps = 3e4
policy = PPO('MlpPolicy', env, verbose=1,learning_rate=0.01).learn(n_step)
```
Then the trained policy can be rollouted and visualised.
```py
reps = 10
env.plot_rollout(policy,reps)
```
<figure>
  <img src="img/init-policy.png" alt="Image title" style="width:100%">
</figure>