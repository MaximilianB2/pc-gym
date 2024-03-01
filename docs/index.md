
<figure>
  <img src="img/pc-gym%20dark.png#only-dark" alt="Image title" style="width:100%">
  <img src="img/pc-gym.png#only-light" alt="Image title" style="width:100%">
</figure>

## Welcome to pc-gym 💪

The process control (pc) gym is a suite of environments and tools designed to aid the development of safe-rl algorithms used for chemical process control.

The pc-gym has been developed to address the lack of chemical process environments setup to test reinforcemnent learning algorithms. However, pc-gym is not only a collection of environements it also allows the user access to constraint violation information, input custom disturbances and visualise the policy's performance. The pc-gym has been developed by a team of researchers in <a href="https://www.imperial.ac.uk/process-systems-engineering">The Sargent Centre for Process Systems Engineering</a>.

### Environments 🌍

pc-gym provides a range of chemical process control environments ranging from first-order sytems to plantwide and bioprocess control. These environments are tabulated below along with links to their documentation and code. 


<div style="display: flex; justify-content: center;">
<table style="width:100%">
  <tr>
    <th>Environment</th>
    <th>Category</th> 
    <th>Source Code</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>CSTR</td>
    <td>Reactor</td> 
    <td><a href="link_to_code">code</a></td>
    <td><a href="link_to_doc">doc</a></td>
  </tr>
  <tr>
    <td>First Order System</td>
    <td>Simple Model</td> 
    <td><a href="link_to_code">code</a></td>
    <td><a href="link_to_doc">doc</a></td>
  </tr>
  <tr>
    <td>Second Order System</td>
    <td>Simple Model</td> 
    <td><a href="link_to_code">code</a></td>
    <td><a href="link_to_doc">doc</a></td>
  </tr>
  <tr>
    <td>Reactor-Separator-Recycle</td>
    <td>Plantwide Control</td> 
    <td><a href="link_to_code">code</a></td>
    <td><a href="link_to_doc">doc</a></td>
  </tr>
  <tr>
    <td>Distillation Column</td>
    <td>Column</td> 
    <td><a href="link_to_code">code</a></td>
    <td><a href="link_to_doc">doc</a></td>
  </tr>
  <tr>
    <td>Multistage Extration</td>
    <td>Column</td> 
    <td><a href="link_to_code">code</a></td>
    <td><a href="link_to_doc">doc</a></td>
  </tr>
  <tr>
    <td>Multistage Extraction</td>
    <td>Column/Reactor</td> 
    <td><a href="link_to_code">code</a></td>
    <td><a href="link_to_doc">doc</a></td>
  </tr>
  <tr>
    <td>Heat Exchanger</td>
    <td>Heat transfer</td> 
    <td><a href="link_to_code">code</a></td>
    <td><a href="link_to_doc">doc</a></td>
  </tr>
  <tr>
    <td>Poltmerisation Reactor</td>
    <td>Reactor</td> 
    <td><a href="link_to_code">code</a></td>
    <td><a href="link_to_doc">doc</a></td>
  </tr>
  <tr>
    <td>Biofilm Reactor</td>
    <td>Bioprocess</td> 
    <td><a href="link_to_code">code</a></td>
    <td><a href="link_to_doc">doc</a></td>
  </tr>
  <tr>
    <td>Four Tank</td>
    <td>Settling Tank</td> 
    <td><a href="link_to_code">code</a></td>
    <td><a href="link_to_doc">doc</a></td>
  </tr>
</table>
</div>



### Installation ⚙️
You can install the latest version of pc-gym from PyPI:
```bash
pip install pc-gym
```

### Getting Started 🚀

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


