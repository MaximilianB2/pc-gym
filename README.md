<h1 align="center">
  <a href="https://github.com/MaximilianB2/pc-gym/blob/main/docs/img/pc-gym-blue-nobg.png">
    <img src="https://github.com/MaximilianB2/pc-gym/blob/main/docs/img/pc-gym-blue-nobg.png"/></a><br>
  <b>Reinforcement learning environments for process control applications 🧪</b><br>
</h1>


## Quick start
Setup a CSTR environment with a setpoint change

```python 

# Setpoint
SP = {'Ca': [0.85 for i in range(int(nsteps/2))] + [0.9 for i in range(int(nsteps/2))]} 

# Action and Observation Space
action_space = {'low': np.array([295]), 'high': np.array([302])}
observation_space = {'low': np.array([0.7,300,0.8]),'high': np.array([1,350,0.9])}

# Construct the environment parameter dictionary
env_params = {
    'N': nsteps, # Number of time steps
    'tsim':T, # Simulation Time
    'SP' :SP, 
    'o_space' : observation_space, 
    'a_space' : action_space, 
    'x0': np.array([0.8, 330, 0.8]), # Initial conditions [Ca, T, Ca_SP]
    'model': 'cstr_ode', # Select the model
}

# Create environment
env = pcgym.Models_env(env_params)


```

## Documentation

You can read the full documentation [here](https://maximilianb2.github.io/pc-gym/)!

## Installation ⏳

The latest pc-gym version can be installed from PyPI:

```bash
pip install pcgym
```
