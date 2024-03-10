<h1 align="center">
  <a href="https://github.com/MaximilianB2/pc-gym/blob/main/docs/img/pc-gym-blue-Ai.png">
    <img src="https://github.com/MaximilianB2/pc-gym/blob/main/docs/img/pc-gym-blue-Ai.png"/></a><br>
  <b>Reinforcement learning environments for process control 🧪</b><br>
</h1>
<p align="center">
      <a href="https://www.python.org/doc/versions/">
        <img src="https://img.shields.io/badge/python-3.10-blue.svg" /></a>  
      <a href="https://opensource.org/license/mit">
        <img src="https://img.shields.io/badge/license-MIT-orange" /></a>
</p>


## Quick start ⚡
Setup a CSTR environment with a setpoint change

```python 

# Setpoint
SP = {'Ca': [0.85 for i in range(int(nsteps/2))] + [0.9 for i in range(int(nsteps/2))]} 

# Action and observation Space
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
env = pcgym.make_env(env_params)

# Reset the environment
obs, state = env.reset()

# Sample a random action
action = env.action_space.sample()

# Perform a step in the environment
obs, rew, done, term, info = env.step(action)
```
## Documentation

You can read the full documentation [here](https://maximilianb2.github.io/pc-gym/)!

## Installation ⏳

The latest pc-gym version can be installed from PyPI:

```bash
pip install pcgym
```

## Examples

TODO: Link example notebooks here

## Implemented Process Control Environments

TODO: Add table of environments

## Other Great Gyms 🔍

TODO: Link other gyms such as Jumanji, safety gymnasium etc.

## Citing `pc-gym`
If you use `pc-gym` in your research, please cite using the following 
```
@software{pcgym2024,
  author = {Max Bloor and ...},
  title = {{pc-gym}: Reinforcement Learning Envionments for Process Control},
  url = {https://github.com/MaximilianB2/pc-gym},
  version = {0.0.4},
  year = {2024},
}
```
