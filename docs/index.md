<figure>
  <img src="img/pc-gym%20dark.png#only-dark" alt="Image title" style="width:100%">
  <img src="img/pc-gym.png#only-light" alt="Image title" style="width:100%">
</figure>

## Quick start 
Setup a CSTR environment with a setpoint change

```python 
import pcgym

# Simulation variables
nsteps = 100
T = 25

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

## Installation 

The latest pc-gym version can be installed from PyPI:

```bash
pip install pcgym
```

## Examples

TODO: Link example notebooks here

## Implemented Process Control Environments 
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
    <td>Multistage Extration</td>
    <td>Column</td> 
    <td><a href="link_to_code">code</a></td>
    <td><a href="link_to_doc">doc</a></td>
  </tr>
</table>
</div>

 
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

## Other Great Gyms 
- ✨[safe-control-gym](https://github.com/utiasDSL/safe-control-gym) 
- ✨[safety-gymnasium](https://github.com/PKU-Alignment/safety-gymnasium)
- ✨[gymnax](https://github.com/RobertTLange/gymnax)
