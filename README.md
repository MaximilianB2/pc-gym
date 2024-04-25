<h1 align="center">
  <a href="https://github.com/MaximilianB2/pc-gym/blob/main/docs/img/pc-gym-blue-Ai.png">
    <img src="https://github.com/MaximilianB2/pc-gym/blob/main/docs/img/pc-gym-blue-Ai.png"/></a><br>
  <b>Reinforcement learning environments for process control </b><br>
</h1>
<p align="center">
      <a href="https://www.python.org/doc/versions/">
        <img src="https://img.shields.io/badge/python-3.10-blue.svg" /></a>  
      <a href="https://opensource.org/license/mit">
        <img src="https://img.shields.io/badge/license-MIT-orange" /></a>
      <a href="https://github.com/astral-sh/ruff">
        <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json" /></a>
</p>


## Quick start ⚡
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

## Installation ⏳

The latest pc-gym version can be installed from PyPI:

```bash
pip install pcgym
```

## Examples
Example notebooks with training walkthroughs, implementing constraints, disturbances and the policy evaluation tool can be found [here](https://github.com/MaximilianB2/pc-gym/tree/main/example_notebooks).

## Implemented Process Control Environments 🎛️

|          Environment          | Reference | Source | Documentation |
|:-----------------------------:|:---------:|:------:|---------------|
|              CSTR             | [Hedengren, 2022](https://github.com/APMonitor/pdc/blob/master/CSTR_Control.ipynb)     | [Source](https://github.com/MaximilianB2/pc-gym/blob/main/src/pcgym/model_classes.py)      |               |
|       First Order Sytem       |      N/A  | [Source](https://github.com/MaximilianB2/pc-gym/blob/main/src/pcgym/model_classes.py)        |               |
| Multistage Extraction Column  |  [Ingham et al, 2007 (pg 471)](https://onlinelibrary.wiley.com/doi/book/10.1002/9783527614219)         | [Source](https://github.com/MaximilianB2/pc-gym/blob/main/src/pcgym/model_classes.py)        |               |
| Nonsmooth Control|[Lim,1969](https://pubs.acs.org/doi/epdf/10.1021/i260031a007)|[Source](https://github.com/MaximilianB2/pc-gym/blob/main/src/pcgym/model_classes.py) ||


 
## Citing `pc-gym`
If you use `pc-gym` in your research, please cite using the following 
```
@software{pcgym2024,
  author = {Max Bloor and  Jose Neto and Ilya Sandoval and Max Mowbray and Akhil Ahmed and Mehmet Mercangoz and Calvin Tsay and Antonio Del Rio-Chanona},
  title = {{pc-gym}: Reinforcement Learning Environments for Process Control},
  url = {https://github.com/MaximilianB2/pc-gym},
  version = {0.1.6},
  year = {2024},
}
```

## Other Great Gyms 🔍
- ✨[safe-control-gym](https://github.com/utiasDSL/safe-control-gym) 
- ✨[safety-gymnasium](https://github.com/PKU-Alignment/safety-gymnasium)
- ✨[gymnax](https://github.com/RobertTLange/gymnax)
