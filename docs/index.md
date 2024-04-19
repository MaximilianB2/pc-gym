<script type="text/javascript"
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS_CHTML">
</script>
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [['$','$'], ['\\(','\\)']],
      processEscapes: true},
      jax: ["input/TeX","input/MathML","input/AsciiMath","output/CommonHTML"],
      extensions: ["tex2jax.js","mml2jax.js","asciimath2jax.js","MathMenu.js","MathZoom.js","AssistiveMML.js", "[Contrib]/a11y/accessibility-menu.js"],
      TeX: {
      extensions: ["AMSmath.js","AMSsymbols.js","noErrors.js","noUndefined.js"],
      equationNumbers: {
      autoNumber: "AMS"
      }
    }
  });
</script>

 
<figure>
  <img src="img/pc-gym%20dark.png#only-dark" alt="Image title" style="width:100%">
  <img src="img/pc-gym.png#only-light" alt="Image title" style="width:100%">
</figure>

## Welcome!
Process Control (pc-) gym is a set of benchmark chemical process control problems for reinforcement learning with integrated policy evaluation methods to aid the development of reinforcement learning algorithms.

The pc-gym was developed within the [Sargent Centre for Process Systems Engineering](https://www.imperial.ac.uk/process-systems-engineering/about-sargent-centre/) and is published as an [open-source package](https://github.com/MaximilianB2/pc-gym) which welcomes contributions from the RL and PSE communities.

*Note: this is a pre-release version of the documentation and may not reflect the current version of pc-gym*


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

## Installation 

The latest production pc-gym version can be installed from PyPI:

```bash
pip install pcgym
```
Alternatively, you can install the latest development version directly from GitHub:

```bash
pip install git+https://github.com/MaximilianB2/pc-gym
```
## Examples
Example notebooks with training walkthroughs, implementing constraints, disturbances and the policy evaluation tool can be found [here](https://github.com/MaximilianB2/pc-gym/tree/main/example_notebooks).

## Environments 
Currently there are three implemented process control environments, this will be expanded to 10 prior to full release.
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
    <td><a href="https://github.com/MaximilianB2/pc-gym/blob/main/src/pcgym/model_classes.py">Code</a></td>
    <td><a href="https://maximilianb2.github.io/pc-gym/env/cstr/">Documentation</a></td>
  </tr>
  <tr>
    <td>First Order System</td>
    <td>Simple Model</td> 
    <td><a href="https://github.com/MaximilianB2/pc-gym/blob/main/src/pcgym/model_classes.py">Code</a></td>
    <td><a href="https://maximilianb2.github.io/pc-gym/env/first_order_system/">Documentation</a></td>
  </tr>
  <tr>
    <td>Multistage Extration</td>
    <td>Column</td> 
    <td><a href="https://github.com/MaximilianB2/pc-gym/blob/main/src/pcgym/model_classes.py">Code</a></td>
    <td><a href="https://maximilianb2.github.io/pc-gym/env/extraction-column/">Documentation</a></td>
  </tr>
    <tr>
    <td>Nonsmooth Control</td>
    <td>Linear System</td> 
    <td><a href="https://github.com/MaximilianB2/pc-gym/blob/main/src/pcgym/model_classes.py">Code</a></td>
    <td><a href="https://maximilianb2.github.io/pc-gym/env/nonsmooth_control/">Documentation</a></td>
  </tr>
</table>
</div>

All environments use the following observation representation for $i$ states and $j$ disturbances:
\begin{align}
\nonumber o = [x_i,..., x_{i,sp}..., d_j,...] 
\end{align}
*Note: observation masking is a feature to be added prior to final release as this will allow the user to investigate partial observability problems.*
## Future Features
The following features are being worked on to be added in the near future:
 - Optimality gap visualisation
 - Observability Mask 
 - More case studies
 - Custom reward functions

## Citing `pc-gym`
If you use `pc-gym` in your research, please cite using the following 
```
@software{pcgym2024,
  author = {Max Bloor and Jose Neto and Ilya Sandoval and Max Mowbray
            and Akhil Ahmed and Mehmet Mercangoz and Calvin Tsay and Antonio Del Rio-Chanona},
  title = {{pc-gym}: Reinforcement Learning Envionments for Process Control},
  url = {https://github.com/MaximilianB2/pc-gym},
  version = {0.0.4},
  year = {2024},
}
```

## Other great gyms 
Other works have built upon the [OpenAI Gymnasium](https://gymnasium.farama.org/) framework:

- ✨[safe-control-gym](https://github.com/utiasDSL/safe-control-gym) 
- ✨[safety-gymnasium](https://github.com/PKU-Alignment/safety-gymnasium)
- ✨[gymnax](https://github.com/RobertTLange/gymnax)
