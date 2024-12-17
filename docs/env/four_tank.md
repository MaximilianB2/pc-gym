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

# Four Tank System
The four-tank system is a multivariable process consisting of four interconnected water tanks. This system is often used as a benchmark for control systems due to its nonlinear dynamics and the coupling between inputs and outputs. The model describes the change in water levels in each tank based on the inflows and outflows.
\begin{align}
\frac{dh_1}{dt} &= -\frac{a_1}{A_1}\sqrt{2g_ah_1} + \frac{a_3}{A_1}\sqrt{2g_ah_3} + \frac{\gamma_1 k_1}{A_1}v_1 \\
\frac{dh_2}{dt} &= -\frac{a_2}{A_2}\sqrt{2g_ah_2} + \frac{a_4}{A_2}\sqrt{2g_ah_4} + \frac{\gamma_2 k_2}{A_2}v_2 \\
\frac{dh_3}{dt} &= -\frac{a_3}{A_3}\sqrt{2g_ah_3} + \frac{(1-\gamma_2)k_2}{A_3}v_2 \\
\frac{dh_4}{dt} &= -\frac{a_4}{A_4}\sqrt{2g_ah_4} + \frac{(1-\gamma_1)k_1}{A_4}v_1
\end{align}

## Observation Space
The observation of the `four_tank` environment provides information on the state variables and their associated setpoints (if they exist) at the current timestep. The observation is an array of shape `(1, 4 + N_SP)` where `N_SP` is the number of setpoints. Therefore, the observation when there a setpoint exists for $h3_SP$ and $h4_SP$ is
``[h1, h2, h3, h4, h3_SP, h4_SP]``.

The observation space is defined by the following bounds corresponding to the ordered state variables: 
```
[[0,0.6],[0,0.6],[0,0.6],[0,0.6],[0,0.6],[0,0.6]]
```
An example, tested set of initial conditions are as follows:
```
[0.141, 0.112, 0.072, 0.42, 0.5, 0.2]
```

## Action Space
The action space consists of two variables (v_1 & v_2) which represent the voltages to the respective pumps. The space is defined as a `continuous box` of `[[0,0],[10,10]]`.

## Reward 
The reward is a continuous value corresponding to square error of the state and its setpoint. For multiple states, these are scaled with a factor `r_scale` and summed to give a single value. The goal of this environment is to drive the $x_1$ state to the origin.

## Reference
The original model was created by [Johansson et. al. (2000)](https://ieeexplore.ieee.org/document/845876).