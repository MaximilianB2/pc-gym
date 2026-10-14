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

# Cement Calcination (Precalciner)
The precalciner is the stage of a cement plant where raw meal is heated so that calcium carbonate decomposes into calcium oxide and CO2 before entering the kiln. This model, adapted from the Gigaton reference simulator, lumps the precalciner (PC) and the two preheater cyclones feeding it (CY4, CY3) into three stages, each with a gas temperature $T_i$ and a degree of calcination $\delta_i \in [0, 1]$ of the material in it.

Each stage temperature follows an energy balance,

<div>
\begin{align}
\frac{dT_i}{dt} = \dot{T}^{\text{amb}}_i + \dot{T}^{\text{fuel}}_i + \dot{T}^{\text{air}}_i + \dot{T}^{\text{mat}}_i + \dot{T}^{\text{gas}}_i + \dot{T}^{\text{calc}}_i,
\end{align}
</div>

with heat loss to ambient, fuel combustion (PC only), cooling by fan-driven air, the heat sink of the descending raw meal, counter-current gas heat exchange between stages (kiln → PC → CY4 → CY3) and the endothermic calcination reaction. Calcination follows Arrhenius kinetics with CO2 equilibrium suppression (Baker, 1962), and material is transported downwards (CY3 → CY4 → PC) at a rate set by the material flow:

<div>
\begin{align}
\frac{d\delta_i}{dt} = k_r(T_i)\,(1-\delta_i)\,s_i + \omega_i\,(\delta_{\text{in},i} - \delta_i),
\end{align}
</div>

where $s_i \in [0,1]$ is the equilibrium suppression factor and $\omega_i$ the material turnover rate of stage $i$. Time is in hours.

## Observation Space
The observation of the `cement_calcination` environment provides information on the state variables and their associated setpoints (if they exist) at the current timestep. The observation is an array of shape `(1, 6 + N_SP)` where `N_SP` is the number of setpoints. Therefore, the observation when a setpoint exists for $T_{PC}$ is
``[t_pc, t_cy4, t_cy3, doc_pc, doc_cy4, doc_cy3, t_pc_SP]``.

The observation space is defined by the following bounds corresponding to the ordered state variables (temperatures in °C, degrees of calcination dimensionless):
```
[[0,1200],[0,1200],[0,1200],[0,1],[0,1],[0,1],[0,1200]]
```
An example, tested set of initial conditions (the steady state of the nominal operating point) is:
```
[900.5745, 797.9064, 652.4700, 0.9220, 0.05, 0.05, 875.0]
```

## Action Space
The action space consists of three variables: the raw meal feed rate `material_flow` (t/h), the precalciner fuel power `pc_fuel_power` (GJ/h) and the fan speed `fan_speed` (%). The space is defined as a `continuous box` of `[[90, 50, 50], [180, 140, 100]]`.

## Reward
The default reward is the square error between the state and its setpoint, as for the other environments. In practice a precalciner is not run to a temperature setpoint: the operating objective is to keep the degree of calcination leaving the precalciner inside a target band at minimum specific fuel consumption, while respecting the temperature limits of the refractory. A per-step cost of this form, used in the operator-in-the-loop study for which this environment was built, is

<div>
\begin{align}
c_t = w_1\,\underbrace{\frac{u^{\text{fuel}}_t}{u^{\text{feed}}_t\,\delta^{PC}_t}}_{\text{specific fuel [GJ/t]}}
    + w_2\,\underbrace{\operatorname{dist}\!\left(100\,\delta^{PC}_t,\,[85, 92]\right)}_{\text{calcination band error [\%]}}
    + w_3\,\underbrace{\operatorname{dist}\!\left(T^{PC}_t,\,[860, 890]\right)}_{\text{soft band violation [°C]}}
    + w_4\,\underbrace{\mathbb{1}\!\left[T^{PC}_t \notin [840, 905]\right]}_{\text{hard limit violation}},
\end{align}
</div>

where $\operatorname{dist}(x, [a, b])$ is the distance from $x$ to the interval (zero inside it). It can be supplied to the environment through `custom_reward` (as a negative reward); see [the study's code](https://github.com/nath-a15/operator-in-the-loop-control) for the weights used.

## Reference
Adapted from the Gigaton precalciner reference simulator (M. Craig), with permission. The CO2 equilibrium suppression follows Baker (1962).