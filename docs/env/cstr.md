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
<div style="display: flex; justify-content: center;">
  <img src="../../img/CSTR_PFD.png" alt="Image title" style="width:40%">
</div>
 

## Description & Equations
The continuously stirred tank reactor (CSTR) is a system which converts species A to species B via the reaction: A  →  B. The reactor's temperature is controlled by a cooling jacket. The following system of equations describes the system:


\begin{align}
  \nonumber\frac{\mathrm{d}C_A}{\mathrm{d}t} &= \frac{q}{V}(C_{A_f} - C_A) - kC_Ae^{\frac{-E_A}{RT}}\\
\end{align}
\begin{align}
  \nonumber\frac{\mathrm{d}T}{\mathrm{d}t} &= \frac{q}{V}(T_f - T) -\frac{\Delta H_R}{\rho C_p}kC_Ae^{\frac{-E_A}{RT}} + \frac{UA}{\rho C_p V}(T_c - T)
\end{align}
where $C_A$, the concentration of species A in the reactor, and $T$, the temperature of the reactor, are the state variables, $\mathbf{x} = [C_A, T]^\intercal \in \mathbb{R}^2$ while, $u = T_c$, the cooling water temperature, is the action variable.

## Observation
The observation is a float32 vector of shape `(2 + N_SP,)`, with model states followed
by their configured setpoints. The registered task has shape `(3,)`: `[Ca, T, Ca_setpoint]`,
with a target concentration of `0.9`. Observations are normalized to `[-1, 1]` by default;
the bounds below are in physical units.

The observation space is defined by the following bounds corresponding to the ordered state variables:
```
[[0.7,1],[300,350],[0.8,0.9]]
```
The registered initial observation, in physical units, is:
```
[0.8, 330, 0.9]
```

## Action
The physical jacket-temperature bounds are `[295, 302]` K. The public action vector has
shape `(1,)` and is normalized to `[-1, 1]` by default.

## Reward

The registered reward is `-((Ca - 0.9) / 0.3)**2`, evaluated at the next state. The initial
training configuration uses 60 transitions over 26 minutes and truncates at the horizon.
See the [fixed training protocol](../reviews/initial_training_protocol.md). Explicit user
setpoints and `r_scale` can define a different task.

## Reference

This model implementation and its description were kindly provided by [Akhil Ahmed](https://scholar.google.com/citations?user=AS34x7cAAAAJ). The original model was created by by [Hedengren (2022)](https://apmonitor.com/pdc/index.php/Main/StirredReactor).
