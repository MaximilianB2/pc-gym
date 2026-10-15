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
The observation provides the four tank heights and configured setpoints as a one-dimensional
float32 array of shape `(4 + N_SP,)`. With the registered default targets its order is
``[h1, h2, h3, h4, h3_SP, h4_SP]``.

The observation space is defined by the following bounds corresponding to the ordered state variables: 
```
[[0,0.6],[0,0.6],[0,0.6],[0,0.6],[0,0.6],[0,0.6]]
```
The physical initial heights and the default targets exposed by reset are:
```
[0.141, 0.112, 0.072, 0.42, 0.1, 0.3]
```

## Action Space
The physical actions are the pump voltages `[v1, v2]`, with lower bounds `[0.1, 0.1] V`
and upper bounds `[10, 9] V`. By default the public action and observation spaces normalize
their physical ranges to `[-1, 1]`; use `normalise_a=False` / `normalise_o=False` for physical units.

The second pump maximum was reduced from 10 V because its upper-tank equilibrium is
$h_3 = ((1-\gamma_2) k_2 v_2/a_3)^2/(2g)$: about 0.736 m at 10 V, exceeding the
0.6 m tank-height domain, versus 0.596 m at 9 V. With the default parameters the flow
balances point into the box on every boundary face for all actions in the corrected range.
Changing model parameters or user-supplied limits requires checking that property again.

## Reward 
The default reward is the negative sum of squared tracking errors for `h3=0.1 m` and
`h4=0.3 m`, each divided by `0.6**2`. Both targets admit a simultaneous steady operating
point within the pump and height limits. Independent equilibrium checks and complete
1000 s action-scenario episodes pass; a validated RL learning recipe is still pending.

## Reference
The original model was created by [Johansson et. al. (2000)](https://ieeexplore.ieee.org/document/845876).
