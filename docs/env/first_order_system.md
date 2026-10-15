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


## Description & Equations
The first order system is an environment which could represent many first order systems in engineering. It is included as a simple environment to use in the initial development of control algorithms. The following system of equations describes the system


\begin{align}
  \nonumber\frac{\mathrm{d}x}{\mathrm{d}t} &= \frac{Ku-x}{\tau}\\
\end{align}



where $x \in \mathbb{R}$ is the state variable and $u$ is the action variable.

## Observation
The registered task returns a float32 vector of shape `(2,)`: `[x, x_setpoint]`.
The physical initial state is `x=0.1` and the target is `0.7`. Both entries have physical
bounds `[0, 1]`; observations are normalized to `[-1, 1]` by default.

## Action
The physical control bounds are `[0, 1]`. The public action is a vector of shape `(1,)`,
normalized to `[-1, 1]` by default.

## Reward

The registered reward is `-(x - 0.7)**2`, evaluated at the next state. The initial training
configuration uses 60 transitions over 10 model time units and truncates at the horizon.
See the [fixed training protocol](../reviews/initial_training_protocol.md) for the controller
and evaluation conditions.

## Reference

This model implementation and its description were kindly provided by [Akhil Ahmed](https://scholar.google.com/citations?user=AS34x7cAAAAJ).
