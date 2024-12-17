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



where $x$, is the state variable, $\mathbf{x} \in \mathbb{R}^2$ while, $u$ is the action variable.

## Observation
The observation of the `First Order System` environment provides information on the state variables and their associated setpoints (if they exist) at the current timestep. The observation is an array of shape `(1, 1 + N_SP)` where `N_SP` is the number of setpoints. Therefore, the observation when there a setpoint exists
``[x, x_Setpoint]``.

## Action
The action space is a `ContinuousBox` of `[0,10]`.

## Reward

The reward is a continuous value corresponding to square error of the state and its setpoint. For multiple states, these are scaled with a factor $\gamma_i$ and summed to give a single value.

## Reference

This model implementation and its description were kindly provided by [Akhil Ahmed](https://scholar.google.com/citations?user=AS34x7cAAAAJ). 