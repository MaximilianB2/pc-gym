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
The environment used in the this problem is a second-order system with distinct poles defined by the the following:


\begin{align}
\nonumber\dfrac{d}{dt} \begin{pmatrix} x_1 \\\ x_2 \ \end{pmatrix} =  \begin{bmatrix} 0 & 1 \\\ -2 & -3 \end{bmatrix} \begin{pmatrix} x_1 \\\ x_2 \ \end{pmatrix} + \begin{pmatrix} 0 \\\ 1 \ \end{pmatrix} u 
\end{align}

where $x$ is the state variable and $u$ is the action variable.

## Observation
The observation of the `Nonsmooth Control` environment provides information on the state variables and their associated setpoints (if they exist) at the current timestep. The observation is an array of shape `(1, 2)`. Therefore, the observation is ``[x_1, x_sp]``.

## Action
The action space is a `ContinuousBox` of `[-1,1]`.

## Reward

The reward is a continuous value corresponding to square error of the state and its setpoint. For multiple states, these are scaled with a factor `r_scale` and summed to give a single value. The goal of this environment is to drive the $x_1$ state to the origin.

## Reference

The original model was created by [Lim (1969)](https://pubs.acs.org/doi/epdf/10.1021/i260031a007).
