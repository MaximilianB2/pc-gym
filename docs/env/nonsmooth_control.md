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
The registered task returns a float32 vector of shape `(3,)`: `[X1, X2, X1_setpoint]`.
The initial model state is `[0.8, 0.0]` and the target for `X1` is zero. Physical bounds
for all three entries are `[-1, 1]`; observations are normalized by default.

## Action
The physical input bounds are `[-1, 1]`, with a public action vector of shape `(1,)`.

## Reward

The registered reward is `-X1**2 / 4`, evaluated at the next state. The initial training
configuration uses 60 transitions over 5 model time units and truncates at the horizon.
The uncontrolled system already decays toward zero; a learned controller must improve its
transient response to beat the zero-input baseline. See the
[fixed training protocol](../reviews/initial_training_protocol.md).

## Reference

The original model was created by [Lim (1969)](https://pubs.acs.org/doi/epdf/10.1021/i260031a007).
