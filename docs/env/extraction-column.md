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
  <img src="..\..\img\Multistage_Extractor.png" alt="Image title" style="width:40%">
</div>
 

## Description & Equations
The multistage extraction column is a key unit operation in chemical engineering that enables mass transfer between liquid and gas phases across multiple theoretical stages, described by coupled differential equations representing the dynamic concentration changes in each phase:

\begin{align}
  \nonumber\frac{\mathrm{d}X_i}{\mathrm{d}t} &= \frac{L}{V_L}(X_{i-1}-X_i) - K_{La}\left(X_i - \frac{Y_i^p}{m}\right)\\
\end{align}
\begin{align}
  \nonumber\frac{\mathrm{d}Y_i}{\mathrm{d}t} &= \frac{G}{V_G}(Y_{i+1}-Y_i) + K_{La}\frac{V_L}{V_G}\left(X_i - \frac{Y_i^p}{m}\right)\\
\end{align}

There are five stages, with interleaved liquid/gas concentrations
$\mathbf{x} = [X_1,Y_1,\ldots,X_5,Y_5]^\intercal \in \mathbb{R}^{10}$.
The implementation uses `p = eq_exponent = 2` by default. The control vector is
$\mathbf{u} = [L,G]^\intercal \in \mathbb{R}^2$.

## Observation
The registered observation is a float32 vector of shape `(11,)`:
`[X1, Y1, X2, Y2, X3, Y3, X4, Y4, X5, Y5, X5_setpoint]`.
The target is `X5=0.4`. Observations are normalized to `[-1, 1]` by default;
the bounds below are in physical units.

The observation space is defined by the following bounds corresponding to the ordered state variables: 
```
[[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0,1],[0.3,0.4]]
```
An example, tested set of initial conditions are as follows:
```
[0.55, 0.3, 0.45, 0.25, 0.4, 0.20, 0.35, 0.15, 0.25, 0.1, 0.4]
```

## Action
The action space is a `ContinuousBox` of `[[5,10],[500,1000]]` which corresponds to a liquid phase flowrate between 5 m$^3$/hr and 500 m$^3$/hr and a gas phase flowrate between 10 m$^3$/hr and 1000 m$^3$/hr.

## Reward

The registered reward is `-(X5 - 0.4)**2`, evaluated at the next state. The initial training
configuration uses 60 transitions over 60 hours and truncates at the horizon. See the
[fixed training protocol](../reviews/initial_training_protocol.md). The public action vector
has shape `(2,)`, normalized to `[-1, 1]` by default.

## Reference

This model and its description were kindly provided by [Akhil Ahmed](https://scholar.google.com/citations?user=AS34x7cAAAAJ). The original model was created by [Ingham et. al. (2007)](https://onlinelibrary.wiley.com/doi/book/10.1002/9783527614219).
