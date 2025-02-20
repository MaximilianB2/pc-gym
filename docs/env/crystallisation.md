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

# Potassium Sulfate Crystallization Model

## Overview
Model describes K$_2$SO$_4$ crystallization using method of moments, tracking crystal size distribution moments (μ$_0$-μ$_3$) and solute concentration (c).

## Core Equations

\begin{align}
\frac{d\mu_0}{dt} &= B_0 \\
\frac{d\mu_1}{dt} &= G_{\infty} (a\mu_0 + b\mu_1 \times 10^{-4}) \times 10^4 \\
\frac{d\mu_2}{dt} &= 2G_{\infty} (a\mu_1 \times 10^{-4} + b\mu_2 \times 10^{-8}) \times 10^8 \\
\frac{d\mu_3}{dt} &= 3G_{\infty} (a\mu_2 \times 10^{-8} + b\mu_3 \times 10^{-12}) \times 10^{12} \\
\frac{dc}{dt} &= -0.5\rho\alpha G_{\infty} (a\mu_2 \times 10^{-8} + b\mu_3 \times 10^{-12})
\end{align}

## Rate Dependencies

\begin{align}
C_{eq} &= -686.2686 + 3.579165(T+273.15) - 0.00292874(T+273.15)^2 \\
S &= c \times 10^3 - C_{eq} \\
B_0 &= k_a \exp\left(\frac{k_b}{T+273.15}\right) (S^2)^{k_c/2} (\mu_3^2)^{k_d/2} \\
G_{\infty} &= k_g \exp\left(\frac{k_1}{T+273.15}\right) (S^2)^{k_2/2}
\end{align}

## Key Parameters
- Nucleation: $k_a$=0.92, $k_b$=-6800, $k_c$=0.92, $k_d$=1.3
- Growth: $k_g$=48, $k_1$=-4900, $k_2$=1.9
- Size-dependent: a=0.51, b=7.3
- Physical: α=7.5, ρ=2.7 g/cm$^3$

Temperature (T) serves as control variable to achieve desired crystal size distribution.


## Observation
The observation of the `crystallisation` environment provides information on the state variables and their associated setpoints (if they exist) at the current timestep. The observation is an array of shape `(1, 7 + N_SP)` where `N_SP` is the number of setpoints. Therefore, the observation when there a setpoint exists for $CV$ and $Ln$ is
``[mu0, mu1, mu2, mu3, conc, CV, Ln, CV_SP, Ln_SP]``.

The observation space is defined by the following bounds corresponding to the ordered state variables:
```
[[0, 1e20], [0, 1e20], [0, 1e20], [0, 1e20], [0, 0.5], [0, 2], [0, 20], [0.9, 1.1], [14, 16]]
```
An example, tested set of initial conditions are as follows:
```
[1478.01, 22995.82, 1800863.24, 248516167.94, 0.1586, 0.5, 15, 1, 15]
```


## Action
The action space is a `ContinuousBox` of `[[-1],[1]]` which corresponds to a change in cooling temperature.

## Reward

The reward is a continuous value corresponding to square error of the state and its setpoint. For multiple states, these are scaled with a factor (`r_scale`)and summed to give a single value.

## Reference

The original model was created by [de Moraes et. al. (2023)](https://pubs.acs.org/doi/10.1021/acs.iecr.3c00739).