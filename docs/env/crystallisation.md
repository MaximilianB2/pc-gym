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
The `crystallization` model describes a seeded K$_2$SO$_4$ batch using population-balance
moments μ$_0$–μ$_3$ and solute concentration $c$. Time is in minutes, temperature in °C,
crystal length in μm and concentration in g/cm³. This implementation is a **growth-only
approximation**: growth and nucleation stop at or below saturation. It does not model
dissolution during reheating and must not be used to study dissolution or seed disappearance.

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
B_0 &= k_a \exp\left(\frac{k_b}{T+273.15}\right) S^{k_c} \mu_3^{k_d}, \quad S>0 \\
G_{\infty} &= k_g \exp\left(\frac{k_1}{T+273.15}\right) S^{k_2}, \quad S>0
\end{align}

Both rates are zero for $S\leq0$. The previous absolute-value extension produced crystals
even below saturation. The supersaturated equations and fitted constants are retained from
the [bundled reference model](../../pc-gym_paper/train_policies/crystalisation/control_problems/Sulf_Potass_Modelo.py).
Stopping at saturation is an explicit domain restriction; it does not establish a dissolution law.

The closed-batch solute inventory
$c + (\rho\alpha/6)10^{-12}\mu_3$ is conserved. The factor $10^{-12}$ converts μm³ to cm³.

## Key Parameters
- Nucleation: $k_a$=0.92, $k_b$=-6800, $k_c$=0.92, $k_d$=1.3
- Growth: $k_g$=48, $k_1$=-4900, $k_2$=1.9
- Size-dependent: a=0.51, b=7.3
- Physical: α=7.5, ρ=2.7 g/cm$^3$

Temperature (T) serves as control variable to achieve desired crystal size distribution.


## Observation
The default observation is a one-dimensional float32 array of shape `(9,)`, ordered
`[Mu0, Mu1, Mu2, Mu3, Conc, CV, Ln, CV_SP, Ln_SP]`.
The quality variables satisfy $CV=\sqrt{\mu_2\mu_0/\mu_1^2-1}$ and $Ln=\mu_1/\mu_0$.
They are recomputed from the physical moments after integration in stepping, simulation and
oracle rollouts. This removes numerical drift in the redundant quality states; it does not
alter or clip the physical moments.

The observation space is defined by the following bounds corresponding to the ordered state variables:
```
[[0, 1e6], [0, 1e7], [0, 5e8], [0, 3e10], [0, 0.2], [0, 3], [0, 50], [0.9, 1.1], [14, 16]]
```
The registered initial state, rounded here for display, is:
```
[1478.0098667, 22995.8230591, 1800863.2407973, 248516167.940593, 0.15861523304, 2.0083289, 15.55863975, 1, 15]
```

Use registered defaults or compute `CV` and `Ln` from the exact moments when supplying `x0`.
Initial moments must be nonnegative and realizable, with positive number, mean and variance;
inconsistent quality entries are rejected. Independent uncertainty draws that break these
relationships are also rejected. The bounds above cover the tested nominal 30-minute task;
they are not a guarantee for arbitrary horizons, seeds or kinetic parameter overrides.
They replace the old `1e20` moment scales that erased moment changes in float32 observations.


## Action
The physical action is **absolute temperature**, bounded by `[10, 40]` °C. Public actions and
observations are normalized to `[-1, 1]` by default. Use `normalise_a=False` or `normalise_o=False`
for physical units. Temperature increments require explicit `a_delta=True`, an initial action
`a_0`, increment bounds in `a_space`, and absolute bounds in `a_space_act`.

Time-varying `ka` and `kg` disturbances are supported. Their full model input order is
`[Tc, ka, kg]`; unspecified rates use nominal parameters. The old `UA` disturbance entry was
removed because there is no heat-transfer state or `UA` parameter in this temperature-driven model.

## Reward

The registered tracking reward is the negative sum of squared errors for `CV=1` and `Ln=15 μm`,
scaled by the respective physical observation-range widths. A nominal four-hold temperature
schedule `[28.0767, 34.8776, 32.6475, 21.8005]` °C, with 7.5 minutes per hold, reaches
approximately `CV=1.0131`, `Ln=15.1168 μm` after 30 minutes. This is a feasible near-target batch
trajectory, not a steady operating point or evidence of learned control.

The nominal task completes midpoint, both temperature extremes, alternating inputs and ten
random-temperature episodes with finite states inside the declared bounds. RL learning quality,
dissolution physics and optional JAX operation are not established by those checks.

## Reference

The original model was created by [de Moraes et. al. (2023)](https://pubs.acs.org/doi/10.1021/acs.iecr.3c00739).
