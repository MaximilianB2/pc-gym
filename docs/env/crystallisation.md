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
crystal length in μm and concentration in g/cm³ of solution. Moment $\mu_j$ is measured in
μm$^j$ per cm³ of suspension. This implementation is a **growth-only
approximation**: growth and nucleation stop at or below saturation. It does not model
dissolution during reheating and must not be used to study dissolution or seed disappearance.

## Core Equations

New crystals enter at zero size. For a crystal of length $L$ in μm, the linear growth law is
$\dot L=v+wL$, where $v=10^4 a G_{\infty}$ (μm/min) and $w=bG_{\infty}$ (1/min). Hence

\begin{align}
\dot\mu_0 &= B_0 \\
\dot\mu_j &= j(v\mu_{j-1}+w\mu_j),\quad j=1,2,3 \\
\dot c &= -(\rho\alpha/6)10^{-12}\dot\mu_3
\end{align}

## Rate Dependencies

\begin{align}
C_{eq} &= -686.2686 + 3.579165(T+273.15) - 0.00292874(T+273.15)^2 \\
S &= c \times 10^3 - C_{eq} \\
B_0 &= k_a \exp\left(\frac{k_b}{T+273.15}\right) S^{k_c} \mu_3^{k_d}, \quad S>0 \\
G_{\infty} &= k_g \exp\left(\frac{k_1}{T+273.15}\right) S^{k_2}, \quad S>0
\end{align}

Here $C_{eq}$ and $S$ are in g/L. The implementation evaluates the same quadratic about
25 °C in g/cm³, subtracts it from $c$, then converts the excess to g/L. This avoids subtracting
large Kelvin-polynomial terms at the switching point in single precision.
Both rates are zero for $S\leq0$. The previous absolute-value extension produced crystals
even below saturation. The supersaturated equations and fitted constants are retained from
the [bundled reference model](../../pc-gym_paper/train_policies/crystalisation/control_problems/Sulf_Potass_Modelo.py).
Stopping at saturation is an explicit domain restriction; it does not establish a dissolution law.

Under the model's constant-volume approximation the closed-batch solute inventory
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
Recovering a small CV by subtracting nearly equal raw moments is ill-conditioned. Instead,
with $r=B_0/\mu_0$ and $\ell=\mu_1/\mu_0$, the solver evolves the equivalent equation
$\dot{CV}=r(1+CV^2)/(2CV)-v\,CV/\ell$. In particular, with no births and size-independent
growth, all crystals translate by the same length: $CV\,\ell$ (the standard deviation) stays constant.
After each integration interval, stepping, simulation and oracle rollouts retain CV, set
$Ln=\ell$ and reconcile $\mu_2=\mu_1\ell(1+CV^2)$. Number, first and third moments and
concentration are not projected. No variance floor or artificial nucleation is introduced.

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

## Checking a plant-model fit

Replaying the ten [published batches (Tables S2–S11)](https://doi.org/10.1021/acs.iecr.3c00739.s001)
with the original parameters gives the following physical-unit, full-trajectory $R^2$ scores.
Each replay starts at the first measured state (5 min), linearly interpolates the measured
suspension temperature, and scores each output separately at the sample times (RK45,
`rtol=1e-8`, `atol=1e-10`, maximum step 0.2 min). Measured CV and mean length are derived
from the measured moments; one negative measured variance in each of batches 8–10 is
excluded **only** from its CV score. These are source-data replays, not held-out validation.

| Across batches | Mu0 | Mu1 | Mu2 | Mu3 | Conc | CV | Ln |
|---|---:|---:|---:|---:|---:|---:|---:|
| Minimum | .873 | .902 | .901 | .891 | .944 | .378 | .481 |
| Median | .969 | .970 | .978 | .981 | .984 | .695 | .830 |
| Maximum | .992 | .994 | .993 | .997 | .991 | .878 | .922 |

For a learned model of the environment with a known initial inventory, one choice of four
independent outputs is `Mu0`, `Mu1`, `Mu3` and `CV`. Recover `Ln` and `Mu2` with the identities
above and `Conc` from the conserved inventory. This avoids recovering a small CV from noisy
raw moments. Scale outputs separately; when fitting all four raw moments instead, the source
uses $\log(\mu_j)/(12+2j)$. Exclude setpoint columns from fit scores. Split by whole batch, score active and
inactive intervals separately, and report an absolute error rather than $R^2$ for constant
targets. Use the applied absolute temperature, not a normalized action, increment or jacket
command. When fitting kinetics, $k_g$, $a$, and $b$ enter only as $k_ga$ and $k_gb$; fix
one of them rather than trying to identify all three from the same observations.

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
