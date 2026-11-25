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
The default is a synthetic, isothermal nitrification control benchmark. Its nominal CasADi task
tracks absorber nitrite at **0.25 mg N/L** over **100 hours / 60 transitions**. The repaired
reaction and oxygen balances, physical bounds and target have independent checks; all 44 tested
action scenarios and short PPO/SAC compatibility runs pass. Multi-seed learning performance and
experimental plant calibration remain separate work.

The fluidized biofilm reactor models oxidation of ammonium to nitrite and then nitrate in
three stages coupled to an absorber. These reactions change nitrogen speciation; they conserve
total nitrogen in the absence of external flows. The concentration balances are:

For each reactor stage n:

\begin{align}
\frac{dS_{1n}}{dt} &= \frac{q_r}{V}(S_{1,n-1} - S_{1n}) - \frac{v_{max1}S_{1n}O_n}{(K_1 + S_{1n})(K_{O1} + O_n)} \\
\frac{dS_{2n}}{dt} &= \frac{q_r}{V}(S_{2,n-1} - S_{2n}) + \frac{v_{max1}S_{1n}O_n}{(K_1 + S_{1n})(K_{O1} + O_n)} - \frac{v_{max2}S_{2n}O_n}{(K_2 + S_{2n})(K_{O2} + O_n)} \\
\frac{dS_{3n}}{dt} &= \frac{q_r}{V}(S_{3,n-1} - S_{3n}) + \frac{v_{max2}S_{2n}O_n}{(K_2 + S_{2n})(K_{O2} + O_n)} \\
\frac{dO_n}{dt} &= \frac{q_r}{V}(O_{n-1} - O_n) - 3.5\frac{v_{max1}S_{1n}O_n}{(K_1 + S_{1n})(K_{O1} + O_n)} - 1.1\frac{v_{max2}S_{2n}O_n}{(K_2 + S_{2n})(K_{O2} + O_n)}
\end{align}


For the well-mixed absorber, fresh feed and effluent both have flow $q$. The fresh feed has
dissolved oxygen $O_F$, zero by default:

\begin{align}
\frac{dS_{1A}}{dt} &= \frac{q_r}{V_A}(S_{13} - S_{1A}) + \frac{q}{V_A}(S_{1F} - S_{1A}) \\
\frac{dS_{2A}}{dt} &= \frac{q_r}{V_A}(S_{23} - S_{2A}) + \frac{q}{V_A}(S_{2F} - S_{2A}) \\
\frac{dS_{3A}}{dt} &= \frac{q_r}{V_A}(S_{33} - S_{3A}) + \frac{q}{V_A}(S_{3F} - S_{3A}) \\
\frac{dO_A}{dt} &= \frac{q_r}{V_A}(O_3 - O_A) + \frac{q}{V_A}(O_F-O_A) + K_{La}(O_{A,eq} - O_A)
\end{align}


The ammonium ($S_1$), nitrite ($S_2$) and nitrate ($S_3$) concentrations use **mg N/L**, so their
sum is total nitrogen. Oxygen ($O$) uses **mg O2/L**. These concentrations in each stage and
the absorber are the 16 state variables. The retained oxygen-demand factors 3.5 and 1.1 use
mg O2 per mg N converted. Inputs are $[q,q_r,S_{1F},S_{2F},S_{3F}]$, in that order.

The absorber uses its own volume $V_A$ for both flow terms, as required by the inventory balance.
The previous oxygen equation omitted the fresh-feed/effluent term. Including it gives the same
external-flow accounting as the nitrogen balances. Gas transfer is the other external oxygen source;
recycle and reaction cancel in the volume-weighted inventory of $O+3.5S_2+4.6S_3$.

## Oxygen equilibrium and assumptions

Define $m=C_{gas}/C_{water}$ for oxygen concentrations in the same mass-per-volume units.
Then $O_{A,eq}=O_{air}/m$. Multiplication would require the reciprocal definition of $m$.
The previous value $m=0.5$ gave 600 mg O2/L with the division convention and was inconsistent
with the nominal dilute-water interpretation.

[NIST's oxygen solubility compilation](https://webbook.nist.gov/cgi/cbook.cgi?ID=C7782447&Mask=10#Solubility)
lists $k_H=0.0013$ mol/(kg bar) at 298.15 K. Using liquid density 1 kg/L and ideal-gas
$R=0.08314462618$ L bar/(mol K), the benchmark now takes

\[
m=\frac{1}{k_H\rho R T}=31.03041965,\qquad
O_{A,eq}=\frac{300}{m}=9.66793242\;\mathrm{mg\ O2/L}.
\]

The fixed gas concentration of 300 mg O2/L gas is a benchmark boundary condition; no pressure,
temperature or salinity dynamics are modeled. Henry's law is linear here: the legacy
`eq_exponent` parameter is retained but values other than 1 are rejected. Positive volumes,
partition and half-saturation constants, nonnegative rates/flows/concentrations, and nonnegative
initial states are checked. These assumptions make the revised lumped benchmark explicit; they
do not reproduce a calibrated experimental reactor from the cited nitrification paper.

## Observation
The default observation includes all 16 model states followed by the `S2_A` setpoint, as a
one-dimensional float32 array of shape `(17,)`. Its order is
`[S1_1, S2_1, S3_1, O_1, ..., S1_A, S2_A, S3_A, O_A, S2_A_SP]`.
The physical upper bounds for each compartment are derived from initial/feed inventory maxima:

| State | Bounds | Reason |
|---|---|---|
| $S_1$ | [0, 2] mg N/L | Ammonium only decreases through reaction; initial/feed maximum is 2 |
| $S_2$ | [0, 2.1] mg N/L | $S_1+S_2$ decreases through nitrite oxidation; its initial/feed maximum is 2.1 |
| $S_3$ | [0, 12.1] mg N/L | Total nitrogen is conserved by reaction; its initial/feed maximum is 12.1 |
| $O$ | [0, 10] mg O2/L | Reaction consumes oxygen; initial/feed/equilibrium maximum is 9.668 |

Positive transport preserves these cumulative bounds under the default input box. They bound the
states reachable from the default initial conditions, not every independent combination in the
rectangular observation space. The setpoint slot uses [0, 2.1]. No observation clipping is used.

The physical initial conditions and the default target exposed by reset are:
```
[2,0.1,10,0.1,2,0.1,10,0.1,2,0.1,10,0.1,2,0.1,10,0.1,0.25]
```


## Action
The physical action bounds are listed below. By default actions and observations are normalized
to `[-1, 1]`; use `normalise_a=False` / `normalise_o=False` for physical units.
```
[[0.0,10.0],    # Feed flowrate (L/hr)
 [1.0,30.0],    # Recycle flowrate (L/hr)
 [0.05,1.0],    # Feed ammonium concentration (mg N/L)
 [0.05,1.0],    # Feed nitrite concentration (mg N/L)
 [0.05,1.0]]    # Feed nitrate concentration (mg N/L)
```

## Parameters
The system parameters are defined as follows:
```python
{
    'V': 10.0,      # Volume of Reactor Stage (L)
    'Va': 15.0,     # Volume of Absorber (L)
    'Kla': 1.5,     # Volumetric Oxygen Transfer Coefficient (1/hr)
    'O_air': 300,   # Gas-phase Oxygen Concentration (mg O2/L gas)
    'O_F': 0.0,     # Fresh-feed Dissolved Oxygen (mg O2/L water)
    'K1': 0.5,      # Ammonium Half-saturation Concentration (mg N/L)
    'K2': 0.1,      # Nitrite Half-saturation Concentration (mg N/L)
    'KO_1': 1.5,    # Oxygen Half-saturation Concentration for Reaction 1 (mg O2/L)
    'KO_2': 0.5,    # Oxygen Half-saturation Concentration for Reaction 2 (mg O2/L)
    'vm_1': 0.8,    # Maximum Ammonium Oxidation Rate (mg N/L/hr)
    'vm_2': 1.0,    # Maximum Nitrite Oxidation Rate (mg N/L/hr)
    'm': 31.030419650651517,  # Gas/liquid Oxygen Concentration Ratio (-)
    'eq_exponent': 1.0
}
```

## Validation and migration

The [physics regressions](../../tests/models/test_initial_task_physics.py) check reaction
stoichiometry, closed nitrogen/oxygen-equivalent inventories, open external fluxes, Henry-law
units, analytic absorber washout/aeration, and an admissible steady control reaching the target.
The 100-hour default passes all 32 action corners, midpoint, alternating extremes and ten random
seeds, including strict containment with physical and normalized observations.

CPU PPO and SAC each complete 2,048 transitions with 34 vectorized resets and identical saved-policy
reload predictions/rollouts. This establishes nominal compatibility, not multi-seed learning quality.
The [repair plan](../plans/training_readiness_plan.md#biofilm-oxygen-balance-and-default-task)
records commands and package-wide results. JAX remains unvalidated locally.

Oxygen dynamics, the nitrite target and observation normalization have changed. Re-evaluate or
retrain saved policies. The old nonlinear equilibrium option is no longer supported; custom
parameters/initial conditions require their own observation bounds and validation.

## Reference
The original implementation and description were contributed by
[Akhil Ahmed](https://scholar.google.com/citations?user=AS34x7cAAAAJ); the original documentation
was created by Tom Savage. [Tanaka & Dunn (1982)](https://pubmed.ncbi.nlm.nih.gov/18546355/)
provides the cited background on two-step biofilm nitrification and oxygen limitation. Its
abstract does not establish this 16-state lumped topology or calibrate these benchmark parameters.
The revised oxygen partition coefficient uses the separate NIST source above.
