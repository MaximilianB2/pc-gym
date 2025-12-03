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
The fluidized biofilm sand bed reactor is a critical unit operation in biological wastewater treatment that enables the removal of nitrogen (in the form of ammonium) through sequential oxidation reactions. The process involves a three-stage reactor system coupled with an absorber, described by coupled differential equations representing the dynamic concentration changes of multiple species:

For each reactor stage n:

\begin{align}
\frac{dS_{1n}}{dt} &= \frac{q_r}{V}(S_{1,n-1} - S_{1n}) - \frac{v_{max1}S_{1n}O_n}{(K_1 + S_{1n})(K_{O1} + O_n)} \\
\frac{dS_{2n}}{dt} &= \frac{q_r}{V}(S_{2,n-1} - S_{2n}) + \frac{v_{max1}S_{1n}O_n}{(K_1 + S_{1n})(K_{O1} + O_n)} - \frac{v_{max2}S_{2n}O_n}{(K_2 + S_{2n})(K_{O2} + O_n)} \\
\frac{dS_{3n}}{dt} &= \frac{q_r}{V}(S_{3,n-1} - S_{3n}) + \frac{v_{max2}S_{2n}O_n}{(K_2 + S_{2n})(K_{O2} + O_n)} \\
\frac{dO_n}{dt} &= \frac{q_r}{V}(O_{n-1} - O_n) - 3.5\frac{v_{max1}S_{1n}O_n}{(K_1 + S_{1n})(K_{O1} + O_n)} - 1.1\frac{v_{max2}S_{2n}O_n}{(K_2 + S_{2n})(K_{O2} + O_n)}
\end{align}


For the absorber:

\begin{align}
\frac{dS_{1A}}{dt} &= \frac{q_r}{V}(S_{13} - S_{1A}) + \frac{q}{V_A}(S_{1F} - S_{1A}) \\
\frac{dS_{2A}}{dt} &= \frac{q_r}{V}(S_{23} - S_{2A}) + \frac{q}{V_A}(S_{2F} - S_{2A}) \\
\frac{dS_{3A}}{dt} &= \frac{q_r}{V}(S_{33} - S_{3A}) + \frac{q}{V_A}(S_{3F} - S_{3A}) \\
\frac{dO_A}{dt} &= \frac{q_r}{V}(O_3 - O_A) + K_{La}(mO_{Air} - O_A)
\end{align}


Where the concentrations of ammonium ($S_1$), nitrite ($S_2$), nitrate ($S_3$), and oxygen ($O$) in each stage and the absorber are the state variables, $\mathbf{x} \in \mathbb{R}^{16}$. The action variables are the recycle flowrate, feed flowrate, and feed concentrations to the absorber, $\mathbf{u} = [q_r, q, S_{1F}, S_{2F}, S_{3F}]^\intercal \in \mathbb{R}^5$.

## Observation
The observation of the `Biofilm Reactor` environment provides information on the concentrations of the three nitrogen species at the outlet of the reactor, $\mathbf{y} = [S_{13}, S_{23}, S_{33}]^\intercal \in \mathbb{R}^3$. The observation space is defined by the following bounds:

```
[[0,10],[0,10],[0,10],[0,500],[0,10],[0,10],[0,10],[0,500],[0,10],[0,10],[0,10],[0,500],[0,10],[0,10],[0,10],[0,500],[0.9,1.1]]
```
An example, tested set of initial conditions are as follows:
```
[2,0.1,10,0.1,2,0.1,10,0.1,2,0.1,10,0.1,2,0.1,10,0.1,1]
```


## Action
The action space is a `ContinuousBox` with bounds:
```
[[0.0,10.0],    # Recycle flowrate (L/hr)
 [1.0,30.0],    # Feed flowrate (L/hr)
 [0.05,1.0],    # Feed ammonium concentration (mg/L)
 [0.05,1.0],    # Feed nitrite concentration (mg/L)
 [0.05,1.0]]    # Feed nitrate concentration (mg/L)
```

## Parameters
The system parameters are defined as follows:
```python
{
    'V': 10.0,      # Volume of Reactor Stage (L)
    'VA': 15.0,     # Volume of Absorber (L)
    'KLa': 1.5,     # Mass Transfer Constant (1/hr)
    'OAir': 300,    # Concentration of Oxygen in Air (mg/L)
    'K1': 0.5,      # Ammonia Saturation Constant for Reaction 1 (mg/L)
    'K2': 0.1,      # Ammonia Saturation Constant for Reaction 2 (mg/L)
    'KO1': 1.5,     # Oxygen Saturation Constant for Reaction 1 (mg/L)
    'KO2': 0.5,     # Oxygen Saturation Constant for Reaction 2 (mg/L)
    'vmax1': 0.8,   # Maximum Velocity Through Bed for Reaction 1 (mg/L hr)
    'vmax2': 1.0,   # Maximum Velocity Through Bed for Reaction 2 (mg/L hr)
    'm': 0.5        # Equilibrium Constant (-)
}
```

## Reference
This model and its description were kindly provided by [Akhil Ahmed](https://scholar.google.com/citations?user=AS34x7cAAAAJ). The original model was created by [Tanaka & Dunn (1982)](https://pubmed.ncbi.nlm.nih.gov/18546355/). This documentation was created by Tom Savage.