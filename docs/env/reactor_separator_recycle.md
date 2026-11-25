# Reactor–separator–recycle level control

```python
from pcgym import make_env

env = make_env({"model": "reactor_separator_recycle", "N": 60, "tsim": 30})
```

The [model](../../src/pcgym/models/reactor_separator_recycle.py) has a reactor, mixer and separator.
Reactions `A -> B -> C` occur in the reactor; a pure-A fresh feed and recycled separator vapour
enter it. C is assumed nonvolatile. This is an isothermal benchmark with consistent native
mass/length/time units, not a validated physical-plant calibration.

For each vessel, the state is `[H,x1,x2,x3]`: liquid height and composition fractions. The
full ordering is reactor, mixer, separator. Density `rho=1` and areas `A_R=A_M=A_B=10` convert
height to mass. Rate constants are `k_1=k_2=0.0167` inverse time; relative volatility is 90.
The feed is pure A; other `x1_O` values are rejected because no other feed fractions are modeled.

The five inputs `[F_O,F_R,F_M,B,D]` are fresh feed, reactor outlet, mixer outlet, bottoms and
recycled distillate, all in mass per time. Their default bounds are
`[0.9,1.8,1.8,0.9,0.9]` to `[1.1,2.2,2.2,1.1,1.1]`.

```text
rho A_R dH_R/dt = F_O + D - F_R
rho A_M dH_M/dt = F_R - F_M
rho A_B dH_B/dt = F_M - B - D
```

Species equations follow these variable-volume balances, including reactor conversion.
The corrected vapour compositions normalize only the two volatile components:

```text
y1 = alpha_1 x1_B / (alpha_1 x1_B + x2_B)
y2 = x2_B         / (alpha_1 x1_B + x2_B)
y3 = 0
```

Previously, the denominator used `1-x1_B`, counting nonvolatile C as volatile B. That could
remove B from the separator when B was absent. The corrected expression follows the existing
nonvolatile-C assumption; it is not a new ternary phase-equilibrium calibration. Pure-C separator
states are outside its domain. Independent tests check absent-species behavior and all three
component inventories, including external flows and reactor sources with unequal vessel areas.

Initial heights are `[9.5,10.5,9.5]`, and each vessel starts with fractions `[0.5,0.3,0.2]`.
The task restores all heights to 10. Height observation bounds are `[8,12]`, fraction bounds
`[0,1]`, and three height targets are appended, followed by time remaining: shape `(16,)`.
Reward is `-sum((H_i-10)^2/16)`. This finite 30-unit task terminates after 60 transitions.
Public actions/observations are normalized by default; observations are float32.

The input box bounds each height's change by 1.2 over 30 units, so the initial levels remain
inside `[8.3,11.7]`. Constant inputs `[1.1,1.9333333333,2.1,0.9333333333,1]` bring every level
to 10 at the final time. Balanced flows `[1,2,2,1,1]` then hold those levels.

Construction/reset and candidate simulations conservatively reject domains where a vessel can
empty or outgoing separator flows could exhaust the initial volatile inventory. The latter check
ignores incoming volatiles, so it can reject a longer trajectory which a less conservative analysis
might prove valid. Default flows over 40 units fail this proof; a 30-unit episode is the qualified
configuration. Initial compositions must be nonnegative and sum to one in each vessel.

All 44 default CasADi action scenarios and PPO/SAC compatibility checks pass. Multi-seed learning,
composition optimization, longer horizons and MPC on this finite task remain unqualified.
