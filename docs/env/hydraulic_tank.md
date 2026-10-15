# Conserved-volume hydraulic transfer

```python
from pcgym import make_env

env = make_env({"model": "hydraulic_tank", "N": 60, "tsim": 15})
```

The [model](../../src/pcgym/models/hydraulic_tank.py) transfers liquid between two tanks:

```text
dq1/dt = -D (q1-q2) + u
dq2/dt =  D (q1-q2) - u
```

Here `q1,q2` are volumes, `D=1` is an inverse-time transfer coefficient, and `u` is a signed
volume flow, positive towards tank 1. Values use consistent benchmark volume/time units;
no plant-specific dimensional calibration is supplied. This is a linear transfer approximation,
not a square-root pressure/head outlet law.

The default initial volumes are `[1,0.5]`, giving invariant `Q=q1+q2=1.5`. The input range is
`[-0.5,0.5]`. Both volumes have observation bounds `[0,1.5]`. The target `q1=0.9` is appended
with bounds `[0.5,1]`, giving observation shape `(3,)` and reward `-(q1-0.9)^2/1.5^2`.
Actions/observations are normalized by default; observations are float32. The 15-unit regulation
episode truncates after 60 transitions.

The controller cannot change the total volume, so the target implies `q2=0.6`.
For constant input, `q1*=(Q+u/D)/2` and

```text
q1(t) = q1* + (q1(0)-q1*) exp(-2 D t)
```

Input `u=0.3` holds the desired equilibrium and reaches it from reset. For all inputs to point
inward at both empty-tank boundaries, require `-D Q <= u <= D Q`. Construction, reset after
uncertainty sampling, and candidate simulations check this condition. Custom initial states
or transfer parameters may therefore require narrower input bounds.

The unchanged equations pass the analytic response and exact inventory checks. All 14 default
CasADi action scenarios and PPO/SAC compatibility checks pass. Feasibility and compatibility
do not yet qualify multi-seed learning performance.
