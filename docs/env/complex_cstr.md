# CSTR with consecutive reactions

```python
from pcgym import make_env

env = make_env({"model": "complex_cstr", "N": 60, "tsim": 15})
```

The [model](../../src/pcgym/models/complex_cstr.py) represents a stirred reactor with
`A -> 2 B` and `B -> C`, an A-only feed and coolant-temperature control. The benchmark's declared
units are seconds, K, m³ and kmol/m³. Its parameters are synthetic; these checks do not establish
experimental calibration.

```text
r1 = k01 exp(-EA1_over_R/T) Ca
r2 = k02 exp(-EA2_over_R/T) Cb
dCa/dt = (q/V)(Caf-Ca) - r1
dCb/dt = -(q/V) Cb + 2 r1 - r2
dCc/dt = -(q/V) Cc + r2
dT/dt = (q/V)(Ti-T) - (deltaHr1 r1 + deltaHr2 r2)/(rho C)
        + UA (Tc-T)/(rho C V)
```

Physical initial state `[Ca, Cb, Cc, T]` is `[0.8, 0.1, 0.1, 330]`. The coolant input ranges
from 295 to 302 K. Physical state bounds are `[0, 0, 0, 290]` to `[1, 2, 2, 400]`.
The target `Cb=0.25` is appended to the observation, giving shape `(5,)`; its bounds are `[0,2]`.
The running reward is `-(Cb-0.25)^2/4`. The nominal 15 s episode truncates after 60 transitions.
Actions/observations are normalized by default; public observations are float32.

Key parameters are `q=100 m³/s`, `V=100 m³`, `rho=1000 kg/m³`, `C=0.239 kJ/(kg K)`,
`k01=7.2e10 s^-1`, `k02=1e10 s^-1`, activation temperatures 8750 and 9000 K,
reaction enthalpies -50000 and -30000 kJ/kmol, `UA=50000 kJ/(s K)`, `Ti=350 K` and
`Caf=1 kmol/m³`. Disturbance input order is `[Tc, Ti, Caf]` when disturbances are enabled.

The combined species quantity `W=2 Ca+Cb+Cc` obeys `dW/dt=(q/V)(2 Caf-W)`.
Adding the chemical inventories `110000 Ca + 30000 Cb` to sensible energy `rho C T`
leaves only feed, outlet and jacket heat fluxes in the energy balance. Both identities are
checked independently of stored reference trajectories.

A steady solution for `Cb=0.25` has a coolant strictly inside the declared range; applying
that coolant from the registered initial state brings `Cb` within 0.002 of its target by
15 s. The default CasADi task passes all 14 midpoint/corner/alternating/ten-seed scenarios
and PPO/SAC compatibility. These bounds cover the tested nominal region; arbitrary initial
states, hot feeds or longer horizons need their own thermal-domain checks. Multi-seed learning
and disturbed tasks remain unqualified. The reaction equations are unchanged.
