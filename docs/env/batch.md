# Consecutive-reaction batch

```python
from pcgym import make_env

env = make_env({"model": "batch", "N": 60, "tsim": 15})
```

The [model](../../src/pcgym/models/batch.py) represents irreversible `A -> 2 B` followed by
`B -> C`. It is a synthetic, constant-volume thermal benchmark. Time is in seconds, temperature
in K, concentrations in kmol/m³, and reaction enthalpies in kJ/kmol.

With `r1 = k01 exp(-EA1/(R T)) Ca` and `r2 = k02 exp(-EA2/(R T)) Cb`, the balances are

```text
dCa/dt = -r1
dCb/dt = 2 r1 - r2
dCc/dt = r2
dT/dt  = -(dH1 r1 + dH2 r2)/(rho Cp) + UA (Tc-T)/(rho Cp V)
```

The default physical state is `[Ca, Cb, Cc, T] = [1, 0, 0, 300]`; coolant temperature `Tc`
ranges from 290 to 310 K. State bounds are `[0, 0, 0, 290]` to `[1, 2, 2, 311]`.
Parameters are `k01=1 s^-1`, `k02=0.5 s^-1`, `EA1=5000 kJ/kmol`, `EA2=6000 kJ/kmol`,
`R=8.314 kJ/(kmol K)`, `dH1=-1000`, `dH2=-1500`, `rho=1000 kg/m³`,
`Cp=4 kJ/(kg K)`, `UA=100 kJ/(s K)` and `V=1 m³`.

The invariant is `2 Ca + Cb + Cc = 2`, rather than the unweighted concentration sum. Under
adiabatic conditions, `T + Ca + 0.375 Cb = 301 K`. Complete reaction releases at most enough
energy to raise the initial material's temperature by 1 K; together with coolant at at most
310 K, this gives the nominal 311 K upper bound. It is not an equipment safety limit.

The objective maximizes `Cb` at 15 s: reward is zero until the terminal transition, then equals
`Cb`. The observation appends `time_remaining = 1-t/N`, giving shape `(5,)`. The final transition
terminates the task. Actions/observations are normalized by default; public observations are float32.

Constant 290 K and 310 K coolant yield approximately 1.13318 and 1.12548 intermediate B at
15 s. This verifies control authority, but the narrow temperature range gives modest yield
differences and the terminal reward is sparse. Multi-seed learning quality remains unqualified.
The CasADi defaults pass all 14 midpoint/corner/alternating/ten-seed scenarios and PPO/SAC
compatibility checks. The original reaction equations are unchanged.
