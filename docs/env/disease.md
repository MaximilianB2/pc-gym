# SIR vaccination benchmark

```python
from pcgym import make_env

env = make_env({"model": "disease", "N": 60, "tsim": 60})
```

The [model](../../src/pcgym/models/disease_model.py) uses population fractions and time in days:

```text
dS/dt = -beta S I - u S
dI/dt =  beta S I - gamma I
dR/dt =  gamma I + u S
```

Vaccination transfers susceptible people to the immune/recovered compartment. There is no loss
of immunity: this is **SIR**, despite the earlier SIRS label. Default transmission and recovery
rates are `beta=0.3/day` and `gamma=0.1/day`. The model is a synthetic control benchmark,
not a calibrated intervention recommendation.

The initial fractions are `[S,I,R]=[0.99,0.01,0]`. All fraction bounds are `[0,1]`, and valid
initial fractions must sum to one. Nonnegative rates preserve the population simplex;
independent initial-state uncertainty that breaks the sum is rejected. Vaccination intensity
`u` lies in `[0,0.05]/day`; the actual transfer rate is `u S`.

The 60-day default minimizes a discrete infection-burden and intervention-effort objective:

```text
reward_t = -dt * (I_(t+1) + 20 * u_t^2)
```

The benchmark effort weight is 20 day², so both terms inside the parentheses are dimensionless.
It penalizes intervention intensity rather than the number of vaccinations. This weight is a task
choice, not an epidemiological parameter. The infection term uses right-endpoint quadrature;
changing the sample time changes that approximation. Supply an explicit `custom_reward` to use
a different objective.

The observation is `[S,I,R,time_remaining]` with shape `(4,)`, and the task terminates after
60 transitions. Actions and observations are normalized by default; observations are float32.
Explicit reward configurations are preserved and do not inherit this default's finite-horizon flag.
An explicit `SP` also requires matching state/setpoint entries in the initial state and bounds.

Constant vaccination intensities 0, 0.025 and 0.05/day have nominal total costs approximately
8.93, 4.40 and 4.49, respectively. Thus zero and maximal effort are not equivalent objectives;
the intermediate policy beats both in this simple comparison. Population conservation and the
analytic no-transmission vaccination/recovery solution pass independent checks. All 14 default
CasADi action scenarios and PPO/SAC compatibility checks pass; multi-seed learning and an optimal
intervention policy are not established. Existing MPC comparisons do not support this custom objective.
