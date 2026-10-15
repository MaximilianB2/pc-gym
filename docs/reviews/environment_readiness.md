# Environment readiness assessment

Assessed on 2026-10-15 at commit `7cf43322fe13341672164f5665703aa57e96baa5`, using the requested
Conda `python3.14` environment (Python 3.14.6).

Follow-up: [implementation plan and training acceptance criteria](../plans/training_readiness_plan.md).

This report preserves the historical baseline. Subsequent shared-interface fixes, model repairs and
their validation are recorded in the linked plan; the model table below describes the audited revision.
Model repairs now cover biofilm/polymerization reaction laws, crystallization growth and moments,
distillation flows, heat-exchanger geometry, four-tank pump limits and three infeasible default targets.
The [latest checkpoint](../plans/training_readiness_plan.md#five-new-default-tasks) adds all five
missing default tasks and corrects reactor–separator vapour composition. All 18 controlled models
now have registered tasks, and 17 pass strict runtime checks (320 scenarios) with PPO/SAC
compatibility evidence. Biofilm still violates its default observation bounds; broader learning,
optional-backend and oracle qualification remain open.
See the [current four-task training results](initial_training_results.md) for repaired-interface
validation, passing training recipes and the learning gaps that remain.

**No environment at the audited revision earns an unconditional Gymnasium-ready recommendation.** All 19 models
that reset with the core CasADi backend fail Gymnasium's observation-space check; the twentieth
fails during integrator construction. This shared interface defect is separate from the much larger
differences in numerical and physical readiness between models.

**The best starting points are CSTR, first-order system, nonsmooth control, and multistage extraction.**
Their default tasks completed every audited action sequence with finite results and no material
observation-bound violations. Fix the shared interface issues before treating them as supported RL
environments. For the other models, the table below identifies the additional work needed.

## Scope and evidence

- Reviewed all 20 canonical entries in the [model registry](../../src/pcgym/models/__init__.py), their
  equations, default tasks/spaces, the shared environment, integration, evaluation and oracle code.
  Aliases are alternate names, not additional environments.
- Ran the existing standard suite: **155 passed, 1 skipped, 1 expected failure, 58 deselected**.
- Ran the slow suite: **32 passed, 1 failed, 25 skipped, 157 deselected**. The failure is
  [test_oracle_constraint_handling[crystallization]](../../tests/oracle/test_oracle.py#L230), with
  `CV_TOO_MUCH_WORK` during simulation. These two runs cover all collected tests.
- Ran **240 episode attempts**, of which **202 completed**, using the
  [reproducible audit script](../../scripts/audit_env_readiness.py). Each episode has 60 steps:
  midpoint inputs, every constant action-space corner, alternating all-low/all-high inputs, and
  uniform random actions with seeds 0, 1 and 2. Autonomous models have one zero-input trajectory.
- The 13 models with default spaces use their actual default task and bounds. Their time steps
  come from the [reference generator](../../scripts/gen_reference_trajectories.py). Its biofilm
  restriction to the top 20% of the input range is **not** applied in this audit.
- The other seven use the reference generator's explicit initial conditions and action bounds,
  with unbounded physical observations and an empty reward. These runs test simulation only;
  they do not establish a meaningful RL task or validate observation bounds.
- Gymnasium checks are strict. The separate numeric-bounds diagnostic ignores dtype and tolerates
  `1e-6` of rounding error, so it can distinguish the shared interface problem from substantive
  state excursions. A completed episode does not imply valid physics.

Runtime versions: NumPy 2.5.4, Gymnasium 1.3.0, CasADi 3.7.2, SciPy 1.18.1, do-mpc 5.1.2.
Diffrax 0.7.2 is installed, but **JAX is absent**, so all 25 JAX tests skip. JAX readiness and
Stable-Baselines3 training were not verified. The audit evaluates the current implementation;
it is not independent certification against the original process-model literature.
The audit script passes Python compilation and editor diagnostics. Ruff is not installed in the
selected environment, so a Ruff check could not be completed.

## Assessment of every model

“Candidate” means suitable to prioritize after the shared fixes, not ready unchanged. “Bounded”
counts completed episodes within the numeric observation bounds. A dash means the explicit
simulation fixture uses infinite bounds. Horizons are in each model's native time units.

| Model | Horizon | Completed | Bounded | Assessment and additional work |
|---|---:|---:|---:|---|
| [first_order_system](../../src/pcgym/models/first_order_system.py) | 10 | 7/7 | 7/7 | **Candidate.** Analytic solution agrees to `6.9e-10` at the tested final time. |
| [cstr](../../src/pcgym/models/cstr.py) | 26 | 7/7 | 7/7 | **Candidate.** Default operating region was stable; explicit disturbance and MPC tests also pass. |
| [nonsmooth_control](../../src/pcgym/models/nonsmooth_control.py) | 5 | 7/7 | 7/7 | **Candidate.** Stable linear plant with bounded inputs. The name does not imply nonsmooth plant equations. |
| [multistage_extraction](../../src/pcgym/models/multistage_extraction.py) | 60 | 9/9 | 9/9 | **Candidate.** Default task and full input corners run; disturbance and MPC tests pass. |
| [four_tank](../../src/pcgym/models/four_tank.py) | 1000 | 9/9 | 7/9 | **Fix bounds.** Valid high second-pump input drives `h3` above the advertised 0.6 limit, to about 0.733. |
| [distillation_column](../../src/pcgym/models/distillation_column.py) | 20 | 9/9 | 9/9 | **Fix input domain.** The allowed feed minimum 80 is below fixed distillate flow 100, giving negative bottoms flow. |
| [cstr_series_recycle](../../src/pcgym/models/cstr_series_recycle.py) | 10 | 21/21 | 21/21 | **Fix default task.** `C2=60` cannot be maintained with the current flow, temperature and kinetic ranges. |
| [multistage_extraction_reactive](../../src/pcgym/models/multistage_extraction_reactive.py) | 60 | 9/9 | 9/9 | **Fix default task.** Steady `XA5` cannot be below about 1.903, versus a target of 0.5. |
| [photobioreactor](../../src/pcgym/models/photo_production.py) | 200 | 9/9 | 9/9 | **Fix task/units.** Product target 120 is incompatible with current production rates; an optimistic upper bound is 0.105 over this horizon. |
| [heat_exchanger](../../src/pcgym/models/heat_exchanger.py) | 10 | 21/21 | 21/21 | **Fix geometry/units.** Volumes omit the factor 1/4 required by the declared diameters. All temperature derivatives are consequently four times too slow under that interpretation. |
| [biofilm_reactor](../../src/pcgym/models/biofilm_reactor.py) | 100 | 4/37 | 0/37 | **Major model fix.** Reaction-source signs contradict the documented balances; midpoint and 32 other cases fail on the first step. Even completing cases become nonphysical. |
| [crystallization](../../src/pcgym/models/crystallization.py) | 30 | 3/7 | 0/7 | **Major runtime/task fix.** Invalid initial CV bound, random-action solver failures, ineffective moment scaling, and broken disturbance support. Constrained MPC also fails. |
| [polymerisation_reactor](../../src/pcgym/models/polymerisation_reactor.py) | 2 | 21/21 | 5/21 | **Major model fix.** Finite integration hides negative monomer concentrations and extreme heating. Propagation rate does not depend on monomer concentration. |
| [batch](../../src/pcgym/models/batch.py) | 15 | 7/7 | — | **Configure before use.** Has a terminal `Cb` reward, but no default spaces or initial condition. Tested species balance is conserved. |
| [complex_cstr](../../src/pcgym/models/complex_cstr.py) | 15 | 7/7 | — | **Configure before use.** Nominal simulation works; supply bounds, initial condition, objective and verified operating region. |
| [disease](../../src/pcgym/models/disease_model.py) | 60 | 7/7 | — | **Configure before use.** Population sum is conserved. Supply a vaccination objective/bounds; equations implement SIR, despite the SIRS description. |
| [hydraulic_tank](../../src/pcgym/models/hydraulic_tank.py) | 15 | 7/7 | — | **Configure before use.** Conserves `q1+q2`; requires a target on that invariant manifold and explicit bounds/initial conditions. |
| [reactor_separator_recycle](../../src/pcgym/models/reactor_separator_recycle.py) | 30 | 37/37 | — | **Configure before use.** Tested nominal-flow region preserves composition sums. Needs a task and vessel-level constraints to keep denominator volumes positive. |
| [invariant_batch](../../src/pcgym/models/invariant_batch.py) | 3 | 1/1 | — | **Simulation only.** Conserves both reaction invariants, but exposes no control inputs or default task. A controller has nothing to actuate. |
| [coupled_oscillator](../../src/pcgym/models/coupled_oscillators.py) | 6 | 0/1 | — | **Backend fix; simulation only.** NumPy concatenation rejects CasADi symbols. No control input, defaults or task; the JAX-only route is unverified here. |

The explicit batch, disease, hydraulic, invariant-batch and recycle fixtures conserve their checked
species/population/composition balances to better than `6e-15` across these trajectories. This is
useful independent evidence, but does not replace operational bounds or task validation.

## Shared fixes needed before an RL-ready claim

### S1 — Observation dtype and normalization contract

[Space construction](../../src/pcgym/pcgym.py#L127) creates float32 boxes, while
[reset](../../src/pcgym/pcgym.py#L451) and [step](../../src/pcgym/pcgym.py#L548) generally return float64
arrays. All 19 resettable models fail `gymnasium.utils.env_checker.check_env` at reset.
There is also endpoint rounding: physical bounds are cast to float32 before normalization, so a
setpoint on a nominal bound can normalize slightly outside `[-1, 1]` (for example, CSTR's 0.9 target).

**Fix:** keep floating simulation state, make returned observation dtype match the declared space,
and normalize consistently with the chosen physical bounds. Repair actual model bounds separately;
clipping every observation would conceal the physical failures found here. Require strict
`observation_space.contains(obs)` checks at reset and throughout valid episodes.

### S2 — Setpoint observation and reward use different time indices

[Step's observation](../../src/pcgym/pcgym.py#L623) appends `SP[t]`, then increments time, while
[the reward](../../src/pcgym/pcgym.py#L811) uses `SP[t+1]`. With a sequence `[0.2, 0.4, 0.6, 0.8]`,
the first returned observation exposes 0.2 while the reward targets 0.4. Reset also preserves a
potentially inconsistent initial setpoint entry: the first-order default gives 0.4 for this explicit
sequence instead of 0.2. Constant default setpoints conceal the step-index discrepancy.

**Fix:** define one time convention across observations, reward, disturbances and the oracle;
populate the initial setpoint from the configured sequence and test transitions around changes.

### S3 — Integer initial conditions silently truncate dynamics

[Reset copies the input dtype](../../src/pcgym/pcgym.py#L474), and integration results are assigned
back into that array. With first-order `x0=np.array([0, 0])`, an explicit setpoint and a positive
input, the state remains integer zero after stepping. This is a silent simulation error.

**Fix:** validate and convert the state buffer to a floating dtype regardless of input representation.

### S4 — Initial-state uncertainty dictionaries are interpreted incorrectly

[The reset loop](../../src/pcgym/pcgym.py#L477) enumerates dictionary keys as percentages. The
existing example `{"x0": {0: 0.2}}` applies zero uncertainty, and five distinct seeds all return
the same initial value. Other indices can affect the wrong state with the wrong magnitude.

**Fix:** iterate dictionary `.items()`, validate state indices, and test variability at specified
indices independently of parameter uncertainty. Existing seeded parameter tests pass but do not
catch this defect.

### S5 — Evaluation ignores termination and breaks with parameter uncertainty

[Policy rollouts](../../src/pcgym/policy_evaluation.py#L89) allocate by `env.Nx`, which excludes the
uncertain-parameter observation entries, and ignore the returned termination flags. Reproductions:

- A trajectory terminating on its first constraint violation is still evaluated for all four steps.
- A one-parameter uncertainty configuration raises `could not broadcast input array from shape (3,)
  into shape (2,)` while storing the initial observation.

**Fix:** size rollout arrays from the full observation layout, stop at episode end, and explicitly
represent shorter trajectories so reward comparisons use valid samples.

### S6 — Advanced feature support needs explicit limits

- `noise=True` without `noise_percentage` fails at the first step with a missing
  `noise_percentage_float` attribute. Initialize a documented default or reject the incomplete config.
- Reset/step return the same mutable `info` dictionary; earlier stored results acquire later `u`
  and `obs` values. Return stable snapshots when exposing per-step information.
- [Oracle setup](../../src/pcgym/oracle.py#L77) assumes `SP` exists, so batch/custom objectives without
  setpoints are not supported. State that limitation or implement matching objectives.
- The [delta-action oracle constraints](../../src/pcgym/oracle.py#L135) have reversed signs for
  do-mpc's `g <= 0` convention: `u-low` is used for the lower bound and `high-u` for the upper bound.
  Correct to `low-u` and `u-high`, then verify feasibility. The soft constraints currently obscure
  the contradiction. This is a source-level finding; the crystallization solver failure alone does
  not establish its cause.
- [Oracle delta-mode selection](../../src/pcgym/oracle.py#L36) enables delta actions even for explicit
  `a_delta=False`. It must follow the same boolean convention as the environment.

## Model-specific blockers and remedies

### M1 — Biofilm reaction balances are reversed

[The model](../../src/pcgym/models/biofilm_reactor.py#L118) defines signed reaction sources and then
subtracts them from the flow balances. For example, `rs1=-r1` followed by `-rs1` produces ammonium
instead of consuming it; nitrate consumption and oxygen production are similarly reversed. These
signs contradict the [documented equations](../env/biofilm.md).

At default `N=60, tsim=100`, midpoint action `[0,0,0,0,0]` fails on the first integration step. At
all-high inputs the episode completes, but oxygen exceeds 500 by step 3 and nitrate becomes
negative by step 5. The absorber equilibrium also uses `O_air/m = 600`, whereas the documentation
uses `m*O_air = 150`; the action order in that documentation differs from the code as well.

**Fix:** reconcile balances, absorber equilibrium and action order against the source model; then
establish a physical operating region and regenerate reviewed references. Restricting reference
inputs to a region that avoids crashes does not validate the current reaction signs.

### M2 — Polymerization integrates outside its physical domain

[The propagation rate](../../src/pcgym/models/polymerisation_reactor.py#L68) omits dependence on `M`.
At `T=400, M=0, I=0.3` with allowed midpoint inputs, `dM/dt=-37.34`: the nonnegative concentration
domain is not preserved. The nominal 60-step trajectory ends near `T=293985` and `M=-23485.7`.
All 21 cases complete numerically, but 16 violate the stated bounds.

**Fix:** verify the intended monomer factor/rate law against the reference, check units, and add
physical-domain and runaway handling. Finite outputs alone must not count as a passing model test.

### M3 — Crystallization needs coordinated model, observation and integration work

- [Default CV](../../src/pcgym/models/crystallization.py#L9) is about 2.00833, already above its
  declared maximum 2. At midpoint temperature, mean length exceeds its bound by step 2.
- Normalizing the four moment states with upper bounds `1e20` makes all four initial moment
  observations exactly `-1` when represented as float32. Use scientifically meaningful scales,
  potentially transformed moments, and verify that relevant variation remains observable.
- Alternating inputs fail on step 30; random seeds 0, 1 and 2 fail on steps 29, 26 and 33,
  respectively, with `CV_TOO_MUCH_WORK`. These actions are inside the default space.
- [Rates use absolute supersaturation powers](../../src/pcgym/models/crystallization.py#L111),
  permitting positive growth/nucleation even below saturation. Review the intended domain and
  dissolution/undersaturation handling before choosing a numerical workaround.
- [Disturbance metadata](../../src/pcgym/models/crystallization.py#L153) advertises `ka`, `kg`, `UA`,
  but the input function uses only temperature and has no `UA` parameter. Supplying a `ka`
  disturbance raises `KeyError: 'UA'`; changing appended disturbance values leaves derivatives
  unchanged. Implement or remove that advertised support.
- Repair the constrained MPC path and require its existing failing test to pass.

### M4 — Three default objectives are incompatible with the model scales

These are stronger findings than unsuccessful random control:

- **Photobioreactor:** with nominal parameters, nonnegative nitrate and feed at most 10,
  `c_N + 504.5*c_x = 150 + 504.5 + integral(F_N dt)`. Since the light factor is at most one and
  product degradation is nonnegative, integration over 200 time units gives
  `c_q <= 0.00016 * [200*(1+150/504.5) + 10*200^2/(2*504.5)] = 0.10494`.
  The target is 120; audited final values are about 0.008–0.057. Reconcile concentration units,
  production parameters, target and reward normalization together.
- **Reactive extraction:** at any nonnegative steady state,
  `XA_i >= [L/(L+Kla*Vl)]*XA_(i-1)`. With `L>=5`, `Kla*Vl=0.05` and feed 2,
  `XA5 >= 2*(5/5.05)^5 = 1.90293`. A steady target of 0.5 is infeasible. Revise the flow/transfer
  parameters or choose a demonstrated reachable target.
- **Series CSTR with recycle:** for nominal endothermic kinetics and feed/coolant temperatures
  at most 320, the reaction constant is bounded by its value at 320. The combined steady material
  balance, `C1<=C_O`, and `F>=0.001` imply the conservative bound
  `C2 >= C_O*[1-k(320)*(V1+V2)/F_min] = 94.6713`, far above 60. Audited final values are
  approximately 96.0–97.3. Reconcile the kinetic/flow scales and select a feasible task.

### M5 — Smaller operating-domain and geometry fixes

- **Four-tank:** at maximum second-pump input the analytic upper-tank equilibrium is
  `h3=[(1-gamma_2)*k2*10/a3]^2/(2*g)`, approximately 0.736. Adjust the advertised observation
  bounds or action limits, and validate low-level behavior near the square-root domain boundary.
- **Distillation:** require feed consistent with `W=F-D>=0`, or change the formulation to support
  the intended operating regime. The current negative-bottoms cases can integrate without error.
- **Heat exchanger:** [the declared diameter parameters](../../src/pcgym/models/heat_exchanger.py#L40)
  require cylindrical/annular volumes `pi*D^2*L/4`. Current volumes omit `/4` while transfer areas
  use `pi*D*L`, which is the diameter convention. With uniform 350 K and a 360 K tube inlet,
  `dTt1/dt=3.1831` instead of 12.7324 under the declared geometry. Resolve parameter meanings,
  correct equations/units, and update the time-scale references. Its metadata also lacks the
  otherwise customary `disturbances` entry.

## Recommended completion order

1. Fix S1–S4 and add strict Gymnasium/full-episode checks for the four candidate models. These
   offer the shortest route to a useful, supportable baseline set.
2. Correct the three major model/runtime blockers: biofilm, polymerization and crystallization.
   Use balance/domain tests and validated equations before regenerating reference trajectories.
3. Repair infeasible targets, physical input/observation bounds and heat-exchanger geometry.
   Demonstrate target tracking with a simple controller or feasible open-loop solution.
4. Fix evaluation and oracle inconsistencies before relying on comparative benchmark returns.
5. Add documented tasks and operating regions to the five configuration-only controlled models;
   identify autonomous examples explicitly. Validate both integrators in a functioning JAX setup.

The existing reference tests preserve the implementation's current trajectories. Because those
references are generated by the same equations, they can preserve an incorrect physical model.
The [default-task test](../../tests/environment/test_default_reward.py#L25) likewise checks that a
target lies inside observation bounds, which does not establish control feasibility. CI should
combine regression trajectories with independent balances, strict API checks, valid-domain tests
and demonstrated task feasibility.

## Reproducing the assessment

From the repository root, using the selected environment:

```bash
/data1/conda/envs/python3.14/bin/python -m pytest -q -ra --tb=short
/data1/conda/envs/python3.14/bin/python -m pytest -q -ra --tb=short -m slow
/data1/conda/envs/python3.14/bin/python scripts/audit_env_readiness.py --output /tmp/pcgym-readiness-audit.json
```

The audit script prints per-model counts and writes detailed scenarios, state extrema, first bound
violations, exceptions, dependency versions, conservation errors and shared-bug probes. Its exit
status indicates whether evidence collection completed, not whether the environments passed.
All additional algebraic/shared-interface probes use CasADi, including when `--backend jax` is
selected for the per-model episode checks. The code and assessment are review deliverables;
no process equations or existing reference trajectories were changed during this audit.
