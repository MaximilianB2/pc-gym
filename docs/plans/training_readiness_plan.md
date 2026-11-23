# Plan: make PC-Gym environments usable for training

Status: implementation in progress. The shared CasADi contract and evaluator are repaired.
Sixteen model implementations now have equation, domain or default-task corrections. All 19 controlled
models have complete registered tasks; all 19 nominal defaults pass strict runtime audits and have PPO/SAC
compatibility evidence. Biofilm now has an explicit Henry-law benchmark parameterization and corrected
oxygen flow accounting. Learning qualification and optional features remain open; experimental plant
calibration is not established by these synthetic benchmarks. Both autonomous models now support
CasADi simulation and are explicitly classified as simulation-only.
Branch: `fix/training-ready-environments`.
Base: audited `dev` revision `7cf43322fe13341672164f5665703aa57e96baa5`.
Local validation environment: Conda `python3.14` (Python 3.14.6).

The [readiness assessment](../reviews/environment_readiness.md) records the starting evidence and
reproductions. The branch starts from `dev` because it contains the registry, default tasks and
other changes being repaired; local `main` is 17 commits behind that revision.

## Goal and first deliverable

Make the 18 models with control inputs usable as documented, tested training tasks. Retain the two
autonomous systems as simulation examples unless a scientifically justified actuator is designed.
Readiness is a claim about a specific task, configuration and backend, not every possible combination
of model parameters and optional features.

The first deliverable is a working CasADi training path for
[first-order system](../../src/pcgym/models/first_order_system.py),
[CSTR](../../src/pcgym/models/cstr.py),
[nonsmooth control](../../src/pcgym/models/nonsmooth_control.py), and
[multistage extraction](../../src/pcgym/models/multistage_extraction.py).
Use Stable-Baselines3 PPO for the initial integration checks, consistent with the existing
[training guide](../guides/training.md); exercise SAC as a second consumer of the continuous-action
interface. A short successful training run establishes compatibility, not learning quality.

## Definition of training-ready

A task is promoted only when the following evidence exists:

| Requirement | Acceptance evidence |
|---|---|
| Valid Gymnasium interface | Gymnasium and SB3 environment checkers pass; reset/step observations have the declared shape and dtype and satisfy strict space containment. Check normalized and physical-unit configurations. |
| Correct transitions | Floating simulation state, consistent schedule indices, correct termination/truncation, and reproducible seeded noise/uncertainty. Observations and saved information remain stable after later steps. |
| Valid physics | Independent balances, analytic solutions or a documented reference support the equations. Valid actions do not silently produce negative concentrations, invalid fractions, nonpositive vessel volumes or other model-specific domain errors. |
| Reliable operating region | Full episodes from documented initial conditions pass midpoint, every action corner, alternating inputs and at least ten random-action seeds. Intentional constraint termination is tested separately from solver failure. |
| Meaningful control task | Units, input limits, observation scaling, sample time, horizon, reward and a reachable target/yield objective are documented. A reference controller or feasible input sequence demonstrates the target under those limits. |
| Training compatibility | CPU PPO and SAC smoke runs complete, including vectorized auto-reset, finite updates, save/load and deterministic prediction. First deliverable: at least 2,048 transitions per algorithm per candidate model. |
| Learning evidence | At least three training seeds, evaluated on ten matched episodes per seed, with a fixed configuration and budget. Compare random, midpoint and a simple reference controller; report tracking/yield, constraint violations, failures and return. Meet task-specific criteria declared before training. |
| Accurate support claim | The recorded backend and optional features have actually been tested. A skipped dependency check does not establish support. |

For noisy observations, define a truthful space for the noisy measurement distribution. Gaussian
measurement noise has unbounded support; do not require it to remain in a finite physical-state box
without an explicitly documented bounded-noise or saturation model. Keep the nominal bounded task
and any noisy/POMDP variants distinct in their validation records.

## Milestones and dependencies

| ID | Deliverable | Depends on | Completion gate | State |
|---|---|---|---|---|
| P0 | Reproducible acceptance fixtures and failure regressions | Existing audit | Candidate cases and known failures reproduce with stated units/horizons | Complete for the current repaired task set: regressions and strict audit mode implemented |
| P1 | Shared environment contract | P0 | Four candidate tasks pass strict API and full-episode tests | Complete for tested CasADi configurations: observations, schedules, episodes, randomness and validation |
| P2 | Trustworthy evaluation and first training examples | P1 | Four candidates pass PPO/SAC compatibility and learning checks | Evaluator and initial experiment complete; all-task learning gate remains open |
| P3 | Repair six otherwise runnable default tasks | P1 | Each corrected task meets the readiness definition | Equation/domain/task repairs and compatibility checks complete; learning qualification open |
| P4 | Repair three major physics/runtime failures | P1; source-model verification | Independent physics tests and training gates pass per model | All three benchmark repairs and compatibility checks complete; biofilm assumptions documented; learning qualification open |
| P5 | Supply tasks for five configuration-only models | P1; relevant physics checks | Each has complete defaults and a demonstrated control objective | Task definitions, physics/domain checks and PPO/SAC compatibility complete; learning gate open |
| P6 | Validate optional backends/features and classify autonomous models | P1; relevant model fixes | Explicit per-model support matrix and passing feature tests | Autonomous CasADi repairs/classification complete; oracle delta, JAX and feature combinations open |
| P7 | Consolidate CI, documentation and migration evidence | P2–P6 | Supported cases pass their required checks; readiness report updated | Planned |

### P0 — Convert the audit into acceptance tests

- Keep the dated [assessment](../reviews/environment_readiness.md) as the baseline; add subsequent
  measurements with the configuration and revision used.
- Reuse [the audit scenarios](../../scripts/audit_env_readiness.py), but turn specific reproduced
  defects into focused regression tests. The audit's default exit means evidence collection
  finished; `--strict` now fails on missing tasks/defaults, checker or episode failures, nonfinite
  values, or strict observation-space violations. Physical validity and learning remain separate gates.
- Define canonical test cases from registered defaults plus explicit `N`/`tsim`. Record backend,
  task, units and physical invariants. Avoid a second divergent table of default state/action values.
- Correct invalid legacy fixtures when adding validation. For example,
  [model tests](../../tests/models/test_model.py) currently use a one-dimensional action space for
  the five-input biofilm model; passing such a fixture does not demonstrate a valid environment.
- Add focused tests for the dtype, integer-state, schedule, uncertainty, noise and evaluator defects
  listed as S1–S6 in the assessment. Introduce each failing regression with its corresponding fix.

Exit: independently reproducible failures and a small, fast acceptance suite for the first four tasks.

### P1 — Fix the shared Gymnasium contract

Primary files: [pcgym.py](../../src/pcgym/pcgym.py), [integrator.py](../../src/pcgym/integrator.py),
[environment tests](../../tests/environment/).

1. **State, observation and action types.** Convert initial state to a floating simulation buffer.
   Keep integration precision separate from the public observation dtype. Build one observation
   path shared by reset/step, respecting partial observations, optional appended values and dtype.
   Preserve physical normalization bounds at sufficient precision; fix endpoint rounding without
   clipping material state excursions into the declared box. Validate action dimensions against
   model inputs rather than relying on scalar broadcasting.
2. **Schedule semantics.** Adopt one documented state-time convention: reset exposes `x_0`,
   `SP[0]`, and `d[0]`; transition `t` integrates with `u_t` and the disturbance `d[t]`; the returned
   observation exposes `x_(t+1)`, `SP[t+1]`, and `d[t+1]`, and its reward uses that same target.
   Continue holding the final schedule value where only `N` values are provided. Test changing
   setpoints/disturbances at both interior and final boundaries; update oracle scheduling and
   examples with the same convention. This intentionally corrects the current disturbance offset.
3. **Episode semantics.** Preserve exactly `N` transitions for nominal regulation. A continuing-task
   time limit truncates; physical terminal conditions terminate. Define a finite-horizon batch task
   as terminating at its objective horizon and expose time remaining for that task so its policy
   can distinguish otherwise identical states at different stages. Cover terminal observations and
   bootstrapping with vectorized training tests; document any changed batch behavior.
4. **Randomness and noise.** Fix dictionary initial-state uncertainty, including nonzero state indices;
   validate percentages and parameter bounds. Initialize or validate missing noise configuration.
   Same seed plus the same actions must reproduce a trajectory; different seeds must affect only
   configured random quantities. Test combined noise, disturbances and uncertainty.
5. **Validation and state ownership.** Validate positive `N`/`tsim`, finite initial states, ordered
   bounds, nonzero normalization widths, supported schedule/state names and input dimensions.
   Return independent observation/info snapshots. Integrate into a temporary result and validate
   it before replacing the episode state; preserve useful failure diagnostics.

Unexpected solver failure remains an acceptance-test failure. A task may intentionally terminate
at a physical constraint, but suppressing an integration exception or fabricating a successful
transition cannot qualify the model as reliable. Any policy-facing failure penalty must be explicit
and tested against the incentive to end an episode early.

Exit: strict Gymnasium checks and complete action tests pass for all four initial candidates; existing
simulation, custom-model, delta-action, layout and seed behavior remains covered.

### P2 — Make evaluation trustworthy and train the initial four tasks

Primary files: [policy_evaluation.py](../../src/pcgym/policy_evaluation.py),
[evaluation tests](../../tests/policy_evaluation/), [training guide](../guides/training.md), and new
small training examples/tests.

- Allocate rollout observations using the complete declared layout, including uncertainty entries.
  Stop on termination/truncation and store an explicit valid-length/mask for shorter trajectories.
  Metrics and plots must exclude padding; uncertainty and early termination must work together.
- Keep state/measurement, physical action, reward and time axes aligned. Preserve applied absolute
  actions when a policy requests action increments. Compare evaluator results to direct stepping.
- Correct batch/custom-reward handling in evaluation; check terminal rewards and final-state data.
  Use matched seeds and scenarios for comparative evaluation so policies see the same perturbations.
- Add a small reproducible PPO example first, then SAC checks. Configure CPU, fixed seeds and explicit
  hyperparameters/budgets. Exercise vectorized resets, model save/load and prediction. Install the
  required training dependencies into the selected Python environment when implementation begins.
- The training guide's model name, obsolete class reference, reward description and integer
  training-step count were corrected alongside schedule documentation. Its PPO workflow still
  needs execution and learning validation in this milestone.
- Run the readiness definition's learning evaluation. Store configuration, package versions, learning
  curves and per-seed metrics. Report normalized tracking error or yield and constraint/failure rates
  alongside return; do not claim cross-model comparability from reward scaling alone.

Exit: the four initial candidates have reproducible training examples and measured learning results.
Passing them does not promote untested noise, oracle or backend variants.

### P3 — Correct default domains, objectives and geometry

Repair each task with its own focused tests and documentation. Preserve defensible physics; prefer
a feasible task over altering physical parameters solely to make an arbitrary target reachable.

| Model | Implementation work | Independent acceptance evidence |
|---|---|---|
| [four_tank](../../src/pcgym/models/four_tank.py) | Reconcile pump limits, tank-height bounds and domain handling near empty tanks | Analytic equilibria at input extremes; nonnegative levels; full episodes contained in the declared space |
| [distillation_column](../../src/pcgym/models/distillation_column.py) | Require physically consistent feed/distillate/bottoms flows; replace infeasible `X0=0.9` objective | `W=F-D>=0`, composition bounds, stable low/high-flow trajectories and feasible purity tracking. With nonnegative bottoms, the total light-component balance gives `D*X0 <= F*X_feed <= 60`, hence `X0 <= 0.6` for `D=100`. |
| [cstr_series_recycle](../../src/pcgym/models/cstr_series_recycle.py) | Verify units/kinetics and replace the infeasible `C2=60` task with a justified configuration | Steady material/energy balance and a feasible controlled operating point, distinct from reset |
| [multistage_extraction_reactive](../../src/pcgym/models/multistage_extraction_reactive.py) | Verify transfer/reaction units and specify a reachable extraction target | Stage balances and reachable range consistent with the analytic steady lower bound |
| [photobioreactor](../../src/pcgym/models/photo_production.py) | Reconcile product concentration units, scaling, horizon and objective | Nutrient balance and production upper bound agree with the selected target; useful reward sensitivity |
| [heat_exchanger](../../src/pcgym/models/heat_exchanger.py) | Resolve diameter/volume conventions, correct geometry and metadata, revisit sample time | Correct geometric volumes, heat balance, temperature bounds and analytic/local time-scale checks |

Exit: all six tasks satisfy the common API, domain, feasibility and training gates with their corrected
configurations. Update reference trajectories only for explained dynamics changes.

### P4 — Repair the three major model/runtime blockers

| Model | Implementation sequence | Required evidence |
|---|---|---|
| [biofilm_reactor](../../src/pcgym/models/biofilm_reactor.py) | Verify the cited balances; correct signed reaction terms; reconcile absorber equilibrium and action order; derive a physical operating region | Reaction-only consumption/production signs, nitrogen/oxygen balances, nonnegative substrates and robust midpoint/random/full-corner episodes. Remove the reference generator's top-20%-input workaround once its admitted domain is validated. |
| [polymerisation_reactor](../../src/pcgym/models/polymerisation_reactor.py) | Verify the propagation law and monomer dependence; correct units; define thermal/positivity limits and a controllable task | No monomer consumption at zero monomer without an explicitly justified mechanism; positive-domain invariance and energy balance; thermal behavior remains physical or reaches a documented terminal condition |
| [crystallization](../../src/pcgym/models/crystallization.py) | Verify saturation/dissolution physics; use consistent moment-derived observables and useful scaling; correct initial bounds; implement or remove advertised disturbances; choose integration settings for the validated equations | Positive moments and concentration, consistency of CV/mean length with moments, physically meaningful behavior at saturation, observable moment changes in float32, random-input robustness, and passing constrained MPC after oracle fixes |

For each model, record the source equation, implementation difference and units before changing it.
When a source interpretation is unresolved, keep that model's readiness status provisional while
continuing independent fixes. Test solver tolerances/step sizes after the equations are correct;
larger solver budgets alone are not a repair of invalid physics. Regenerate only affected reference
trajectories and explain their changes alongside independent tests.

Exit: each model independently passes the readiness gates; an unresolved model does not delay
documenting successful results for the others.

### P5 — Add complete tasks to five controlled models

Use the existing [registration/default-task mechanism](../../src/pcgym/models/_registry.py) and
[model contribution checklist](../guides/adding_a_model.md).

| Model | Task to establish | Configuration requirements |
|---|---|---|
| [batch](../../src/pcgym/models/batch.py) | Finite-horizon intermediate-product yield | Register physical initial state, temperature bounds and horizon; preserve the stoichiometric balance and terminal objective; expose time remaining |
| [complex_cstr](../../src/pcgym/models/complex_cstr.py) | Reachable concentration/product regulation | Verify the coupled reaction operating point and thermal bounds; demonstrate reachable target and coolant authority |
| [disease](../../src/pcgym/models/disease_model.py) | Infection reduction with an explicit vaccination-effort tradeoff | Document SIR semantics, population-fraction bounds, vaccination limits, horizon and objective; preserve total population |
| [hydraulic_tank](../../src/pcgym/models/hydraulic_tank.py) | Transfer/regulation on the invariant total-volume manifold | Choose a feasible target, nonnegative state region and admissible transfer inputs; document conserved total |
| [reactor_separator_recycle](../../src/pcgym/models/reactor_separator_recycle.py) | Reachable composition/level control | Establish balanced nominal flows, positive vessel volumes, valid compositions, constrained action domain and a reference controller |

Choose exact targets and limits from the validated operating points during implementation, then
fix the evaluation criteria before training. Add a units/equations/task page for each model.

Exit: all five have a complete, meaningful minimal task and the same API/physics/training evidence
as the original default models. Empty-reward reference fixtures remain simulation tests only.

### P6 — Validate optional support and classify autonomous examples

- **Oracle:** correct delta-mode selection and constraint signs in [oracle.py](../../src/pcgym/oracle.py).
  Align disturbance/setpoint scheduling with P1, including changing forecasts within the MPC horizon.
  Make the objective match the configured supported reward; reject unsupported batch/custom reward
  configurations clearly rather than silently solving a different task. Validate action bounds,
  solver feasibility and terminal conditions, beyond checking returned array shapes.
- **JAX:** establish a functioning optional dependency set in the requested Python environment before
  claiming local validation. If an upstream dependency does not support it, record that limitation;
  a supported existing CI runtime can establish a separate backend/runtime result. Test required
  floating precision, finite gradients where applicable, parameter changes after reset, representative
  full episodes and trajectory agreement using documented per-state tolerances. Preserve the core
  CasADi install's independence from JAX and do-mpc.
- **Feature combinations:** exercise the documented combinations of normalization, partial observation,
  noise, uncertainty, disturbances, constraints and delta actions on representative valid models.
  Model-specific unsupported features must be rejected early and listed explicitly.
- **Autonomous examples:** mark [invariant_batch](../../src/pcgym/models/invariant_batch.py) and
  [coupled_oscillator](../../src/pcgym/models/coupled_oscillators.py) as simulation-only. Repair the
  oscillator's symbolic concatenation and validate its conserved quantities/backend agreement.
  Do not manufacture ignored action dimensions or claim trainability. A future controlled version
  requires an explicit actuator, objective and separate physical validation.

Exit: an evidence-backed support matrix by model, backend, runtime and optional feature, including
clear unsupported/simulation-only entries.

### P7 — Make the results maintainable

- Add fast API/domain/regression gates to [PR CI](../../.github/workflows/ci.yml) as models are promoted.
  Use ordinary failing assertions for required supported behavior; do not hide regressions with broad
  expected failures or dependency skips. Optional-feature jobs must assert their dependencies exist.
- Put longer multi-seed learning, complete boundary sweeps and expensive MPC/JAX validation in the
  [full-suite workflow](../../.github/workflows/nightly.yml). Keep stochastic performance criteria
  separate from deterministic PR checks. Record budgets and tolerances before measuring outcomes.
- Preserve the declared Python minimum and existing CI coverage. Using Python 3.14 locally does not
  by itself expand the package's formal tested-runtime claim; document the results actually obtained.
- Update model pages, default-task/training/evaluation guides and examples as each change lands.
  Document changes to dynamics, target units, scaling, action ranges, scheduling and horizon semantics;
  saved policies relying on old observations or dynamics may need retraining.
- Publish a new readiness assessment with test commands, exact cases, results and remaining limits.
  Keep the original assessment as historical evidence. Require the currently failing crystallization
  MPC test to pass before claiming the full existing suite is healthy.

## Reviewable implementation sequence

The initial implementation was recorded in four dependency-ordered commit groups:

1. Shared environment and evaluation contracts, including registry support and focused regressions.
2. Model dynamics, physical-domain validation and default tasks, with physics checks and updated
   reference trajectories alongside the equation changes that explain them.
3. Reusable environment audits and PPO/SAC training checks.
4. Model and API documentation, the readiness assessment, training evidence and this plan.

Runtime and evaluation changes stay together because their schedule and rollout APIs depend on
each other. The pre-existing editor-settings change is excluded from these commits.

The model follow-ups use three additional groups: biofilm oxygen physics and default-task repairs
with their regressions and model documentation; autonomous CasADi support and simulation-only
classification with their checks and contributor guidance; and the combined readiness and
default-task documentation.

The chronological checkpoints below record the shared-contract repairs, the initial training
experiment and subsequent model repairs. Earlier checkpoint counts and outstanding-work lists
describe that point in development; the milestone table and final checkpoint describe current status.

## First slice results — 2026-10-15

Implemented in [pcgym.py](../../src/pcgym/pcgym.py):

- Preserve float64 simulation state, including when `x0` is an integer array or list.
- Keep physical normalization bounds in float64 through both disturbance and uncertainty extensions.
- Share reset/step observation formatting; return float32 arrays matching the public space.
- Keep unmasked measurements at float64 precision for rewards and evaluation. Return independent
  observation/info snapshots, including constraint histories. Preserve genuine out-of-bounds values.

The [observation guide](../guides/observations.md) documents this behavior and the compatibility
change for callers that previously received float64 observations despite a declared float32 space.

The new [regression suite](../../tests/environment/test_observation_contract.py) initially produced
128 failures and 2 passes against the audited implementation. All 130 tests pass after the fix.
These include strict Gymnasium checks, decimal-bound endpoints, appended bounds, independent
snapshots, visible domain excursions and an analytic first-order response from integer `x0`.

Full-episode tests use registered defaults, CasADi and 60 transitions. Each observation mode is
checked at the action midpoint, every action corner, alternating extremes and ten random-action
seeds. Every observation must have the exact declared dtype and satisfy `Box.contains`, with
finite states/rewards and truncation exactly at the horizon.

| Model | `tsim` (model time units) | Gymnasium checker, normalized / physical | Full episodes, both observation modes |
|---|---|---|---|
| `first_order_system` | 10 | Pass / Pass | 28 / 28 |
| `cstr` | 26 | Pass / Pass | 28 / 28 |
| `nonsmooth_control` | 5 | Pass / Pass | 28 / 28 |
| `multistage_extraction` | 60 | Pass / Pass | 32 / 32 |

Validation on Conda `python3.14` (Python 3.14.6, NumPy 2.5.4, Gymnasium 1.3.0, CasADi 3.7.2),
using the uncommitted branch changes over the base revision recorded above:

```bash
/data1/conda/envs/python3.14/bin/python -m pytest tests/environment/test_observation_contract.py tests/environment/test_observation_layout.py tests/environment/test_make_env_x0_validation.py tests/environment/test_simulate.py tests/policy_evaluation/test_rollout_units.py -q -ra --tb=short
# 154 passed, 1 deselected

/data1/conda/envs/python3.14/bin/python -m pytest -q -ra --tb=short
# 285 passed, 1 skipped, 1 xfailed, 58 deselected

/data1/conda/envs/python3.14/bin/python scripts/audit_env_readiness.py first_order_system cstr nonsmooth_control multistage_extraction --seeds 10 --output /tmp/pcgym-observation-audit.json
# 58/58 normalized episodes completed and numerically bounded; all four checkers passed
```

The selected audit confirms the runtime improvement; its exit status is still evidence collection,
not a readiness gate. The strict acceptance assertions are in the regression suite. The skipped
and expected-failure standard tests remain the known coupled-oscillator CasADi limitations.

At this first checkpoint, schedule alignment, noise/uncertainty fixes, remaining P1 validation,
evaluator repairs and PPO/SAC learning evidence were still pending. The slow suite was not rerun
for the first slice; its baseline crystallization MPC failure remained unresolved.

## Second slice results — 2026-10-15

Implemented schedule alignment in [pcgym.py](../../src/pcgym/pcgym.py) and
[oracle.py](../../src/pcgym/oracle.py):

- Reset exposes `SP[0]` even when `x0` contains a different setpoint placeholder. The input
  configuration is preserved.
- Transition `t` integrates with `d[t]`. Its observation contains `x_(t+1)`, `SP[t+1]` and
  `d[t+1]`; its tracking reward uses that same target. Custom rewards receive the next
  measurement and the physical input vector applied during the completed interval.
- Both the environment and MPC hold the final supplied schedule value. This covers constant,
  `N`-value and `N+1`-value schedules, including the final observation and forecast lookahead.
- MPC setpoints and disturbances now use do-mpc time-varying parameters, with an independent
  value at each predicted stage. Stage `k` uses `SP[t+k]` and integrates with `d[t+k]`.
  The oracle simulator and logged disturbances use the current interval's value.

The [environment regressions](../../tests/environment/test_schedule_alignment.py) and
[oracle regressions](../../tests/oracle/test_oracle_schedules.py) initially produced 34 failures
and 8 passes. All 42 now pass. They cover multiple setpoints, disturbance subsets and ordering,
reset reproducibility, normalized/physical observations, custom rewards, terminal values,
analytic integration, per-stage forecasts and an MPC problem with known optimal controls.
Existing oracle rollout/repetition checks also pass. The observation layout fixture now declares
physical actions, matching the coolant temperatures it supplies.

A CSTR oracle trajectory was replayed through the environment with changing disturbances.
An independent SciPy solve (`rtol=1e-11`, `atol=1e-12`) confirmed that residual differences are
from the two CVODES tolerance settings: maximum environment error was approximately `1.35e-5`
in concentration and `8.6e-4 K` over one minute. The replay test allows `2e-5` and `0.002 K`,
respectively; solver settings and model equations were not changed.

Validation used the same Conda Python 3.14 environment, with do-mpc 5.1.2:

```bash
/data1/conda/envs/python3.14/bin/python -m pytest -q -ra --tb=short
# 316 passed, 1 skipped, 1 xfailed, 69 deselected

/data1/conda/envs/python3.14/bin/python -m pytest -q -ra --tb=short -m slow
# 43 passed, 1 failed, 25 skipped, 318 deselected
```

The sole slow failure remains
`tests/oracle/test_oracle.py::test_oracle_constraint_handling[crystallization]`
with `CV_TOO_MUCH_WORK`. All 25 slow skips require the unavailable JAX backend; the standard
skip/expected failure remain the coupled-oscillator CasADi limitations. The corrected observation
layout fixture was also rerun separately after the standard suite: 8 passed.
Ruff 0.17.0 was installed in the selected Conda environment; lint and formatting checks pass
for all nine changed Python files. Its Conda package exposes the executable
`/data1/conda/envs/python3.14/bin/ruff` rather than a `python -m ruff` module.

The [observation](../guides/observations.md#schedule-timing), [disturbance](../guides/disturbances.md)
and [training](../guides/training.md) guides document the timing change. The disturbance example
now uses a feasible 345 K value and a schedule of exactly `nsteps` values, including for odd lengths.
Basic stale API references in those examples were corrected; PPO itself has not been run.

Remaining P1 work includes batch termination/time observations, noise and initial-state uncertainty,
action/configuration validation and solver-state validation. The evaluator, delta-action oracle
constraints, model repairs, JAX support and PPO/SAC training evidence remain open. These results
establish schedule correctness for the tested cases, not complete training readiness.

## Third slice results — 2026-10-15

Completed the remaining shared-contract work in [pcgym.py](../../src/pcgym/pcgym.py):

- A nonempty built-in batch objective terminates at `N` and appends `time_remaining = 1 - t/N`
  after the other observation entries. Normalization maps this fraction from `[0, 1]` to
  `[-1, 1]`. Custom objectives can opt in with `finite_horizon=True`. Regulation and explicit
  empty-reward simulations still truncate. Constraint termination takes precedence.
- Stepping requires an active episode. Reset clears previous actions and custom information.
  Stable-Baselines3 `DummyVecEnv` tests verify the final observation, automatic reset, and
  `TimeLimit.truncated` flag for both batch and regulation tasks.
- Dictionary initial-state uncertainty now samples the requested indices, including nonzero
  indices. Empirical initial-state and parameter sampling is supported without adding a false
  `x0` parameter slot. `Nx` includes the full observation layout, and `NUn` counts parameters.
- Noise accepts finite nonnegative Python/NumPy numbers or dictionaries of state names. Enabling
  noise requires an explicit percentage; reset and step use the same measurement rule. Gaussian
  measurements and normally distributed uncertain entries have unbounded public spaces, while
  finite physical bounds are retained for normalization. Masked entries have public bounds `[0, 0]`.
- Construction validates horizons, finite initial states, bounds and normalization widths, input
  dimensions, schedule/state names, percentages, distributions and sampling support. Uniform and
  empirical parameter bounds must contain the full support. Actions must have the declared vector
  shape, contain finite values and lie inside the public bounds; delta inputs retain absolute saturation.
- Integration uses temporary state/input copies. Exceptions, malformed results and nonfinite states
  raise an error with the backend, model, step and inputs, preserving the previous episode state,
  time, action and info. Tests exercise both adapter paths; the JAX adapter uses a stub and does
  not establish working JAX integration.

The new [training-contract regressions](../../tests/environment/test_training_contract.py) initially
produced **63 failures and 1 pass**. The final suite contains **90 passing cases**, including the
combined batch/noise/disturbance/initial-state/parameter configuration, Gymnasium checks and
Stable-Baselines3 vectorized episode handling. The earlier 130 observation-contract cases still pass,
including complete midpoint/corner/alternating/ten-seed episodes for all four initial candidates.

Validation uses the selected Conda Python 3.14.6 environment; Stable-Baselines3 2.9.0 imports and
runs the vectorized checks. The standard suite passes **406 tests**, with one existing
coupled-oscillator skip, one expected failure and 69 slow tests deselected:

```bash
/data1/conda/envs/python3.14/bin/python -m pytest -q -ra --tb=short
```

The first slow run exposed a tuple-valued `a_delta` and scalar `a_0` in the crystallization
fixture, plus CSTR MPC actions up to `3.0194e-6 K` above the physical upper bound. The fixture
now uses a boolean and vector. [The oracle](../../src/pcgym/oracle.py) sets IPOPT's
`bound_relax_factor=0`, keeping returned controls within the declared action space. Both affected
tests pass after these corrections, including the formerly failing crystallization constraint case.
This single regression does not establish crystallization physics or delta-mode oracle correctness;
the underlying model and nonlinear delta-constraint repairs remain in P4/P6.

The final slow run passes **44 tests**, with **25 JAX-dependent skips** and no failures:

```bash
/data1/conda/envs/python3.14/bin/python -m pytest -q -ra --tb=short -m slow
```

JAX remains unavailable in the selected environment. Ruff lint and formatting checks pass for
all 14 changed Python files; editor diagnostics and `git diff --check` are clean.

Other corrected fixtures include the five-input biofilm action shape, physical-unit uncertainty
bounds, and the delta-saturation test's valid increments. The [observation guide](../guides/observations.md)
and [training guide](../guides/training.md) document the API changes. Existing batch policies need
the new time entry, and old seeded noisy trajectories change because reset now draws measurement noise.

Next: P2 evaluation of early termination, batch/custom rewards and matched scenarios, followed by
PPO/SAC training and learning comparisons for the first four tasks. No PPO/SAC training or learning
quality claim is made by the vectorized interface tests. Strict audit mode, model repairs, optional
backends and the remaining oracle features are still open.

## Fourth slice results — 2026-10-15

The [training report](../reviews/initial_training_results.md),
[predeclared protocol](../reviews/initial_training_protocol.md),
[episode data](../reviews/initial_training_results.json) and
[learning curves](../reviews/initial_training_curves.svg) record the completed P2 experiment.

- Evaluation stops at termination/truncation, retains final observations and rewards, records
  physical applied controls, and provides explicit lengths/masks with NaN padding. Metrics and
  plots exclude padding; reward statistics use episode returns and constraint plots use positive
  violations. Each policy receives matched scenario seeds and recurrent state resets per episode.
- Nominal tracking oracle comparisons preserve the evaluator's live episode. Unsupported batch,
  custom-reward, stochastic, delta-action and early-termination comparisons raise clear errors.
- The 21 new evaluation regressions pass; six independent analytic/energy/material checks also
  pass. Updated guides and model pages describe the actual bounds, layouts, equations and tasks.
- The reproducible [runner](../../scripts/check_training_readiness.py) uses the requested Conda
  Python 3.14, CPU PPO/SAC, three seeds and 8,192 transitions per run. All **24 runs** pass
  compatibility, totaling **196,608 transitions** and **3,264 vectorized resets**. All 240 saved-policy
  evaluation episodes exactly reproduce their recorded outcomes after reload.
- Learning passes for **14/24 runs**: first order PPO/SAC 3/3 each, CSTR PPO 3/3 and SAC 2/3,
  extraction PPO 0/3 and SAC 3/3, nonsmooth PPO/SAC 0/3 each. There are no final evaluation bound
  violations or solver failures. These results support algorithm-specific recipes, not a blanket
  promotion of all four tasks. The first-order task alone clears both algorithms across all seeds.
- Final validation: **433 standard tests pass**, with one coupled-oscillator skip and one expected
  failure; **44 slow tests pass**, with 25 JAX-dependent skips. The slow suite uses one BLAS thread.
  The audit completes all 58 selected action-scenario episodes; strict audit mode remains open.

Further P2 experiments should investigate reward/value scaling, PPO rollout/learning settings
and action saturation for nonsmooth/extraction, more training for nonsmooth SAC, and CSTR SAC
seed sensitivity. Declare that experiment separately and retain these failed runs and criteria.
Successful first-order, CSTR PPO and extraction SAC recipes can be used in their tested settings
while broader qualification continues. P3–P7 are not complete.

## Priority correction and first model repairs — 2026-10-15

The first four slices changed shared runtime/evaluation behavior and added validation evidence;
they did **not** change process model implementations. They did not establish that the remaining
models were physically correct. Model/domain/default-task repairs now precede further tuning of
PPO/SAC recipes. P3–P5 no longer depend on closing every P2 learning gap.

The first model changes are:

- **Biofilm:** add the already-signed reaction sources to the transport balances in all three
  stages. This matches the nitrification equations in the [model documentation](../env/biofilm.md):
  ammonia is consumed, nitrite is formed then consumed, nitrate is produced, and oxygen is consumed.
  Independent isolated-reaction tests check direction and oxygen stoichiometry; closed-system
  simulation checks nitrogen and oxygen-equivalent inventories with unequal stage/absorber volumes.
  No primary-source resolution of the absorber equilibrium convention is claimed: code still uses
  `O_air**eq_exponent / m`, versus `m*O_air` in the documentation. Its oxygen/nitrate observation
  bounds and default objective remain under review. This model is **not training-ready**.
- **Polymerization:** apply elementary propagation mass action, `r_p = k_p M R`, with
  quasi-steady radicals `R = sqrt(f k_d I / k_t)` from `2 f k_d I = 2 k_t R^2`.
  The previous expression omitted `M`. Initiator disappearance is `k_d I`, independently of the
  radical efficiency; the previous `2 f k_d I` was the radical generation rate. Pre-exponential
  units now distinguish bimolecular propagation/termination from first-order decomposition.
  No primary calibrated reference for this parameter set has been identified, so these are
  structural kinetic corrections, not validation against experimental plant data.
  The energy balance gives `E = T + 12.5 M`, `dE/dt = F/V (E_feed - E)`, hence the nominal reset/feed
  region bounds temperature by 460 K. The old 400 K observation maximum excluded valid trajectories;
  the corrected maximum represents the energy envelope, not an equipment safety rating.
  Replace `M=3` with `M=2`: a bounded numerical operating-point search found no steady input for the
  old target, whereas the new target has an analytic admissible feed solution and converges from
  the registered initial state in a 10 s simulation. The steady solution uses `T_f=320 K`, `M_f=4`,
  `I_f=0.3`; concentrations are in kmol/m3. Learning and broader task qualification remain open.
- **Four-tank:** reduce the second pump's default maximum from 10 to 9 V. The original maximum
  admitted an upper-tank equilibrium near 0.736 m above the 0.6 m height limit. At 9 V it is about
  0.596 m. The unchanged height equations point inward at every face/corner of the default box for
  all admissible inputs; the two registered targets have an analytic feasible steady pump pair.

The focused [physics checks](../../tests/models/test_initial_task_physics.py) reproduced 15 failures
before the initial corrections. Subsequent tests also cover initiator efficiency, closed-system
integration, reachable steady inputs and complete bounded episodes. Only the biofilm and
polymerization reference trajectories were regenerated, because their dynamics changed. The biofilm reference
generator's top-20%-input workaround is removed: corrected reactions complete the full input range.
Reference agreement alone remains insufficient evidence of valid physics.

Full CasADi action scenarios (midpoint, every constant corner, alternating inputs and ten random
seeds, 60 transitions each) now give:

| Model | Episode duration | Complete | Within declared observation bounds | Remaining qualification |
|---|---|---|---|---|
| Biofilm | 100 h | 44/44 | 0/44 | Absorber equilibrium convention, oxygen/nitrate bounds, feasible objective, training |
| Polymerization | 2 s | 28/28 | 28/28 | Parameter/source validation, horizon/training recipe, learning |
| Four-tank | 1000 s | 16/16 | 16/16 | Training recipe and learning |

All three pass the Gymnasium checker. Biofilm's bound failures show why an API pass cannot establish
physical/task readiness. The polymerization 10 s target-convergence check is separate from the 2 s
action audit. Bounded episodes are also checked with both physical and normalized observations.

Validation in Conda Python 3.14.6: **44 focused physics/episode tests pass**, **471 standard tests
pass** (one autonomous-model skip and one expected failure), and **44 slow tests pass** with
25 JAX-dependent skips. Ruff lint/format checks, editor diagnostics and `git diff --check` pass.
JAX remains unavailable; no new PPO/SAC learning claim is made for these repaired models.

Next: finish biofilm's source/domain/task reconciliation and repair crystallization, followed by
the remaining P3 task/geometry defects. Existing learning gaps stay recorded without blocking these
model repairs. The package-wide training-readiness goal is not complete.

## Remaining P3 and crystallization repairs

Validated on 2026-10-15. This checkpoint changes six further model implementations. Together with the preceding repairs,
all six P3 models and polymerization/crystallization now have bounded nominal tasks and PPO/SAC
compatibility evidence. They have not yet met the multi-seed learning gate.

### Equation, domain and task changes

- **Crystallization:** stop fitted nucleation and growth at/below saturation. The previous
  absolute-supersaturation extension formed crystals in an undersaturated solution. Retain the
  positive-supersaturation fits as an explicitly **growth-only** approximation; dissolution and
  seed disappearance are outside its scope. Validate realizable, nondegenerate seed moments and
  initial CV/Ln consistency; recompute those two redundant observables from unchanged moments
  after each environment/simulation/oracle interval. Replace ineffective `1e20` moment scales,
  implement advertised `ka`/`kg` disturbances and remove nonexistent `UA`.
  The closed-solute inventory `Conc + rho*alfa/6 * 1e-12 * Mu3` is conserved. A four-hold
  30-minute temperature trajectory reaches approximately `CV=1.0131`, `Ln=15.1168 µm` for targets
  1 and 15. This is near-target terminal reachability, not exact steady feasibility.
  The [model page](../env/crystallisation.md) records the units, bounds and approximation.
  The bundled source has the same absolute-supersaturation defect; agreement with it is not
  independent physical validation. Independent moment uncertainty which breaks CV/Ln consistency
  is rejected. Parameter/disturbance/horizon extensions still require separate domain checks.
- **Distillation:** correct feed vapour flow to `V_below = V_above - (1-q)*F`. Require
  nonnegative reflux, bottoms and internal flows throughout the action box, and initial mole
  fractions in `[0,1]`. Raise default feed minimum to the fixed distillate flow of 100 kmol/hr.
  Replace infeasible purity 0.9 with 0.5; reflux 5 and feed approximately 250.07 kmol/hr give a
  valid steady solution. Unequal-holdup light-component balance and uniform-mixture checks at
  feed qualities 0, 0.5, 1 and 1.5 independently verify the corrected flow relation.
- **Series CSTR:** preserve the equations; change the default `C2` target from 60 to 97 and
  update its appended target/bounds. A steady solution at `F≈0.00222673`, `L=0.005`,
  `Tc1=Tc2=305` has state approximately `[97.0721, 298.6812, 97, 298.8379]`.
  The combined material/energy balance passes independently.
- **Reactive extraction:** preserve the equations; change `XA5` from 0.5 to 1.95 and update
  appended target/bounds. A valid steady solution uses `L≈9.84944`, `G=100`. Closed-system
  A+C and B+C inventories are conserved with unequal phase volumes. Correct the description
  of `k` to a bimolecular reaction rate constant.
- **Photoproduction:** preserve the equations; replace target 120 with 0.04 in their native
  concentration units and reduce the product/state-setpoint upper bounds to 0.11. The default
  nutrient inventory yields an optimistic 200-hour production bound of about 0.10494. Constant
  light 100 and nitrate feed approximately 5.09110066 reach product 0.04 at that horizon.
  This is terminal reachability under the existing running reward, not a steady state or
  a verified unit conversion from the old target. See [the corrected task](../env/photoproduction.md).
- **Heat exchanger:** use `π L D²/4` and corresponding annular differences for all per-stage
  volumes, consistent with documented diameters. Unchanged surface areas and parameters now give
  temperature derivatives four times larger. Add geometry, thermal-capacity and nonnegative
  mass-flow validation. The physical-capacity energy balance and analytic inlet response pass;
  `Tt8=320 K` still has a feasible steady input. Nominal parameters are not plant calibration.

The [default-task guide](../guides/default_tasks.md) documents the revised targets/scales.
Optional numeric state-validation, input-domain and derived-state hooks are documented in
[the model contribution guide](../guides/adding_a_model.md#optional-domain-and-derived-state-hooks).
Only crystallization and heat-exchanger references were regenerated in this checkpoint; together
with the earlier biofilm/polymerization references, these match the four changed default dynamics.
Distillation's default `q=1` makes the vapour correction inactive in its stored nominal reference.

### Runtime and training evidence

The strict audit uses registered defaults, CasADi, normalized actions/observations and 60 transitions
per episode: midpoint, all constant action corners, alternating extremes and ten random seeds.
Every selected task passes Gymnasium's checker, finite-state/reward checks and strict
`observation_space.contains` throughout **220/220 episodes**. Focused episode regressions additionally
cover physical observations for the eight repaired tasks.

| Default task | Episode duration (model time units) | Strict scenarios | PPO/SAC compatibility |
|---|---:|---:|---|
| first_order_system | 10 | 14/14 | Previous four-task experiment |
| cstr | 26 | 14/14 | Previous four-task experiment |
| nonsmooth_control | 5 | 14/14 | Previous four-task experiment |
| multistage_extraction | 60 | 16/16 | Previous four-task experiment |
| four_tank | 1000 | 16/16 | Both pass |
| polymerisation_reactor | 2 | 28/28 | Both pass |
| crystallization | 30 | 14/14 | Both pass |
| distillation_column | 20 | 16/16 | Both pass |
| cstr_series_recycle | 10 | 28/28 | Both pass |
| multistage_extraction_reactive | 60 | 16/16 | Both pass |
| photobioreactor | 200 | 16/16 | Both pass |
| heat_exchanger | 10 | 28/28 | Both pass |

The eight-model smoke uses seed 0, CPU, one Torch thread, `[64,64]` networks and 2,048
transitions per algorithm/model. PPO uses rollout 256 and batch 64; SAC uses warmup 256,
batch 64 and replay capacity 4,096; remaining settings are SB3 2.9.0 defaults. All **16 runs**
pass, totaling **32,768 transitions and 544 vectorized auto-resets**. Checks require finite
rewards/observations/learned parameters, strict observation containment including terminal
observations, identical deterministic actions after save/load, and a complete bounded 60-step
episode from each reloaded policy. These runs establish compatibility only; multi-seed learning
comparisons against random, midpoint and reference controllers are still required.

Final local validation uses Conda Python 3.14.6, SB3 2.9.0, Torch 2.13.0 and CasADi 3.7.2:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /data1/conda/envs/python3.14/bin/python -m pytest -q -ra --tb=short
# 523 passed, 1 coupled-oscillator skip, 1 expected failure, 69 slow deselected

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /data1/conda/envs/python3.14/bin/python -m pytest -q -ra --tb=short -m slow
# 44 passed, 25 JAX-dependent skips, 525 deselected

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /data1/conda/envs/python3.14/bin/python scripts/audit_env_readiness.py \
  cstr first_order_system nonsmooth_control multistage_extraction four_tank \
  polymerisation_reactor crystallization distillation_column cstr_series_recycle \
  multistage_extraction_reactive photobioreactor heat_exchanger \
  --strict --seeds 10 --output /tmp/pcgym-p3-audit.json
# Exit 0: 220/220 complete, finite and strictly contained scenarios
```

The focused physics/episode file now contains **96 passing cases**. Crystallization MPC fixtures
now use absolute temperatures 10–40 °C instead of the invalid -1–1 °C; the delta fixture retains
-1–1 °C **increments** with explicit absolute limits. All five crystallization MPC tests pass.
This does not repair or qualify delta-mode oracle selection, constraint signs or prediction semantics.

The strict gate correctly rejects biofilm: with `--strict --seeds 0`, all 34 midpoint/corner/
alternating episodes complete but none stay inside the declared bounds, and the command exits 1.
Its reaction mechanism is supported by the cited nitrification source; the lumped absorber
calibration (`O_air/m` versus the earlier `m*O_air` documentation) remains unresolved.
Do not widen its bounds or promote its target solely to obtain a runtime pass.

### Remaining work

Prioritize P5's five missing default tasks and biofilm's source/domain reconciliation. Then close
the explicit oracle delta defects, classify/fix the autonomous examples, and validate optional
backends. JAX is absent in the selected Conda environment. Define learning protocols for repaired
tasks and retain the earlier nonsmooth/extraction PPO/CSTR SAC learning gaps. CI/support-matrix
consolidation remains open. Changed dynamics, targets and observation scales require re-evaluating
or retraining policies saved against the old environments. The package-wide readiness goal is
not complete. At this checkpoint, all work was still uncommitted on the repair branch.

## Five new default tasks

Validated on 2026-10-15 in Conda Python 3.14.6. P5 now has complete task definitions and
independent physics/domain checks, plus CPU PPO/SAC compatibility for all five models.
All 18 controlled models have registered spaces, initial conditions and objectives. Biofilm
remains provisional; a registered task alone does not establish physical or learning readiness.

| Model | Nominal horizon / transitions | Objective and feasibility evidence | Strict scenarios |
|---|---|---|---:|
| [batch](../env/batch.md) | 15 s / 60 | Terminal intermediate yield; `2 Ca+Cb+Cc=2` and independent adiabatic energy balance. Coolant extremes give different terminal yields. | 14/14 |
| [complex_cstr](../env/complex_cstr.md) | 15 s / 60 | `Cb=0.25`; a valid steady coolant brings the reset state within 0.002 of target. Combined species/energy balances pass. | 14/14 |
| [disease](../env/disease.md) | 60 days / 60 | Minimize infection burden plus quadratic vaccination effort; population conservation and analytic vaccination/recovery checks pass. | 14/14 |
| [hydraulic_tank](../env/hydraulic_tank.md) | 15 native time units / 60 | `q1=0.9`, `q2=0.6` on conserved total 1.5; input 0.3 reaches/holds the analytic equilibrium. | 14/14 |
| [reactor_separator_recycle](../env/reactor_separator_recycle.md) | 30 native time units / 60 | Restore all three levels to 10; admissible constant flows reach them at the horizon. Component inventories and positive-domain bounds pass. | 44/44 |

The first four models' differential equations are unchanged. Their model modules now supply
defaults and physical-domain checks. Batch retains its terminal `Cb` reward. Disease is correctly
described as **SIR**, with permanent immunity, and gets a documented running cost
`dt*(I_next + 20*u^2)` in physical units. The weight is a benchmark choice, not an epidemiological
calibration. The default prefers intermediate vaccination over zero or maximal constant effort
in the tested comparison. Population/composition input validation permits float32 rounding while
rejecting invalid totals; independent uncertainty that breaks a conserved simplex is unsupported.

**Reactor–separator physics correction:** the old vapour denominator counted nonvolatile C
as volatile B. With C present and B absent, it could remove B from the separator. Normalize
only A and B: `y1=alpha*x1/(alpha*x1+x2)`, `y2=x2/(alpha*x1+x2)`, `y3=0`, consistent with the
model's existing nonvolatile-C assumption. Tests check the absent-species boundary and external
component balances with unequal areas/density. Only its stored reference trajectory was regenerated;
the five newly documented tasks retain the other four legacy simulation references.

Its default input box limits every height change to 1.2 over 30 time units, keeping initial
levels `[9.5,10.5,9.5]` inside `[8.3,11.7]`. A conservative inventory check also proves that
separator volatiles cannot be exhausted, even ignoring incoming material. Longer horizons or
different initial inventories/flow boxes may fail this proof and are rejected rather than silently
reaching a singular denominator. This proof can reject a trajectory which a tighter analysis
might establish as valid. The nominal phase law and parameters are not experimental calibration.

The shared registry adds `CustomReward` through the existing callback contract, and regulation
tasks may explicitly declare a finite horizon. Batch, disease and reactor–separator append time
remaining and terminate at the horizon; complex CSTR and hydraulic regulation truncate. Explicit
user rewards retain their own semantics. An optional `validate_episode(state, low, high, duration)`
hook checks joint inventory/input/horizon domains at construction, reset after uncertainty sampling,
and candidate simulation. Simulations validate their supplied controls and duration rather than
requiring them to fit the environment's action box. These contracts are documented in
[the contribution guide](../guides/adding_a_model.md#optional-domain-and-derived-state-hooks).

The [training runner](../../scripts/check_training_readiness.py) now has a reusable
`--compatibility-only` mode for different objective types and multiple setpoints:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /data1/conda/envs/python3.14/bin/python scripts/check_training_readiness.py \
  --compatibility-only --models batch complex_cstr disease hydraulic_tank reactor_separator_recycle \
  --output /tmp/pcgym-p5-training
```

All **10 PPO/SAC runs** pass: 2,048 transitions per algorithm/model, seed 0, one CPU thread,
`[64,64]` networks, PPO rollout 256/batch 64 and SAC warmup 256/batch 64/replay capacity 4,096.
The runner records the full configuration and versions. Across **20,480 transitions and 340
vectorized resets**, it checks strict observation containment (including terminal observations),
finite rewards/updates, horizon semantics, identical deterministic save/load actions and complete
identical reloaded-policy rollouts. This does not establish multi-seed learning quality. Batch
also has modest coolant authority and sparse reward, which the learning protocol must account for.

The combined strict audit now passes **320/320 scenarios over all 17 repaired defaults**, with
60 transitions, every action corner, midpoint, alternating extremes and ten random seeds. Focused
episode regressions cover both normalized and physical observations. The final standard suite
passes **555 tests**, with one coupled-oscillator skip and one expected failure. The slow suite
passes **44 tests**, with 25 JAX-dependent skips. JAX remains absent. The two obsolete tests
expecting batch to lack defaults now use an autonomous model for their missing-default checks.

Next: reconcile biofilm's absorber convention, bounds and objective; address the known oracle delta
defects; classify/fix the autonomous examples; and validate optional backends. Define multi-seed
learning protocols for repaired tasks and close the earlier learning gaps. The staged readiness
goal and CI/support-matrix consolidation remain incomplete. Implementation changes are recorded in
the grouped commits described above. Saved policies affected by new tasks, time entries or the
corrected phase law need re-evaluation/retraining.

## Biofilm oxygen balance and default task

Validated on 2026-10-16 in Conda Python 3.14.6. This completes the remaining controlled model's
nominal benchmark repair; it does not establish experimental calibration or learning quality.

Changes in [biofilm_reactor.py](../../src/pcgym/models/biofilm_reactor.py):

- Define the oxygen partition coefficient as gas/liquid concentration, so equilibrium is
  `O_air / m`. [NIST's oxygen solubility data](https://webbook.nist.gov/cgi/cbook.cgi?ID=C7782447&Mask=10#Solubility)
  gives `kH=0.0013 mol/(kg bar)` at 298.15 K. With water density 1 kg/L and the ideal-gas law,
  `m=1/(kH*rho*R*T)=31.03041965`, replacing 0.5. The retained gas boundary of 300 mg O2/L gas
  gives 9.66793242 mg O2/L water. Henry equilibrium requires `eq_exponent=1`; other exponents
  now raise a clear error.
- Include fresh-feed/effluent oxygen, `F/Va*(O_F-O_A)`, in the absorber balance. New parameter
  `O_F=0` declares deoxygenated feed. Previously, external oxygen outflow was omitted even though
  nitrogen species had the corresponding washout terms.
- Use mg N/L for nitrogen species and mg O2/L for oxygen. Initial/feed maxima bound ammonium
  by 2, ammonium+nitrite by 2.1, and total nitrogen by 12.1. The oxygen supply bounds dissolved
  oxygen below 10. These yield useful physical observation scales without clipping trajectories.
- Replace the infeasible nitrite target of 2 with 0.25 mg N/L. Independent steady-state solving
  finds an admissible constant control that reaches it within the 100-hour nominal episode;
  lower and higher admissible feed nitrite concentrations bracket the target.
- Validate physical parameters, initial concentrations and inputs. In
  [simulate](../../src/pcgym/pcgym.py), run candidate validation before constructing the integrator,
  so invalid overrides fail clearly rather than causing division by zero during compilation.

The [biofilm page](../env/biofilm.md) separates the cited nitrification mechanism from this
revised synthetic topology/parameterization. The available Tanaka–Dunn abstract does not establish
the exact lumped model or its calibration. The separate NIST data and explicit mass balances support
the revised benchmark assumptions; original experimental reproduction is not claimed.

The [physics tests](../../tests/models/test_initial_task_physics.py) now cover pressure-based Henry
conversion, open oxygen-equivalent fluxes, analytic washout/aeration, physical parameter rejection,
target feasibility and both physical/normalized complete episodes, alongside the prior reaction and
closed-inventory checks. Only the biofilm reference trajectory was regenerated, because its oxygen
dynamics changed. Its new target and observation scales also require saved policies to be re-evaluated
or retrained.

Validation:

- **44/44 biofilm scenarios** pass strict bounds and Gymnasium checks at 100 h / 60 transitions:
  all 32 action corners, midpoint, alternating extremes and ten seeded random action sequences.
- The combined audit now passes **364/364 scenarios across all 18 controlled defaults**.
- PPO and SAC each complete **2,048 transitions**, seed 0, CPU, one Torch thread: **4,096 transitions
  and 68 vectorized resets** total, with finite updates, correct terminal observations and identical
  deterministic save/load predictions and full reload rollouts. No learning-quality gate is claimed.
- **574 standard tests pass**, with the existing coupled-oscillator skip/expected failure and 69 slow
  tests deselected. **44 slow tests pass** with 25 JAX-dependent skips; JAX remains absent.
- Ruff lint/format checks and editor diagnostics pass for the changed Python files.

Reproduce with the selected Conda interpreter and single-thread BLAS/OpenMP settings:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
PCGYM_PYTHON=/data1/conda/envs/python3.14/bin/python
"$PCGYM_PYTHON" scripts/audit_env_readiness.py biofilm_reactor --strict --seeds 10 \
  --output /tmp/pcgym-biofilm-repaired-audit.json
"$PCGYM_PYTHON" scripts/check_training_readiness.py --compatibility-only --models biofilm_reactor \
  --output /tmp/pcgym-biofilm-training
"$PCGYM_PYTHON" -m pytest -q -ra --tb=short
"$PCGYM_PYTHON" -m pytest -q -m slow -ra --tb=short
```

Next: repair coupled-oscillator CasADi compatibility and classify the two autonomous models as
simulation-only; fix the known oracle delta-action defects; validate optional backends and feature
combinations; then complete learning qualification and CI/support-matrix consolidation.

## Autonomous CasADi simulations

Also completed on 2026-10-16:

- [Coupled oscillators](../../src/pcgym/models/coupled_oscillators.py) now assembles derivatives
  from indexed scalars, avoiding NumPy concatenation of a CasADi symbolic column. The equations
  are unchanged. Tests compare trajectories against the linear system's matrix exponential for
  1, 2 and 5 masses, check Hamiltonian energy/total momentum and center-of-mass drift, and complete
  a default ten-mass zero-input episode.
- [Invariant batch](../../src/pcgym/models/invariant_batch.py) retains its reaction equations;
  tests verify nonnegative concentrations, both conserved component inventories and zero-input
  episode semantics. Both models now reject invalid physical parameters and initial states.
- Both registry entries explicitly declare `simulation_only=True`, omit a default training task
  and expose zero inputs. The registry rejects a simulation-only declaration with a default task;
  audit JSON records the classification, and its strict controlled-task gate rejects these models.
- Removed the oscillator's CasADi expected failure and reference-test skip. Its existing JAX
  reference is preserved and compared using the existing cross-backend tolerance, so the repair
  is checked against earlier independent-backend results. Future reference generation supports
  all 20 models with core CasADi; no new oscillator reference or physics change was introduced.

The [default-task guide](../guides/default_tasks.md#autonomous-simulation-examples) includes the
classification and a complete simulation example. Neither model is claimed as an RL control task,
and historical JAX reference agreement does not establish fresh local JAX support.

Final validation after both repair slices: **589 standard tests pass with no skips or expected
failures** (69 slow tests deselected). **44 slow tests pass**, with 25 missing-JAX skips. Both
autonomous audit episodes complete; the strict training gate exits 1 for those models as intended.
The 18 controlled-default audit remains **364/364**, and the biofilm PPO/SAC compatibility evidence
above is unchanged. Ruff checks and editor diagnostics pass. The remaining work is oracle delta
semantics, optional-backend/feature checks, multi-seed learning and CI/support-matrix consolidation.

## Surge tank: averaging level control

Added on 2026-11-23: [surge_tank](../../src/pcgym/models/surge_tank.py), a horizontal cylindrical
drum with one outlet valve, documented on [its page](../env/surge_tank.md). The level is a band, not
a setpoint. The registered `CustomReward` default charges %-point-minutes outside [43, 58] % plus
valve travel, starting from a hand-over with the valve 8 % below the inflow.

- The input is the valve rate, so the ±0.2 %-per-10-min actuation contract is the action box.
  Suction-loss and overflow gates make (0, 100) % invariant for any input and horizon. Horizons
  over which the rate bound could drive the opening outside [0, 100] % are rejected.
- Independent tests check the capacity against the closed-form cylinder volume, volume conservation
  with a tight SciPy integrator (CVODES at its default tolerance agrees to 1e-3), the empty/flood
  equilibria, the reward, the band constraint, the inflow schedule and a per-episode resampling
  wrapper. The default task is not met by holding (−5477 over 8 h); an admissible ramp scores −2.0.
- The reference trajectory is CasADi. JAX 0.8.1 (a throwaway install with diffrax 0.7.2) agrees to
  8e-4 %-points and its slow reference test passes; JAX is still absent from the Conda environment.
- The strict audit passes 7/7 scenarios with each backend, and PPO/SAC compatibility passes
  (2,048 transitions, seed 0). Learning quality is not qualified.
- `SurgeTankShifts` is the stochastic benchmark: a new inflow schedule at every reset and, by
  default, no inflow in the observation (or in `info["obs"]`), so anticipation needs memory. On 200
  shifts the best memoryless controllers cost 2.2 (`lam=1`) and 14.8 (`lam=10`); a trend rule
  using one past reading costs 1.1 and 8.9. A test keeps that headroom (ratio 0.41 on 40 shifts,
  threshold 0.6). Matching the plant record's heavier upset tail was tried and rejected: upsets
  no rate-limited valve can absorb then dominate, and good controllers become indistinguishable.

Validation: **619 standard tests pass** (70 slow deselected). The surge-tank model is synthetic:
its parameters are rounded and generic, not a plant calibration.

## Progress

- [x] Audit all 20 registered models and record reproducible findings.
- [x] Create and check out the repair branch at the audited revision.
- [x] Write the staged plan and acceptance criteria.
- [x] Complete the first observation/state slice, including strict API and full-episode checks for the four candidates.
- [x] Align environment and oracle schedules, with analytic transition and MPC forecast regressions.
- [x] Complete the P1 episode, randomness and validation work for the tested CasADi configurations.
- [x] Implement P0 strict audit mode and verify that it accepts repaired tasks and rejects biofilm.
- [x] Repair the evaluator and record the first 24 PPO/SAC training results.
- [ ] Close P2's remaining learning gaps across the four tasks and both algorithms.
- [x] Repair all P3 equation/domain/default-task defects and verify PPO/SAC compatibility.
- [x] Repair polymerization and growth-only crystallization, with independent physics and compatibility checks.
- [x] Supply all five P5 default tasks, independently check physics/domains and verify PPO/SAC compatibility.
- [x] Repair biofilm oxygen flow accounting and define a source-backed synthetic equilibrium, bounds and objective.
- [ ] Establish multi-seed learning qualification for the repaired controlled tasks.
- [x] Repair oscillator CasADi support, check autonomous invariants and classify simulation-only models.
- [x] Add the surge-tank averaging-level model with independent physics, both-backend audits and PPO/SAC compatibility.
- [ ] Complete P6 optional-backend and feature-combination checks.
- [ ] Complete P7 and update the readiness assessment.
