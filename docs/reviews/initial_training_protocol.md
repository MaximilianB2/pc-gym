# Initial training protocol

Declared before the first PPO/SAC training run on `fix/training-ready-environments`.
This experiment separates software compatibility from control performance. It does not
qualify untested models, optional backends or physics as training-ready.

## Fixed configuration

Use the registered task, action/observation bounds, initial states and reward scaling for
`first_order_system`, `cstr`, `nonsmooth_control` and `multistage_extraction`. Use CasADi,
normalized observations/actions, 60 transitions, and simulation durations 10, 26, 5 and
60 model time units respectively. Train nominal tasks on CPU with one Torch thread.

Run PPO and SAC for **8,192 transitions each**, with training seeds **0, 1, 2**. Both use
two 64-unit hidden layers, learning rate 0.0003 and discount 0.99. PPO uses 256 rollout
steps, batch size 64, ten epochs, GAE 0.95 and clip 0.2. SAC uses batch size 64,
256 warmup transitions, a 20,000-transition replay buffer, one update per transition,
target smoothing 0.005 and automatic entropy tuning. Retain SB3 defaults for remaining
options and record package versions and effective configuration in the result file.

Evaluate deterministic predictions at 0, 2,048, 4,096, 6,144 and 8,192 transitions.
Each checkpoint uses **ten matched seeds (100–109)**, with uniform ±5% uncertainty in
the initial tracked state and all other task values unchanged. This tests modest initial
condition variation; it is not a test of parameter uncertainty or measurement noise.
No evaluation trajectories enter training. Use the same episodes for random actions,
action midpoints and the references below. Save/reload each final policy and verify
identical deterministic predictions. Check finite policy parameters, losses, rewards,
observations and actions, plus vectorized horizon resets and terminal observations.

## References and success criteria

References use known model equations; their calibration does not use learned policies.
All physical controls saturate at the registered bounds.

| Task | Reference controller | Full-episode tracking RMSE ceiling |
|---|---|---:|
| First order | Exact sampled first-order inverse aiming at the target on the next step | 0.10 |
| CSTR | Steady-state material/energy-balance coolant plus proportional concentration correction | 0.025 |
| Nonsmooth | Feedback `u = -4 X1 - X2` | 0.30 |
| Extraction | Solve the nominal constant-flow target at the horizon, then proportional liquid-flow correction | 0.025 |

For each algorithm/task, report all three training seeds and require **every seed** to:

1. Finish all evaluation episodes without numerical failure, nonfinite data or excursions
   outside the registered finite observation/action bounds.
2. Improve mean episode return over its untrained checkpoint and the matched random baseline.
3. Close at least 25% of the return gap from the midpoint to the reference, provided the
   reference is better. Otherwise require at least the midpoint return.
4. Meet the task's RMSE ceiling above.

Record episode returns, tracking RMSE, final absolute tracking error, bounds violations,
constraint violations (when configured), episode lengths and failures. These defaults
have tracking objectives and no extra operational constraints; yield is not applicable.
Plot checkpoint evaluation returns and report failed criteria without changing the budget,
reward, controller or thresholds after seeing training results. Compatibility can pass
while these learning criteria fail. A failed criterion motivates a separately declared
follow-up experiment, not an automatic model-equation change.
