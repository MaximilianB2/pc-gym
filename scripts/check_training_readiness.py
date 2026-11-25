#!/usr/bin/env python3
"""Reproduce the initial CPU PPO/SAC experiment; see initial_training_protocol.md.

Example: python scripts/check_training_readiness.py --output /tmp/pcgym-training
The output includes the frozen configuration, episode metrics, learning curves and
saved policies. A learning failure is reported separately from an execution failure.
Use --compatibility-only --models <names> for nominal smoke checks of other default tasks.
"""

import argparse
import copy
import importlib.metadata
import json
import platform
from pathlib import Path

import matplotlib
import numpy as np
import torch
from gen_reference_trajectories import DT
from scipy.optimize import brentq
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from pcgym import make_env
from pcgym.evaluation_metrics import episode_returns
from pcgym.policy_evaluation import policy_eval

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TASKS = {
    "first_order_system": (10, 0.10),
    "cstr": (26, 0.025),
    "nonsmooth_control": (5, 0.30),
    "multistage_extraction": (60, 0.025),
}
EVAL_SEEDS = list(range(100, 110))
PPO_CONFIG = dict(n_steps=256, batch_size=64, n_epochs=10, gae_lambda=0.95, clip_range=0.2)
SAC_CONFIG = dict(
    batch_size=64,
    learning_starts=256,
    buffer_size=20_000,
    train_freq=1,
    gradient_steps=1,
    tau=0.005,
    ent_coef="auto",
)
COMMON = dict(learning_rate=0.0003, gamma=0.99, policy_kwargs={"net_arch": [64, 64]}, device="cpu", verbose=0)


def task_params(name):
    return {"model": name, "N": 60, "tsim": TASKS[name][0], "integration_method": "casadi"}


class Baseline:
    """Midpoint, reproducible random, or an equation-based physical controller."""

    def __init__(self, env, kind):
        self.env, self.kind = env, kind
        self.reset(seed=0)
        self.low, self.high = (env.env_params["a_space"][key] for key in ("low", "high"))
        self.target_name = next(iter(env.SP))
        self.index = env.model.info()["states"].index(self.target_name)
        self.target = float(env.SP[self.target_name][0])
        self.feedforward = None
        if kind == "reference" and env.env_params["model"] == "multistage_extraction":
            gas = (self.low[1] + self.high[1]) / 2

            def final_error(liquid):
                trajectory = env.simulate(env.x0, np.tile([liquid, gas], (env.N, 1)))
                return trajectory[-1, self.index] - self.target

            liquid = brentq(final_error, self.low[0], self.high[0], xtol=1e-9)
            self.feedforward = np.array([liquid, gas])

    def reset(self, *, seed):
        self.rng = np.random.default_rng(seed)

    def predict(self, obs, deterministic=True):
        if self.kind == "random":
            return self.rng.uniform(-1, 1, self.env.action_space.shape).astype(np.float32), None
        if self.kind == "midpoint":
            return np.zeros(self.env.action_space.shape, dtype=np.float32), None
        env = self.env
        bounds = env.observation_space_base
        x = bounds.low + (np.asarray(obs, dtype=float) + 1) * (bounds.high - bounds.low) / 2
        model = env.model
        name = env.env_params["model"]
        if name == "first_order_system":
            decay = np.exp(-env.dt / model.tau)
            physical = [(self.target - decay * x[0]) / (model.K * (1 - decay))]
        elif name == "cstr":
            rate = model.q / model.V * (model.Caf - self.target)
            temperature = -model.EA_over_R / np.log(rate / (model.k0 * self.target))
            heat = model.UA / (model.rho * model.C * model.V)
            coolant = (
                temperature
                - (model.q / model.V * (model.Ti - temperature) - model.deltaHr * rate / (model.rho * model.C)) / heat
            )
            physical = [coolant + 100 * (x[0] - self.target)]
        elif name == "nonsmooth_control":
            physical = [-4 * x[0] - x[1]]
        else:
            physical = self.feedforward.copy()
            physical[0] += 1000 * (self.target - x[self.index])
        action = 2 * (np.clip(physical, self.low, self.high) - self.low) / (self.high - self.low) - 1
        return action.astype(np.float32), None


def evaluate(config, policies):
    evaluator = policy_eval(make_env, policies, len(EVAL_SEEDS), config, seeds=EVAL_SEEDS)
    try:
        data = evaluator.get_rollouts()
        env = evaluator.env
        state_index = env.model.info()["states"].index(next(iter(env.SP)))
        target = np.asarray(next(iter(env.SP.values())))
        target = target[np.minimum(np.arange(env.N + 1), len(target) - 1)]
        metrics = {}
        for name, entry in data.items():
            for key in ("x", "r", "u"):
                if not np.isfinite(entry[key]).all():
                    raise RuntimeError(f"{name}: incomplete/nonfinite evaluation {key}")
            error = entry["x"][state_index, 1:, :] - target[1:, None]
            lower = env.observation_space_base.low[:, None, None]
            upper = env.observation_space_base.high[:, None, None]
            state_violations = np.any((entry["x"] < lower - 1e-9) | (entry["x"] > upper + 1e-9), axis=0)
            low, high = (env.env_params["a_space"][k][:, None, None] for k in ("low", "high"))
            input_violations = np.any((entry["u"] < low - 1e-9) | (entry["u"] > high + 1e-9), axis=0)
            metrics[name] = {
                "returns": episode_returns(entry).ravel().tolist(),
                "rmse": np.sqrt(np.mean(error**2, axis=0)).tolist(),
                "final_absolute_error": np.abs(error[-1]).tolist(),
                "state_bound_violations": state_violations.sum(axis=0).tolist(),
                "input_bound_violations": input_violations.sum(axis=0).tolist(),
                "constraint_violations": ((entry["g"] > 0).sum(axis=(0, 1, 2)).tolist() if "g" in entry else None),
                "lengths": entry["lengths"].tolist(),
                "terminated": entry["terminated"].tolist(),
                "truncated": entry["truncated"].tolist(),
                "seeds": entry["seeds"].tolist(),
                "failures": 0,
            }
        return metrics
    finally:
        evaluator.env.close()


def validate_parameters(model):
    if not all(torch.isfinite(parameter).all().item() for parameter in model.policy.parameters()):
        raise RuntimeError("Nonfinite policy parameters")
    values = {key: float(value) for key, value in model.logger.name_to_value.items() if key.startswith("train/")}
    if not all(np.isfinite(value) for value in values.values()):
        raise RuntimeError("Nonfinite training diagnostics")
    if model.__class__ is SAC and not torch.isfinite(model.log_ent_coef).all().item():
        raise RuntimeError("Nonfinite entropy coefficient")
    return values


class CompatibilityChecks(BaseCallback):
    """Strict observation and terminal/reset checks for regulation and finite tasks."""

    def __init__(self, config):
        super().__init__()
        env = make_env(config)
        self.space = env.observation_space
        self.horizon = env.N
        self.finite_horizon = env.finite_horizon
        self.episodes = 0
        env.close()

    def _on_step(self):
        assert all(self.space.contains(obs) for obs in self.locals["new_obs"])
        assert np.isfinite(self.locals["new_obs"]).all() and np.isfinite(self.locals["rewards"]).all()
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self.episodes += 1
                assert self.space.contains(info["terminal_observation"])
                assert info["TimeLimit.truncated"] == (not self.finite_horizon)
                assert info["episode"]["l"] == self.horizon
        return True


def compatibility_rollout(env, model):
    obs, _ = env.reset(seed=100)
    assert env.observation_space.contains(obs)
    total = 0.0
    for step in range(env.N):
        action, _ = model.predict(obs, deterministic=True)
        assert env.action_space.contains(action)
        obs, reward, terminated, truncated, _ = env.step(action)
        assert env.observation_space.contains(obs) and np.isfinite(reward)
        assert terminated == (env.finite_horizon and step == env.N - 1)
        assert truncated == (not env.finite_horizon and step == env.N - 1)
        total += reward
    return float(total)


def run_compatibility(args, versions):
    results = {
        "python": platform.python_version(),
        "packages": versions,
        "mode": "compatibility_only",
        "config": {
            "steps": args.steps,
            "seeds": args.training_seeds,
            "torch_threads": 1,
            "common": COMMON,
            "PPO": PPO_CONFIG,
            "SAC": {**SAC_CONFIG, "buffer_size": 4096},
        },
        "tasks": {},
    }
    for name in args.models:
        config = {"model": name, "N": 60, "tsim": DT[name] * 60, "integration_method": "casadi"}
        env = make_env(config)
        task = {
            "config": config,
            "resolved_environment": copy.deepcopy(env.env_params),
            "model_parameters": env.model.info()["parameters"],
            "runs": [],
        }
        # Store the importable callback name rather than a process-specific function repr.
        reward = task["resolved_environment"].get("custom_reward")
        if reward is not None:
            task["resolved_environment"]["custom_reward"] = f"{reward.__module__}.{reward.__qualname__}"
        results["tasks"][name] = task
        for algorithm in (PPO, SAC):
            for seed in args.training_seeds:
                vector_env = DummyVecEnv([lambda config=config: Monitor(make_env(copy.deepcopy(config)))])
                run = {"algorithm": algorithm.__name__, "seed": seed}
                task["runs"].append(run)
                try:
                    kwargs = PPO_CONFIG if algorithm is PPO else {**SAC_CONFIG, "buffer_size": 4096}
                    model = algorithm("MlpPolicy", vector_env, seed=seed, **copy.deepcopy(COMMON), **kwargs)
                    callback = CompatibilityChecks(config)
                    model.learn(total_timesteps=args.steps, callback=callback)
                    diagnostics = validate_parameters(model)
                    assert model.num_timesteps == args.steps and callback.episodes == args.steps // env.N
                    obs, _ = env.reset(seed=100)
                    action, _ = model.predict(obs, deterministic=True)
                    path = args.output / f"{name}_{algorithm.__name__}_{seed}.zip"
                    model.save(path)
                    loaded = algorithm.load(path, device="cpu")
                    np.testing.assert_array_equal(action, loaded.predict(obs, deterministic=True)[0])
                    expected = compatibility_rollout(env, model)
                    assert compatibility_rollout(env, loaded) == expected
                    run.update(
                        compatibility=True,
                        transitions=int(model.num_timesteps),
                        vectorized_episodes=callback.episodes,
                        diagnostics=diagnostics,
                        save_load_prediction_equal=True,
                        save_load_rollout_equal=True,
                        evaluation_return=expected,
                    )
                    print(
                        f"{name} {algorithm.__name__} seed={seed}: pass, "
                        f"{callback.episodes} resets, return={expected:.6g}",
                        flush=True,
                    )
                except Exception as error:
                    run.update(compatibility=False, failure=f"{type(error).__name__}: {error}")
                    print(f"{name} {algorithm.__name__} seed={seed}: {run['failure']}", flush=True)
                finally:
                    vector_env.close()
                save_results(results, args.output)
        env.close()
    return int(any(not run["compatibility"] for task in results["tasks"].values() for run in task["runs"]))


class EvaluationCheckpoints(BaseCallback):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.checkpoints = []
        self.episodes = 0

    def record(self):
        metrics = evaluate(self.config, {"learned": self.model})["learned"]
        self.checkpoints.append({"steps": int(self.num_timesteps), "evaluation": metrics})
        print(f"  steps={self.num_timesteps} return={np.mean(metrics['returns']):.6f}", flush=True)

    def _on_training_start(self):
        self.record()

    def _on_step(self):
        for obs in self.locals["new_obs"]:
            if not np.isfinite(obs).all():
                raise RuntimeError("Nonfinite training observation")
        if not np.isfinite(self.locals["rewards"]).all():
            raise RuntimeError("Nonfinite training reward")
        for done, info in zip(self.locals["dones"], self.locals["infos"]):
            if done:
                self.episodes += 1
                assert info["TimeLimit.truncated"]
                assert np.isfinite(info["terminal_observation"]).all()
                assert info["episode"]["l"] == 60
        if self.num_timesteps % 2048 == 0:
            validate_parameters(self.model)
            # Preserve global RNG states so evaluation cannot change training exploration.
            numpy_rng, torch_rng = np.random.get_state(), torch.get_rng_state()
            try:
                self.record()
            finally:
                np.random.set_state(numpy_rng)
                torch.set_rng_state(torch_rng)
        return True


def criteria(run, baseline, ceiling):
    initial, final = (run["checkpoints"][i]["evaluation"] for i in (0, -1))
    mean_return = np.mean(final["returns"])
    midpoint, reference = (np.mean(baseline[key]["returns"]) for key in ("midpoint", "reference"))
    return {
        "complete_and_bounded": (
            all(length == 60 for length in final["lengths"])
            and not any(final["state_bound_violations"])
            and not any(final["input_bound_violations"])
            and final["failures"] == 0
        ),
        "improves_untrained": bool(mean_return > np.mean(initial["returns"])),
        "beats_random": bool(mean_return > np.mean(baseline["random"]["returns"])),
        "closes_reference_gap": bool(mean_return >= midpoint + 0.25 * max(reference - midpoint, 0)),
        "meets_rmse_ceiling": bool(np.mean(final["rmse"]) <= ceiling),
    }


def save_results(results, destination):
    target = destination / "results.json"
    temporary = target.with_suffix(".json.tmp")

    def numpy_json(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Cannot serialize {type(value).__name__}")

    temporary.write_text(json.dumps(results, indent=2, allow_nan=False, default=numpy_json) + "\n")
    temporary.replace(target)


def plot_results(results, destination):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (name, task) in zip(axes.flat, results["tasks"].items()):
        styles = {"random": ("0.4", ":"), "midpoint": ("tab:purple", "--"), "reference": ("tab:green", "-.")}
        for baseline, metrics in task["baselines"].items():
            color, style = styles[baseline]
            ax.axhline(np.mean(metrics["returns"]), color=color, linestyle=style, label=baseline, alpha=0.8)
        for algorithm, color in (("PPO", "tab:blue"), ("SAC", "tab:orange")):
            runs = [run for run in task["runs"] if run["algorithm"] == algorithm and "checkpoints" in run]
            for i, run in enumerate(runs):
                steps = [point["steps"] for point in run["checkpoints"]]
                returns = [np.mean(point["evaluation"]["returns"]) for point in run["checkpoints"]]
                ax.plot(steps, returns, color=color, alpha=0.7, label=algorithm if i == 0 else None)
        ax.set(title=name, xlabel="Training transitions", ylabel="Mean evaluation return")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(destination / "learning_curves.svg")
    fig.savefig(destination / "learning_curves.png", dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=sorted(DT), default=list(TASKS))
    parser.add_argument("--steps", type=int)
    parser.add_argument("--training-seeds", nargs="+", type=int)
    parser.add_argument(
        "--compatibility-only",
        action="store_true",
        help="Check nominal training, auto-resets and save/load without a learning claim",
    )
    parser.add_argument("--baselines-only", action="store_true")
    parser.add_argument(
        "--resume", action="store_true", help="Retain completed compatible runs with the same configuration"
    )
    args = parser.parse_args()
    if args.steps is None:
        args.steps = 2048 if args.compatibility_only else 8192
    if args.training_seeds is None:
        args.training_seeds = [0] if args.compatibility_only else [0, 1, 2]
    if args.compatibility_only and (args.resume or args.baselines_only):
        parser.error("--compatibility-only cannot be combined with --resume or --baselines-only")
    if not args.compatibility_only and any(name not in TASKS for name in args.models):
        parser.error(
            "The learning protocol covers only the initial four tasks; use --compatibility-only for other models"
        )
    if args.steps < 2048 or args.steps % 2048:
        parser.error("--steps must be a positive multiple of 2048")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    versions = {
        name: importlib.metadata.version(name)
        for name in ("pcgym", "numpy", "gymnasium", "casadi", "scipy", "torch", "stable-baselines3")
    }
    if args.compatibility_only:
        return run_compatibility(args, versions)
    results = {
        "python": platform.python_version(),
        "packages": versions,
        "config": {
            "steps": args.steps,
            "training_seeds": args.training_seeds,
            "evaluation_seeds": EVAL_SEEDS,
            "torch_threads": 1,
            "common": COMMON,
            "PPO": PPO_CONFIG,
            "SAC": SAC_CONFIG,
            "evaluation_initial_tracked_state_uncertainty": 0.05,
        },
        "tasks": {},
    }
    if args.resume:
        previous = json.loads((args.output / "results.json").read_text())
        previous_config = copy.deepcopy(previous["config"])
        # Older harness runs recorded SAC's insertion of its default in the shared dict.
        if previous_config["common"]["policy_kwargs"].get("use_sde") is False:
            previous_config["common"]["policy_kwargs"].pop("use_sde")
        if previous_config != results["config"] or previous["packages"] != results["packages"]:
            parser.error("Resume requires identical experiment configuration and package versions")
        results["tasks"] = previous["tasks"]
    for name in args.models:
        config = task_params(name)
        env = make_env(config)
        target_index = env.model.info()["states"].index(next(iter(env.SP)))
        evaluation_config = {**config, "uncertainty_percentages": {"x0": {target_index: 0.05}}}
        policies = {key: Baseline(env, key) for key in ("random", "midpoint", "reference")}
        baselines = evaluate(evaluation_config, policies)
        previous_runs = results["tasks"].get(name, {}).get("runs", [])
        task = {
            "config": config,
            "resolved_environment": copy.deepcopy(env.env_params),
            "model_parameters": env.model.info()["parameters"],
            "baselines": baselines,
            "runs": [run for run in previous_runs if run.get("compatibility")],
        }
        results["tasks"][name] = task
        print(name, {key: float(np.mean(value["returns"])) for key, value in baselines.items()}, flush=True)
        for algorithm in () if args.baselines_only else (PPO, SAC):
            for seed in args.training_seeds:
                if any(run["algorithm"] == algorithm.__name__ and run["seed"] == seed for run in task["runs"]):
                    continue
                print(f"{name} {algorithm.__name__} seed={seed}", flush=True)
                vector_env = DummyVecEnv([lambda config=config: Monitor(make_env(copy.deepcopy(config)))])
                run = {"algorithm": algorithm.__name__, "seed": seed}
                task["runs"].append(run)
                try:
                    kwargs = PPO_CONFIG if algorithm is PPO else SAC_CONFIG
                    # SB3 may add algorithm-specific options to policy_kwargs in place.
                    model = algorithm("MlpPolicy", vector_env, seed=seed, **copy.deepcopy(COMMON), **kwargs)
                    callback = EvaluationCheckpoints(evaluation_config)
                    model.learn(total_timesteps=args.steps, callback=callback)
                    # on_step occurs before the last optimizer update; record the actual final policy.
                    callback.checkpoints.pop()
                    callback.record()
                    diagnostics = validate_parameters(model)
                    obs, _ = env.reset(seed=100)
                    action, _ = model.predict(obs, deterministic=True)
                    save_path = args.output / f"{name}_{algorithm.__name__}_{seed}.zip"
                    model.save(save_path)
                    loaded = algorithm.load(save_path, device="cpu")
                    restored_action, _ = loaded.predict(obs, deterministic=True)
                    np.testing.assert_array_equal(action, restored_action)
                    restored_metrics = evaluate(evaluation_config, {"loaded": loaded})["loaded"]
                    np.testing.assert_array_equal(
                        restored_metrics["returns"], callback.checkpoints[-1]["evaluation"]["returns"]
                    )
                    assert env.action_space.contains(restored_action)
                    assert model.num_timesteps == args.steps and callback.episodes > 0
                    run.update(
                        compatibility=True,
                        transitions=int(model.num_timesteps),
                        vectorized_episodes=callback.episodes,
                        diagnostics=diagnostics,
                        save_load_prediction_equal=True,
                        save_load_rollout_equal=True,
                        checkpoints=callback.checkpoints,
                    )
                    run["criteria"] = criteria(run, baselines, TASKS[name][1])
                    run["learning_pass"] = all(run["criteria"].values())
                    print(f"  compatibility=pass learning={run['learning_pass']} {run['criteria']}", flush=True)
                except Exception as error:
                    run.update(compatibility=False, failure=f"{type(error).__name__}: {error}")
                    print(f"  FAILED: {run['failure']}", flush=True)
                finally:
                    vector_env.close()
                save_results(results, args.output)
        env.close()
        save_results(results, args.output)
    plot_results(results, args.output)
    return int(any(not run["compatibility"] for task in results["tasks"].values() for run in task["runs"]))


if __name__ == "__main__":
    raise SystemExit(main())
