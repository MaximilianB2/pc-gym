"""Audit runtime readiness without changing models or regenerating references.

Run from a checkout: python scripts/audit_env_readiness.py --output /tmp/pcgym-audit.json
The JSON records failures as evidence; use --strict for a failing exit status on runtime/task-contract failures.
Even a strict runtime pass does not certify the process physics or learned control quality.
Default tasks use 60 steps at the reference generator's model-specific dt. Models without
defaults use its explicit operating points, unbounded raw observations and a zero reward;
those runs establish simulation capability only, not a usable control task.
"""

import argparse
import copy
import importlib.metadata
import itertools
import json
import platform
import sys
import warnings
from pathlib import Path

import numpy as np
from gen_reference_trajectories import DT, EXPLICIT

from pcgym import make_env
from pcgym.models import get_model_spec, list_models
from pcgym.policy_evaluation import policy_eval

STEPS = 60


def configuration(name, backend):
    spec = get_model_spec(name)
    dt = DT[name] if spec.defaults else EXPLICIT[name]["dt"]
    params = {"model": name, "N": STEPS, "tsim": STEPS * dt, "integration_method": backend}
    if spec.defaults is None:
        cfg = EXPLICIT[name]
        x0 = np.asarray(cfg["x0"], dtype=float)
        params.update(
            x0=x0,
            a_space={key: np.asarray(val, dtype=float) for key, val in zip(("low", "high"), cfg["u"])},
            o_space={"low": np.full(x0.size, -np.inf), "high": np.full(x0.size, np.inf)},
            normalise_o=False,
            reward_states=[],
        )
    return params


def error_text(exc):
    return f"{type(exc).__name__}: {exc}"


def numeric_bounds(env, obs):
    """Ignore dtype and tiny float32 bound-rounding errors, but never clip observations."""
    obs = np.asarray(obs)
    low, high = env.observation_space.low, env.observation_space.high
    return np.isfinite(obs) & (obs >= low - 1e-6) & (obs <= high + 1e-6)


def action_sequences(n_inputs, seeds):
    yield "midpoint", np.zeros((STEPS, n_inputs), dtype=np.float32)
    if not n_inputs:
        return
    for corner in itertools.product((-1.0, 1.0), repeat=n_inputs):
        label = "corner_" + "".join("L" if v < 0 else "H" for v in corner)
        yield label, np.tile(np.asarray(corner, dtype=np.float32), (STEPS, 1))
    alternating = np.where(np.arange(STEPS)[:, None] % 2, 1.0, -1.0)
    yield "alternating", np.tile(alternating, (1, n_inputs)).astype(np.float32)
    for seed in range(seeds):
        yield f"random_{seed}", np.random.default_rng(seed).uniform(-1, 1, (STEPS, n_inputs)).astype(np.float32)


def conserved_quantities(name, x):
    if name == "batch":
        return np.array([2 * x[0] + x[1] + x[2]])
    if name == "invariant_batch":
        return np.array([x[0] + x[2] + 2 * x[3], x[1] + x[2] + x[3]])
    if name == "disease":
        return np.array([sum(x[:3])])
    if name == "hydraulic_tank":
        return np.array([sum(x[:2])])
    if name == "reactor_separator_recycle":
        return np.array([sum(x[i + 1 : i + 4]) for i in (0, 4, 8)])
    return np.array([])


def rollout(params, actions):
    result = {"steps": 0, "finite": True, "out_of_bounds": {}, "dtype_matches": True, "space_contains": True}
    rewards = []
    env = None
    try:
        env = make_env(params)
        obs, _ = env.reset(seed=0)
        state_min, state_max = env.state.copy(), env.state.copy()
        initial_balance = conserved_quantities(params["model"], env.state)
        if initial_balance.size:
            result["max_conservation_error"] = 0.0
        names = [name for name, _ in env.observation_info()]

        def inspect(observation, step):
            result["dtype_matches"] &= observation.dtype == env.observation_space.dtype
            result["space_contains"] &= env.observation_space.contains(observation)
            for idx in np.flatnonzero(~numeric_bounds(env, observation)):
                result["out_of_bounds"].setdefault(
                    names[idx], {"step": step, "observation": float(observation[idx]), "state": float(env.state[idx])}
                )

        inspect(obs, 0)
        for step, action in enumerate(actions, 1):
            assert env.action_space.contains(action), "Audit supplied an invalid action"
            obs, reward, terminated, truncated, _ = env.step(action)
            result["steps"] = step
            inspect(obs, step)
            rewards.append(float(reward))
            state_min = np.minimum(state_min, env.state)
            state_max = np.maximum(state_max, env.state)
            if initial_balance.size:
                result["max_conservation_error"] = max(
                    result["max_conservation_error"],
                    float(np.max(np.abs(conserved_quantities(params["model"], env.state) - initial_balance))),
                )
            if not (np.isfinite(obs).all() and np.isfinite(reward)):
                result["finite"] = False
                break
            if terminated or truncated:
                result["flags"] = [terminated, truncated]
                break
        result["state_min"] = dict(zip(names, state_min.tolist()))
        result["state_max"] = dict(zip(names, state_max.tolist()))
        result["final_state"] = dict(zip(names, env.state.tolist()))
        result["return"] = sum(rewards)
        expected_flags = [env.finite_horizon, not env.finite_horizon]
        result["complete"] = result["steps"] == STEPS and result.get("flags") == expected_flags
    except Exception as exc:
        result["error"] = error_text(exc)
        result["complete"] = False
    finally:
        if env is not None:
            env.close()
    return result


def checker(params):
    from gymnasium.utils.env_checker import check_env

    env = None
    try:
        env = make_env(params)
        env.action_space.seed(0)
        check_env(env, skip_render_check=True)
        return "pass"
    except Exception as exc:
        return error_text(exc)
    finally:
        if env is not None:
            env.close()


def common_probes():
    """Small reproductions of cross-model issues identified during source review."""
    probes = {}
    params = {"model": "first_order_system", "N": 4, "tsim": 1, "normalise_o": False}
    env = make_env({**params, "SP": {"x": [0.2, 0.4, 0.6, 0.8]}})
    reset_obs, reset_info = env.reset(seed=0)
    probes["explicit_setpoint_at_reset"] = {"observed": float(reset_obs[-1]), "expected": 0.2}
    _, reward, _, _, info = env.step(np.array([0.0]))
    probes["setpoint_after_first_step"] = {
        "observed": float(env.state[-1]),
        "reward_target": env.SP["x"][env.t],
        "reward": float(reward),
    }
    probes["reset_info_mutates_after_step"] = reset_info is info and "u" in reset_info

    env = make_env(
        {
            **params,
            "uncertainty_percentages": {"x0": {0: 0.2}},
            "uncertainty_bounds": {"low": np.array([]), "high": np.array([])},
        }
    )
    samples = [float(env.reset(seed=seed)[0][0]) for seed in range(5)]
    probes["dict_x0_uncertainty_samples"] = samples

    try:
        env = make_env({**params, "noise": True})
        env.reset()
        env.step(np.array([0.0]))
        probes["noise_without_percentage"] = "pass"
    except Exception as exc:
        probes["noise_without_percentage"] = error_text(exc)

    env = make_env({**params, "x0": np.array([0, 0]), "SP": {"x": [0.7] * 4}})
    env.reset()
    env.step(np.array([1.0]))
    probes["integer_x0_after_step"] = {"state": env.state.tolist(), "dtype": str(env.state.dtype)}

    env = make_env({"model": "crystallization", "N": 4, "tsim": 2})
    obs, _ = env.reset()
    probes["crystallization_normalised_moments_float32"] = obs[:4].astype(np.float32).tolist()
    x = env.x0[: env.Nx_oracle]
    a = env.model(x, np.array([25.0, 0.1, 0.1]))
    b = env.model(x, np.array([25.0, 100.0, 100.0]))
    probes["crystallization_disturbance_vector_ignored"] = bool(np.array_equal(a, b))
    disturbed = copy.deepcopy(env.env_params)
    disturbed.update(
        disturbances={"ka": [1.0] * 4},
        disturbance_bounds={"low": np.array([0.0]), "high": np.array([2.0])},
    )
    try:
        env = make_env(disturbed)
        env.reset()
        env.step(np.array([0.0]))
        probes["crystallization_ka_disturbance"] = "pass"
    except Exception as exc:
        probes["crystallization_ka_disturbance"] = error_text(exc)

    class ConstantPolicy:
        def predict(self, obs, deterministic=True):
            return np.array([0.0]), None

    constrained = {
        **params,
        "constraints": {"x": [0.0, 0.11]},
        "done_on_cons_vio": True,
        "r_penalty": False,
    }
    env = make_env(constrained)
    env.reset()
    terminated = env.step(np.array([0.0]))[2]
    evaluator = policy_eval(make_env, {"constant": ConstantPolicy()}, 1, constrained)
    _, _, _, _, metadata = evaluator.rollout(ConstantPolicy(), return_metadata=True)
    probes["evaluation_continues_after_termination"] = {
        "first_step_terminated": terminated,
        "evaluated_steps": metadata["lengths"],
    }
    uncertain = {
        **params,
        "uncertainty_percentages": {"K": 0.1},
        "uncertainty_bounds": {"low": np.array([0.9]), "high": np.array([1.1])},
    }
    try:
        evaluator = policy_eval(make_env, {"constant": ConstantPolicy()}, 1, uncertain)
        evaluator.rollout(ConstantPolicy())
        probes["evaluation_with_parameter_uncertainty"] = "pass"
    except Exception as exc:
        probes["evaluation_with_parameter_uncertainty"] = error_text(exc)

    # Independent algebraic checks of feasibility and basic physical domains.
    photo = get_model_spec("photobioreactor").cls(int_method="casadi")
    horizon = STEPS * DT["photobioreactor"]
    probes["photobioreactor_product_upper_bound"] = photo.k_m * (
        horizon * (1 + 150 / photo.Y_NX) + 10 * horizon**2 / (2 * photo.Y_NX)
    )
    reactive = get_model_spec("multistage_extraction_reactive").cls(int_method="casadi")
    probes["reactive_extraction_steady_XA5_lower_bound"] = reactive.XA0 * (5 / (5 + reactive.Kla * reactive.Vl)) ** 5
    series = get_model_spec("cstr_series_recycle").cls(int_method="casadi")
    k_max = series.k * np.exp(-series.E / (series.R * 320))
    probes["series_recycle_steady_C2_conservative_lower_bound"] = series.C_O * (
        1 - k_max * (series.V1 + series.V2) / 0.001
    )
    polymer = get_model_spec("polymerisation_reactor").cls(int_method="casadi")
    probes["polymerisation_derivative_at_zero_monomer"] = polymer(
        np.array([400.0, 0.0, 0.3]), np.array([1.05, 340.0, 6.0, 0.65])
    ).tolist()
    heat = get_model_spec("heat_exchanger").cls(int_method="casadi")
    uniform = np.tile([350.0, 350.0, 350.0], 8)
    actual = heat(uniform, np.array([1.0, 1.0, 360.0, 350.0]))[0]
    expected = 10 / (heat.rhot * heat.L * np.pi * heat.Dt**2 / 4)
    probes["heat_exchanger_diameter_volume_check"] = {
        "actual_Tt1_derivative": float(actual),
        "expected_using_declared_diameter": float(expected),
    }
    env = make_env(configuration("first_order_system", "casadi"))
    env.reset()
    for _ in range(STEPS):
        env.step(np.array([0.0]))
    expected = 0.5 + (0.1 - 0.5) * np.exp(-env.tsim / 0.5)
    probes["first_order_analytic_final_state_error"] = abs(float(env.state[0]) - float(expected))
    return probes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", help="Canonical model names (default: all)")
    parser.add_argument("--backend", choices=("casadi", "jax"), default="casadi")
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail unless selected controlled default tasks pass every runtime scenario and Gymnasium check",
    )
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")
    versions = {}
    for package in ("pcgym", "numpy", "gymnasium", "casadi", "jax", "diffrax", "scipy", "do-mpc"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    report = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "versions": versions,
        "backend": args.backend,
        "steps": STEPS,
        "seeds": args.seeds,
        "models": {},
    }
    for name in args.models or list_models():
        spec = get_model_spec(name)
        params = configuration(name, args.backend)
        model_info = spec.cls(int_method="casadi").info()
        item = {
            "default_spaces": spec.defaults is not None,
            "default_task": repr(spec.task),
            "states": model_info["states"],
            "inputs": model_info["inputs"],
            "tsim": params["tsim"],
            "gymnasium_checker": checker(params),
            "rollouts": {},
        }
        for label, actions in action_sequences(len(model_info["inputs"]), args.seeds):
            item["rollouts"][label] = rollout(params, actions)
        report["models"][name] = item
        complete = sum(run["complete"] for run in item["rollouts"].values())
        bounded = sum(run["complete"] and not run["out_of_bounds"] for run in item["rollouts"].values())
        print(f"{name:32s} complete={complete}/{len(item['rollouts'])} numerically_bounded={bounded}", flush=True)
    report["common_probes"] = common_probes()
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Evidence written to {args.output}")
    else:
        print(json.dumps(report, indent=2))
    if args.strict:
        failures = [
            name
            for name, item in report["models"].items()
            if not item["default_spaces"]
            or not item["inputs"]
            or get_model_spec(name).task is None
            or item["gymnasium_checker"] != "pass"
            or any(
                not (run["complete"] and run["finite"] and run["space_contains"]) for run in item["rollouts"].values()
            )
        ]
        if failures:
            print("Strict runtime checks failed: " + ", ".join(failures), file=sys.stderr)
            raise SystemExit(1)
        print("Strict runtime checks passed for all selected tasks; physics and learning require separate evidence.")


if __name__ == "__main__":
    main()
