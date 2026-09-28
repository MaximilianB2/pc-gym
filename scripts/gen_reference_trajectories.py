"""Generate the per-model reference trajectories used by tests/models/test_reference_trajectories.py.

Each registered model is integrated from a fixed x0 under a fixed input sequence with a fixed dt, and the
result is written to tests/reference_trajectories/<model>.npz together with everything needed to reproduce
it. The CI test re-integrates every case and compares against the stored trajectory, pinning the physics
against silent regressions (model refactors, integrator changes, dependency bumps).

Only regenerate a reference when a change to a model's dynamics is *intended*, and say so in the PR:

    python scripts/gen_reference_trajectories.py            # all models
    python scripts/gen_reference_trajectories.py cstr four_tank
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pcgym import make_env  # noqa: E402
from pcgym.model_defaults import get_default, has_defaults  # noqa: E402
from pcgym.models import get_model_spec, list_models  # noqa: E402

OUT_DIR = REPO_ROOT / "tests" / "reference_trajectories"
N_STEPS = 30

# Time step per model, matching the benchmark configurations (tsim / 60 steps).
DT = {
    "cstr": 26 / 60,
    "first_order_system": 10 / 60,
    "nonsmooth_control": 5 / 60,
    "multistage_extraction": 1.0,
    "cstr_series_recycle": 10 / 60,
    "distillation_column": 20 / 60,
    "multistage_extraction_reactive": 1.0,
    "four_tank": 1000 / 60,
    "photobioreactor": 200 / 60,
    "heat_exchanger": 10 / 60,
    "biofilm_reactor": 100 / 60,
    "polymerisation_reactor": 2 / 60,
    "crystallization": 0.5,
}

# Models without registered default spaces: x0 (model states), input bounds and dt.
EXPLICIT = {
    "batch": {"x0": [1.0, 0.0, 0.0, 300.0], "u": ([290.0], [310.0]), "dt": 0.25},
    "invariant_batch": {"x0": [1.0, 0.8, 0.0, 0.0], "u": ([], []), "dt": 0.05},
    "complex_cstr": {"x0": [0.8, 0.1, 0.1, 330.0], "u": ([295.0], [302.0]), "dt": 0.25},
    "hydraulic_tank": {"x0": [1.0, 0.5], "u": ([-0.5], [0.5]), "dt": 0.25},
    "disease": {"x0": [0.99, 0.01, 0.0], "u": ([0.0], [0.05]), "dt": 1.0},
    "coupled_oscillator": {"x0": list(np.linspace(-1, 1, 10)) + [0.0] * 10, "u": ([], []), "dt": 0.1},
    "reactor_separator_recycle": {
        "x0": [10.0, 0.5, 0.3, 0.2] * 3,
        # Nominal balanced flows F_O=1, F_R=2, F_M=2, B=1, D=1, perturbed by +/-10%.
        "u": ([0.9, 1.8, 1.8, 0.9, 0.9], [1.1, 2.2, 2.2, 1.1, 1.1]),
        "dt": 0.5,
    },
}

# Models whose equations are not CasADi-compatible; their reference is integrated with JAX.
JAX_ONLY = {"coupled_oscillator"}

# Fraction of the default input range to start from. From the default x0, biofilm_reactor's Monod terms
# S / (K + S) blow up at low feed rates (substrate is driven negative and CVODES fails), so its
# reference stays in the top 20% of the input range.
INPUT_RANGE_START = {"biofilm_reactor": 0.8}


def reference_case(model: str) -> dict:
    """x0, input sequence and dt for one model's reference trajectory."""
    n_x = len(get_model_spec(model).cls(int_method="casadi").info()["states"])
    if model in EXPLICIT:
        cfg = EXPLICIT[model]
        x0 = np.asarray(cfg["x0"], dtype=float)
        low, high = (np.asarray(b, dtype=float) for b in cfg["u"])
        dt = cfg["dt"]
    else:
        x0 = get_default(model, "x0")[:n_x].astype(float)
        a_space = get_default(model, "a_space")
        low, high = a_space["low"].astype(float), a_space["high"].astype(float)
        dt = DT[model]
        low = low + INPUT_RANGE_START.get(model, 0.0) * (high - low)

    # Deterministic, persistently exciting inputs within bounds: a phase-shifted sinusoid per input.
    k = np.arange(N_STEPS)[:, None]
    phase = np.arange(low.shape[0])[None, :]
    u_seq = low + (high - low) * (0.5 + 0.35 * np.sin(2 * np.pi * k / 12 + phase))
    return {"model": model, "x0": x0, "u_seq": u_seq.reshape(N_STEPS, low.shape[0]), "dt": dt}


def reference_env(case: dict, integration_method: str):
    """A minimal env whose simulate() integrates the case's model with its dt."""
    x0 = np.asarray(case["x0"], dtype=float)
    n_u = np.asarray(case["u_seq"]).shape[1]
    return make_env(
        {
            "model": str(case["model"]),
            "N": N_STEPS,
            "tsim": float(case["dt"]) * N_STEPS,
            "x0": x0,
            "a_space": {"low": -np.ones(n_u), "high": np.ones(n_u)},
            "o_space": {"low": -np.inf * np.ones(x0.shape[0]), "high": np.inf * np.ones(x0.shape[0])},
            "reward_states": [],
            "maximise_reward": True,
            "normalise_o": False,
            "integration_method": integration_method,
        }
    )


def main(models: list[str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for model in models:
        case = reference_case(model)
        backend = "jax" if model in JAX_ONLY else "casadi"
        traj = reference_env(case, backend).simulate(case["x0"], case["u_seq"])
        if not np.all(np.isfinite(traj)):
            raise RuntimeError(f"{model}: non-finite reference trajectory")
        np.savez(OUT_DIR / f"{model}.npz", trajectory=traj, backend=backend, **case)
        print(f"{model:32s} {backend:6s} x{traj.shape}  x_N[:3]={np.round(traj[-1, :3], 5)}")


if __name__ == "__main__":
    main(sys.argv[1:] or list_models())
    assert all(has_defaults(m) or m in EXPLICIT for m in list_models())
