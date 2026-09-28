"""Regression-pin every model's physics against a stored reference trajectory.

References live in tests/reference_trajectories/<model>.npz and are produced by
scripts/gen_reference_trajectories.py. A failure here means a model's dynamics changed. If that change is
intended, regenerate the affected reference with the script and say so in the PR.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from pcgym.models import list_models

REPO_ROOT = Path(__file__).resolve().parents[2]
REF_DIR = REPO_ROOT / "tests" / "reference_trajectories"

_spec = importlib.util.spec_from_file_location(
    "gen_reference_trajectories", REPO_ROOT / "scripts" / "gen_reference_trajectories.py"
)
gen = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gen)

# Errors are scaled by each state's magnitude in the reference.
CASADI_TOL = 1e-6  # CasADi reproduces its own references exactly; the margin covers platform differences.
JAX_TOL = 1e-3  # diffrax vs CVODES agree to ~1e-4, limited by CVODES' default tolerances.


def _load(model):
    path = REF_DIR / f"{model}.npz"
    if not path.exists():
        pytest.fail(f"No reference trajectory for '{model}'. Run: python scripts/gen_reference_trajectories.py {model}")
    data = dict(np.load(path))
    case = {k: data[k] for k in ("model", "x0", "u_seq", "dt")}
    return case, data["trajectory"], str(data["backend"])


def _scaled_error(traj, ref):
    return float(np.max(np.abs(traj - ref) / (np.abs(ref).max(axis=0) + 1e-12)))


@pytest.mark.parametrize("model", list_models())
def test_reference_trajectory_casadi(model):
    case, ref, backend = _load(model)
    if backend != "casadi":
        pytest.skip(f"{model} is not CasADi-compatible; checked by the JAX test")
    traj = gen.reference_env(case, "casadi").simulate(case["x0"], case["u_seq"])
    assert traj.shape == ref.shape
    assert _scaled_error(traj, ref) < CASADI_TOL


@pytest.mark.slow
@pytest.mark.parametrize("model", list_models())
def test_reference_trajectory_jax(model):
    pytest.importorskip("diffrax", reason="JAX backend needs pcgym[jax]")
    case, ref, _ = _load(model)
    traj = gen.reference_env(case, "jax").simulate(case["x0"], case["u_seq"])
    assert _scaled_error(traj, ref) < JAX_TOL


def test_no_stale_reference_files():
    stale = {p.stem for p in REF_DIR.glob("*.npz")} - set(list_models())
    assert not stale, f"Reference files for unregistered models: {sorted(stale)}"
