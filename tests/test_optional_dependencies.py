"""pcgym must import and run the default CasADi backend without the optional extras installed."""

import subprocess
import sys
import textwrap

# Setting sys.modules[name] = None makes any later `import name` raise ImportError, which simulates
# an environment where the extras (pcgym[jax], pcgym[oracle]) are not installed.
BLOCK_EXTRAS = textwrap.dedent(
    """
    import sys
    for name in ("jax", "jax.numpy", "jaxlib", "diffrax", "equinox", "do_mpc"):
        sys.modules[name] = None
    """
)

ENV_PARAMS = textwrap.dedent(
    """
    import numpy as np
    params = {
        "model": "cstr",
        "N": 5,
        "tsim": 1,
        "SP": {"Ca": [0.85] * 5},
        "a_space": {"low": np.array([295]), "high": np.array([302])},
        "o_space": {"low": np.array([0.7, 300, 0.8]), "high": np.array([1, 350, 0.9])},
        "x0": np.array([0.8, 330, 0.8]),
    }
    """
)


def _run(code):
    return subprocess.run(
        [sys.executable, "-c", BLOCK_EXTRAS + ENV_PARAMS + textwrap.dedent(code)],
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_casadi_episode_runs_without_optional_extras():
    result = _run(
        """
        from pcgym import make_env
        env = make_env(params)
        env.reset(seed=0)
        for _ in range(5):
            env.step(np.array([0.0]))
        assert "jax" not in sys.modules or sys.modules["jax"] is None
        print("OK")
        """
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_jax_backend_without_extra_gives_install_hint():
    result = _run(
        """
        from pcgym import make_env
        params["integration_method"] = "jax"
        env = make_env(params)
        try:
            env.reset()
        except ImportError as e:
            print(e)
        """
    )
    assert result.returncode == 0, result.stderr
    assert 'pip install "pcgym[jax]"' in result.stdout


def test_oracle_without_extra_gives_install_hint():
    result = _run(
        """
        from pcgym import make_env
        class Zero:
            def predict(self, o, deterministic=True):
                return np.array([0.0]), None
        env = make_env(params)
        try:
            env.get_rollouts({"zero": Zero()}, reps=1, oracle=True)
        except ImportError as e:
            print(e)
        """
    )
    assert result.returncode == 0, result.stderr
    assert 'pip install "pcgym[oracle]"' in result.stdout
