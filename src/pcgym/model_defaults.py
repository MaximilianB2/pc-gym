"""Canonical default action/observation spaces and initial states per model.

These let users construct an environment without hand-specifying ``a_space``,
``o_space`` and ``x0`` every time - which is especially tedious for the larger
models (the heat exchanger has 24 states, the reactive extraction column 20).
``make_env`` falls back to these values for any of the three keys the user
omits from ``env_params``.

The bounds mirror the configurations used for the benchmark suite
(``scripts/benchmark_models.py``). ``o_space``/``x0`` include the model's
canonical set-point dimension(s), so they are intended to be used together
(and with a set point of matching dimensionality).
"""

import numpy as np

# crystallization initial coefficient of variation / mean length, derived from
# the leading moment initial conditions.
_CRYST_CV0 = float(np.sqrt(1800863.24079725 * 1478.00986666666 / (22995.8230590611**2) - 1))
_CRYST_LN0 = 22995.8230590611 / (1478.00986666666 + 1e-6)


def _spaces():
    """Return the default-spaces table.

    Built lazily so every call yields fresh NumPy arrays and no caller can
    mutate the shared defaults.
    """
    return {
        "cstr": {
            "a_space": {"low": np.array([295.0]), "high": np.array([302.0])},
            "o_space": {"low": np.array([0.7, 300, 0.8]), "high": np.array([1.0, 350, 0.9])},
            "x0": np.array([0.8, 330, 0.8]),
        },
        "first_order_system": {
            "a_space": {"low": np.array([0.0]), "high": np.array([1.0])},
            "o_space": {"low": np.array([0.0, 0.0]), "high": np.array([1.0, 1.0])},
            "x0": np.array([0.1, 0.4]),
        },
        "nonsmooth_control": {
            "a_space": {"low": np.array([-1.0]), "high": np.array([1.0])},
            "o_space": {"low": np.array([-1.0, -1.0, -1.0]), "high": np.array([1.0, 1.0, 1.0])},
            "x0": np.array([0.8, 0.0, 0.0]),
        },
        "multistage_extraction": {
            "a_space": {"low": np.array([5.0, 10.0]), "high": np.array([500.0, 1000.0])},
            "o_space": {"low": np.array([0.0] * 10 + [0.3]), "high": np.array([1.0] * 10 + [0.4])},
            "x0": np.array([0.55, 0.3, 0.45, 0.25, 0.4, 0.2, 0.35, 0.15, 0.25, 0.1, 0.3]),
        },
        "cstr_series_recycle": {
            "a_space": {
                "low": np.array([0.001, 0.001, 290.0, 290.0]),
                "high": np.array([0.01, 0.01, 320.0, 320.0]),
            },
            "o_space": {
                "low": np.array([0.0, 290.0, 0.0, 290.0, 55.0]),
                "high": np.array([100.0, 400.0, 100.0, 400.0, 85.0]),
            },
            "x0": np.array([90.0, 310.0, 85.0, 310.0, 80.0]),
        },
        "distillation_column": {
            "a_space": {"low": np.array([1.0, 80.0]), "high": np.array([10.0, 300.0])},
            "o_space": {"low": np.array([0.0] * 9 + [0.8]), "high": np.array([1.0] * 9 + [0.95])},
            "x0": np.array([0.85, 0.75, 0.6, 0.4, 0.2, 0.15, 0.1, 0.05, 0.02, 0.85]),
        },
        "multistage_extraction_reactive": {
            "a_space": {"low": np.array([5.0, 10.0]), "high": np.array([500.0, 1000.0])},
            "o_space": {"low": np.array([0.0] * 20 + [0.25]), "high": np.array([2.0] * 20 + [0.55])},
            "x0": np.array([1.0, 0.0, 1.0, 0.0] * 5 + [0.3]),
        },
        "four_tank": {
            "a_space": {"low": np.array([0.1, 0.1]), "high": np.array([10.0, 10.0])},
            "o_space": {"low": np.array([0.0] * 6), "high": np.array([0.6] * 6)},
            "x0": np.array([0.141, 0.112, 0.072, 0.42, 0.5, 0.2]),
        },
        "photobioreactor": {
            "a_space": {"low": np.array([100.0, 0.0]), "high": np.array([400.0, 10.0])},
            "o_space": {
                "low": np.array([0.0, 0.0, 0.0, 50.0]),
                "high": np.array([10.0, 1000.0, 300.0, 200.0]),
            },
            "x0": np.array([1.0, 150.0, 0.0, 80.0]),
        },
        "heat_exchanger": {
            "a_space": {
                "low": np.array([0.5, 0.5, 340.0, 300.0]),
                "high": np.array([5.0, 5.0, 360.0, 320.0]),
            },
            "o_space": {"low": np.array([280.0] * 24 + [300.0]), "high": np.array([400.0] * 24 + [360.0])},
            "x0": np.concatenate([np.tile([350.0, 340.0, 310.0], 8), [340.0]]),
        },
        "biofilm_reactor": {
            "a_space": {
                "low": np.array([0.0, 1.0, 0.05, 0.05, 0.05]),
                "high": np.array([10.0, 30.0, 1.0, 1.0, 1.0]),
            },
            "o_space": {
                "low": np.array([0.0, 0.0, 0.0, 0.0] * 4 + [0.9]),
                "high": np.array([10.0, 10.0, 10.0, 500.0] * 4 + [2.5]),
            },
            "x0": np.array([2.0, 0.1, 10.0, 0.1] * 4 + [1.0]),
        },
        "polymerisation_reactor": {
            "a_space": {"low": np.array([0.1, 320.0, 4.0, 0.3]), "high": np.array([2.0, 360.0, 8.0, 1.0])},
            "o_space": {"low": np.array([300.0, 0.0, 0.0, 1.5]), "high": np.array([400.0, 10.0, 1.0, 3.5])},
            "x0": np.array([340.0, 5.0, 0.3, 2.0]),
        },
        "crystallization": {
            "a_space": {"low": np.array([10.0]), "high": np.array([40.0])},
            "o_space": {
                "low": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9, 14.0]),
                "high": np.array([1e20, 1e20, 1e20, 1e20, 0.5, 2.0, 20.0, 1.1, 16.0]),
            },
            "x0": np.array(
                [
                    1478.00986666666,
                    22995.8230590611,
                    1800863.24079725,
                    248516167.940593,
                    0.15861523304,
                    _CRYST_CV0,
                    _CRYST_LN0,
                    1.0,
                    15.0,
                ]
            ),
        },
    }


def has_defaults(model_name: str) -> bool:
    """Whether canonical default spaces exist for ``model_name``."""
    return model_name in _spaces()


def get_default(model_name: str, key: str):
    """Return a fresh copy of the default ``key`` (``a_space``/``o_space``/``x0``).

    Raises:
        KeyError: if no defaults are registered for ``model_name``.
        ValueError: if ``key`` is not one of the supported space keys.
    """
    table = _spaces()
    if model_name not in table:
        raise KeyError(
            f"No default {key} is registered for model '{model_name}'. "
            f"Please provide '{key}' explicitly in env_params. "
            f"Models with defaults: {sorted(table)}"
        )
    if key not in table[model_name]:
        raise ValueError(f"'{key}' is not a defaultable space; expected one of a_space, o_space, x0")
    return table[model_name][key]
