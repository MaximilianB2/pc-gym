"""Canonical default action/observation spaces and initial states per model.

``make_env`` falls back to these for any of ``a_space`` / ``o_space`` / ``x0`` omitted from ``env_params``.
Defaults are declared next to each model via ``register_model(..., defaults=...)``; this module is the lookup
API. ``o_space``/``x0`` include the model's canonical set-point dimension(s), so they are intended to be used
with a set point of matching dimensionality.
"""

from pcgym.models import get_model_spec

_SPACE_KEYS = ("a_space", "o_space", "x0")


def _defaults_for(model_name: str):
    try:
        spec = get_model_spec(model_name)
    except ValueError:
        return None
    return spec.defaults


def has_defaults(model_name: str) -> bool:
    """Whether canonical default spaces exist for ``model_name``."""
    return _defaults_for(model_name) is not None


def get_default(model_name: str, key: str):
    """Return a fresh copy of the default ``key`` (``a_space``/``o_space``/``x0``).

    Raises:
        KeyError: if no defaults are registered for ``model_name``.
        ValueError: if ``key`` is not one of the supported space keys.
    """
    defaults = _defaults_for(model_name)
    if defaults is None:
        from pcgym.models import MODEL_REGISTRY

        with_defaults = sorted(n for n, s in MODEL_REGISTRY.items() if s.defaults is not None)
        raise KeyError(
            f"No default {key} is registered for model '{model_name}'. "
            f"Please provide '{key}' explicitly in env_params. "
            f"Models with defaults: {with_defaults}"
        )
    if key not in _SPACE_KEYS:
        raise ValueError(f"'{key}' is not a defaultable space; expected one of a_space, o_space, x0")
    return defaults()[key]
