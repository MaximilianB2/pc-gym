"""Deferred imports for optional dependencies.

The default CasADi backend only needs the core requirements. JAX/diffrax (``integration_method="jax"``)
and do-mpc (the MPC oracle) are imported the first time they are actually used, so ``import pcgym``
works without them.
"""

import importlib
from types import ModuleType

# Module -> the pcgym extra that installs it.
_EXTRAS = {
    "jax": "jax",
    "jax.numpy": "jax",
    "diffrax": "jax",
    "do_mpc": "oracle",
}


def require(module: str) -> ModuleType:
    """Import an optional dependency, raising an ImportError that names the extra to install."""
    try:
        return importlib.import_module(module)
    except ImportError as e:
        extra = _EXTRAS.get(module, "all")
        raise ImportError(
            f"'{module}' is required for this feature but could not be imported ({e}). "
            f'Install it with: pip install "pcgym[{extra}]"'
        ) from e


class LazyModule:
    """Module proxy that imports ``name`` on first attribute access."""

    def __init__(self, name: str) -> None:
        self._name = name
        self._module = None

    def __getattr__(self, attr: str):
        if self._module is None:
            self._module = require(self._name)
        return getattr(self._module, attr)
