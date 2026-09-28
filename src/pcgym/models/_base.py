from dataclasses import dataclass

import numpy as np  # noqa: F401  (re-exported for model modules)

from pcgym._optional import LazyModule

# Only imported when a model is evaluated with int_method="jax".
jnp = LazyModule("jax.numpy")


@dataclass(frozen=False, kw_only=True)
class BaseModel:
    int_method: str = "jax"

    def info(self) -> dict:
        info = {
            "parameters": self.__dict__.copy(),
            "states": self.states,
            "inputs": self.inputs,
            "disturbances": self.disturbances,
            "uncertainties": list(self.uncertainties.keys()) if self.uncertainties else [],
        }
        info["parameters"].pop("int_method", None)
        return info
