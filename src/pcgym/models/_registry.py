"""Model registry: built-in models register themselves so ``make_env`` can find them by name."""

from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass(frozen=True)
class ModelSpec:
    """A registered model.

    Attributes:
        name: Canonical name used as ``env_params["model"]``.
        cls: The model class.
        aliases: Alternative names that resolve to this model.
        defaults: Optional callable returning fresh ``a_space`` / ``o_space`` / ``x0`` defaults, used when
            those keys are omitted from ``env_params``.
    """

    name: str
    cls: type
    aliases: tuple = field(default_factory=tuple)
    defaults: Optional[Callable[[], dict]] = None


MODEL_REGISTRY: dict[str, ModelSpec] = {}
_ALIASES: dict[str, str] = {}


def register_model(name: str, *, aliases: tuple = (), defaults: Optional[Callable[[], dict]] = None):
    """Class decorator that registers a model under ``name`` (and any ``aliases``).

    Example:
        @register_model("my_reactor", defaults=_defaults)
        @dataclass(frozen=False, kw_only=True)
        class my_reactor(BaseModel):
            ...
    """

    def decorator(cls):
        for key in (name, *aliases):
            if key in MODEL_REGISTRY or key in _ALIASES:
                raise ValueError(f"A model is already registered under the name '{key}'")
        MODEL_REGISTRY[name] = ModelSpec(name=name, cls=cls, aliases=tuple(aliases), defaults=defaults)
        for alias in aliases:
            _ALIASES[alias] = name
        return cls

    return decorator


def get_model_spec(name: str) -> ModelSpec:
    """Look up a registered model by canonical name or alias."""
    spec = MODEL_REGISTRY.get(_ALIASES.get(name, name))
    if spec is None:
        raise ValueError(f"Model '{name}' is not registered. Available models: {list_models()}")
    return spec


def list_models() -> list[str]:
    """Canonical names of all registered models."""
    return sorted(MODEL_REGISTRY)
