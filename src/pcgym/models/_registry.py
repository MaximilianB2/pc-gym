"""Model registry: built-in models register themselves so ``make_env`` can find them by name."""

from dataclasses import dataclass, field
from typing import Callable, Optional, Union


@dataclass(frozen=True)
class Regulation:
    """Default task: track constant setpoints.

    When ``SP`` is omitted, ``make_env`` tracks each state at its value here for the whole episode. The
    squared tracking error of each state is scaled by ``1 / (o_space range)**2``, so returns are comparable
    across models.
    """

    setpoint: dict


@dataclass(frozen=True)
class Batch:
    """Default task: end-of-episode yield of ``reward_states`` (maximised unless ``maximise=False``)."""

    reward_states: tuple
    maximise: bool = True


@dataclass(frozen=True)
class ModelSpec:
    """A registered model.

    Attributes:
        name: Canonical name used as ``env_params["model"]``.
        cls: The model class.
        aliases: Alternative names that resolve to this model.
        defaults: Optional callable returning fresh ``a_space`` / ``o_space`` / ``x0`` defaults, used when
            those keys are omitted from ``env_params``.
        task: Optional default task (:class:`Regulation` or :class:`Batch`), which defines the reward used
            when neither ``SP`` nor a reward configuration is given.
    """

    name: str
    cls: type
    aliases: tuple = field(default_factory=tuple)
    defaults: Optional[Callable[[], dict]] = None
    task: Optional[Union[Regulation, Batch]] = None


MODEL_REGISTRY: dict[str, ModelSpec] = {}
_ALIASES: dict[str, str] = {}


def register_model(
    name: str,
    *,
    aliases: tuple = (),
    defaults: Optional[Callable[[], dict]] = None,
    task: Optional[Union[Regulation, Batch]] = None,
):
    """Class decorator that registers a model under ``name`` (and any ``aliases``).

    Example:
        @register_model("my_reactor", defaults=_defaults, task=Regulation(setpoint={"Ca": 0.9}))
        @dataclass(frozen=False, kw_only=True)
        class my_reactor(BaseModel):
            ...
    """

    def decorator(cls):
        for key in (name, *aliases):
            if key in MODEL_REGISTRY or key in _ALIASES:
                raise ValueError(f"A model is already registered under the name '{key}'")
        MODEL_REGISTRY[name] = ModelSpec(name=name, cls=cls, aliases=tuple(aliases), defaults=defaults, task=task)
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
