from dataclasses import dataclass

import numpy as np
import pytest

from pcgym import make_env
from pcgym.models import MODEL_REGISTRY, BaseModel, get_model_spec, list_models, register_model
from pcgym.models._registry import _ALIASES


def test_all_builtin_models_are_registered():
    expected = {
        "batch",
        "biofilm_reactor",
        "complex_cstr",
        "coupled_oscillator",
        "crystallization",
        "cstr",
        "cstr_series_recycle",
        "disease",
        "distillation_column",
        "first_order_system",
        "four_tank",
        "heat_exchanger",
        "hydraulic_tank",
        "invariant_batch",
        "multistage_extraction",
        "multistage_extraction_reactive",
        "nonsmooth_control",
        "photobioreactor",
        "polymerisation_reactor",
        "reactor_separator_recycle",
    }
    assert set(list_models()) == expected


@pytest.mark.parametrize(
    "alias, canonical",
    [
        ("photo_production", "photobioreactor"),
        ("disease_model", "disease"),
        ("coupled_oscillators", "coupled_oscillator"),
        ("RSR", "reactor_separator_recycle"),
    ],
)
def test_aliases_resolve(alias, canonical):
    assert get_model_spec(alias) is get_model_spec(canonical)


def test_alias_works_in_make_env_with_defaults():
    env = make_env({"model": "photo_production", "N": 5, "tsim": 1, "SP": {"c_q": [0.1] * 5}})
    assert type(env.model).__name__ == "photo_production"


def test_unknown_model_lists_available_models():
    with pytest.raises(ValueError, match="not registered. Available models: .*'cstr'"):
        make_env({"model": "nope", "N": 5, "tsim": 1})


def test_legacy_import_paths_still_work():
    from pcgym.model_classes import cstr
    from pcgym.model_defaults import get_default, has_defaults

    assert cstr is get_model_spec("cstr").cls
    assert has_defaults("cstr") and not has_defaults("batch")
    assert get_default("cstr", "x0") is not get_default("cstr", "x0")  # fresh arrays each call


@pytest.fixture
def toy_model():
    def _defaults():
        return {
            "a_space": {"low": np.array([-1.0]), "high": np.array([1.0])},
            "o_space": {"low": np.array([-5.0, -5.0]), "high": np.array([5.0, 5.0])},
            "x0": np.array([0.0, 1.0]),
        }

    @register_model("toy_decay", aliases=("toy",), defaults=_defaults)
    @dataclass(frozen=False, kw_only=True)
    class toy_decay(BaseModel):
        k: float = 1.0
        int_method: str = "casadi"
        states: list = None
        inputs: list = None
        disturbances: list = None
        uncertainties: dict = None

        def __post_init__(self):
            self.states = ["x"]
            self.inputs = ["u"]
            self.disturbances = []

        def __call__(self, x, u):
            return [-self.k * x[0] + u[0]]

    yield toy_decay
    MODEL_REGISTRY.pop("toy_decay")
    _ALIASES.pop("toy")


def test_registered_model_is_usable_by_name(toy_model):
    env = make_env({"model": "toy", "N": 4, "tsim": 1, "SP": {"x": [1.0] * 4}, "model_params": {"k": 2.0}})
    assert isinstance(env.model, toy_model) and env.model.k == 2.0
    env.reset(seed=0)
    env.step(np.array([0.5]))


def test_duplicate_registration_is_rejected(toy_model):
    with pytest.raises(ValueError, match="already registered"):
        register_model("cstr")(toy_model)
    with pytest.raises(ValueError, match="already registered"):
        register_model("brand_new", aliases=("toy",))(toy_model)
