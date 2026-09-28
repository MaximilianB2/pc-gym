"""Built-in pc-gym models.

Each model lives in its own module and registers itself with :func:`register_model`, together with its
default spaces. Adding a model is a single new file plus an import below (see the contributor guide).
"""

from pcgym.models._base import BaseModel
from pcgym.models._registry import MODEL_REGISTRY, ModelSpec, get_model_spec, list_models, register_model

# Importing each module registers its model.
from pcgym.models.batch import batch
from pcgym.models.biofilm_reactor import biofilm_reactor
from pcgym.models.complex_cstr import complex_cstr
from pcgym.models.coupled_oscillators import coupled_oscillators
from pcgym.models.crystallization import crystallization
from pcgym.models.cstr import cstr
from pcgym.models.cstr_series_recycle import cstr_series_recycle
from pcgym.models.disease_model import disease_model
from pcgym.models.distillation_column import distillation_column
from pcgym.models.first_order_system import first_order_system
from pcgym.models.four_tank import four_tank
from pcgym.models.heat_exchanger import heat_exchanger
from pcgym.models.hydraulic_tank import hydraulic_tank
from pcgym.models.invariant_batch import invariant_batch
from pcgym.models.multistage_extraction import multistage_extraction
from pcgym.models.multistage_extraction_reactive import multistage_extraction_reactive
from pcgym.models.nonsmooth_control import nonsmooth_control
from pcgym.models.photo_production import photo_production
from pcgym.models.polymerisation_reactor import polymerisation_reactor
from pcgym.models.reactor_separator_recycle import RSR

__all__ = [
    "BaseModel",
    "RSR",
    "batch",
    "biofilm_reactor",
    "complex_cstr",
    "coupled_oscillators",
    "crystallization",
    "cstr",
    "cstr_series_recycle",
    "disease_model",
    "distillation_column",
    "first_order_system",
    "four_tank",
    "heat_exchanger",
    "hydraulic_tank",
    "invariant_batch",
    "multistage_extraction",
    "multistage_extraction_reactive",
    "nonsmooth_control",
    "photo_production",
    "polymerisation_reactor",
    "MODEL_REGISTRY",
    "ModelSpec",
    "get_model_spec",
    "list_models",
    "register_model",
]
