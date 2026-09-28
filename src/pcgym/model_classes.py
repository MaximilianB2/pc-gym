"""Backwards-compatible import location for the built-in models.

The models now live in :mod:`pcgym.models`, one module per model.
"""

from pcgym.models import (  # noqa: F401
    RSR,
    BaseModel,
    batch,
    biofilm_reactor,
    complex_cstr,
    coupled_oscillators,
    crystallization,
    cstr,
    cstr_series_recycle,
    disease_model,
    distillation_column,
    first_order_system,
    four_tank,
    heat_exchanger,
    hydraulic_tank,
    invariant_batch,
    multistage_extraction,
    multistage_extraction_reactive,
    nonsmooth_control,
    photo_production,
    polymerisation_reactor,
)
