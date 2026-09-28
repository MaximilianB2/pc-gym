# Models

Built-in models live in `pcgym.models`, one module per model. Each registers itself by name with
`register_model`; see the [contributor checklist](../guides/adding_a_model.md) for adding one.

::: src.pcgym.models._registry
    options:
      members:
        - register_model
        - get_model_spec
        - list_models
        - ModelSpec
      show_root_heading: false
      show_source: false

::: src.pcgym.models
    options:
      members:
        - RSR
        - batch
        - biofilm_reactor
        - complex_cstr
        - coupled_oscillators
        - crystallization
        - cstr
        - cstr_series_recycle
        - disease_model
        - distillation_column
        - first_order_system
        - four_tank
        - heat_exchanger
        - hydraulic_tank
        - invariant_batch
        - multistage_extraction
        - multistage_extraction_reactive
        - nonsmooth_control
        - photo_production
        - polymerisation_reactor
      show_root_heading: true
      show_source: false
