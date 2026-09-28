## Summary

<!-- What does this PR change, and why? Link the issue it closes, e.g. "Closes #123". -->

## Testing

<!-- How did you test it? Paste the relevant pytest output. -->

## Adding a model?

If this PR adds a built-in model, tick every item (see docs/guides/adding_a_model.md):

- [ ] One file `src/pcgym/models/<name>.py`, registered with `@register_model` and imported in `src/pcgym/models/__init__.py`
- [ ] Units documented for every parameter, state and input
- [ ] Verified steady state as the default `x0`, with registered `a_space` / `o_space` / `x0` defaults
- [ ] Works under both CasADi and JAX (index, don't unpack; no side effects in `__call__`)
- [ ] Reference trajectory generated and committed (`python scripts/gen_reference_trajectories.py <name>`)
- [ ] Docs page `docs/env/<name>.md`, linked in `mkdocs.yml`
