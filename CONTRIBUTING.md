# Contributing to PC-Gym

Thanks for taking the time to contribute. This guide covers the dev workflow,
how CI gates PRs, and how releases are cut.

## Reporting issues

Use the GitHub issue templates:

- **Bug report** — for behavior that doesn't match the docs or the paper.
- **Feature request** — for new environments, metrics, or workflow ideas.

If you've found a security-relevant issue, see [SECURITY.md](SECURITY.md)
instead of opening a public issue.

## Development setup

PC-Gym targets Python 3.11+ (3.11, 3.12, 3.13 are exercised in CI).

```bash
git clone https://github.com/MaximilianB2/pc-gym.git
cd pc-gym
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

The `dev` extras pull in pytest, coverage, ruff, pyright, pre-commit,
pip-audit, build, twine, and nbmake.

## Running tests

The suite uses a `slow` marker to split fast PR-time tests from
heavier MPC / JAX / parameter-uncertainty tests.

```bash
pytest                  # default: fast tests only (matches PR CI)
pytest -m slow          # only the slow tests
pytest -m ""            # the full suite (matches nightly CI)
pytest -n auto          # in parallel (recommended locally)
```

Coverage is reported by `--cov=pcgym` if you want a local report.

## Lint and format

Ruff handles both. Pre-commit runs both on every commit; you can also
invoke them directly:

```bash
ruff format .           # auto-format
ruff format --check .   # check only, no edits (what CI runs)
ruff check .            # lint
ruff check . --fix      # lint and auto-fix safe issues
```

Type checks via `pyright` are advisory in CI (warnings, not failures)
while the type annotations are filled in incrementally.

## Submitting a PR

1. Branch off `main`.
2. Make your change and add or update tests. Mark expensive tests
   with `@pytest.mark.slow` so they don't slow down PR runs.
3. Run the local pre-commit + `pytest` before pushing.
4. Push and open a PR against `main`. The PR template asks for
   a summary, motivation, and the testing you did.
5. CI must be green on: `lint`, `test` (3.11, 3.12, 3.13), `build`.
   `security` and `pyright` are advisory.
6. At least one CODEOWNER review is required before merge.

## Release process (maintainers)

Releases go to PyPI via OIDC Trusted Publishing — there is no API
token in repo secrets. Setup is documented in the header of
[.github/workflows/release.yml](.github/workflows/release.yml).

To cut a release once Trusted Publishing is configured:

1. Bump `version` in `pyproject.toml` on `main`.
2. Tag the release commit: `git tag v0.X.Y && git push origin v0.X.Y`.
3. The release workflow builds sdist + wheel, runs `twine check`,
   and waits on the `release` GitHub environment for reviewer
   approval before publishing.

Dependabot opens grouped weekly PRs for `pip` and `github-actions`
dependencies; merging those is part of routine maintenance.
