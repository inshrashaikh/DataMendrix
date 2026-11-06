# Contributing to DataMendrix

Thank you for helping improve DataMendrix. This project is an early-stage,
typed Python library for auditing and preparing tabular datasets. Contributions
should stay focused, tested and consistent with the existing architecture.

## Before you start

- Search existing issues and pull requests to avoid duplicate work.
- Open an issue for significant features, public API changes or dependency
  changes before implementing them.
- For bug reports, include a minimal reproduction and the versions listed in the
  bug report template.
- Do not include secrets, credentials or sensitive datasets in issues, tests or
  commits.

## Development setup

From the project root:

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# Linux or macOS
source .venv/bin/activate
```

Install the package with development dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

This installs pytest, pytest-cov, Ruff, mypy and pandas type stubs in addition
to the runtime dependencies.

## Quality gates

Every change should pass the same checks used by CI:

```bash
python -m pytest
python -m pytest --cov --cov-report=term-missing
python -m ruff check .
python -m ruff format --check .
python -m mypy
```

To apply formatting automatically:

```bash
python -m ruff format .
```

CI is configured to run on Python 3.11 and 3.12. Please run the suite locally
with at least one supported Python version before opening a pull request.

## Code guidelines

- Keep compatibility with Python 3.11 and newer.
- Use type hints for public and internal functions and methods.
- Add docstrings for public APIs and non-obvious internal helpers.
- Preserve the one-way dependency direction:
  `profiling -> detection -> core`, `profiling -> core`, `repair -> core`,
  `validation -> core`.
- Do not make analysis or preview functions mutate a `Dataset` or a caller-owned
  DataFrame. Repair functions must return a new dataset and an immutable report.
- Keep report and operation objects frozen and their collection fields immutable,
  normally tuples.
- Avoid new runtime dependencies unless there is a clear need and the change has
  been discussed first.
- Keep public errors under `DataMendrixError` and update the exported API when a
  new public type or function is introduced.
- Do not implement reporting formats, the HTTP API or frontend as part of an
  unrelated change.
- Document repair semantics and edge cases in `docs/repair.md` when cleaning
  behaviour changes.

## Tests

- Add or update tests for every behavioural change.
- Prefer focused regression tests that demonstrate the bug before the fix.
- Use `tmp_path` for filesystem tests and avoid network access.
- Test public behaviour and important edge cases rather than private details.
- Keep tests deterministic and independent of execution order.
- If a change affects warnings, dtype handling or pandas semantics, cover the
  relevant supported versions where practical.

## Documentation

Update documentation when behaviour, public API, supported environments or
limitations change:

- `README.md` for installation, examples, architecture and development.
- `docs/known-limitations.md` for intentional constraints and workarounds.
- Docstrings for function and class-level API details.

## Pull requests

1. Create a focused branch from `main`.
2. Make the smallest coherent change that solves the problem.
3. Add regression tests and run the full quality gates.
4. Update documentation where needed.
5. Open a pull request using the repository template.
6. Describe the problem, the solution, the test evidence and any API or
   compatibility impact.

Please keep pull requests free of unrelated formatting, refactoring or generated
artifacts. Maintainers may ask for changes to keep the public surface and the
current milestone consistent.

## Commit style

Clear commit messages are encouraged. Conventional Commit prefixes such as
`feat:`, `fix:`, `docs:`, `test:`, `ci:` and `chore:` are welcome but not
required.

## Reporting security issues

Do not open a public issue for a vulnerability. Contact the maintainers
privately through the security reporting mechanism configured for the
repository.

## License

By contributing, you agree that your contributions are licensed under the
project's MIT license.
