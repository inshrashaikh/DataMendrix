# DataMendrix

DataMendrix is an open-source toolkit for auditing, cleaning and preparing
datasets for machine learning.

This repository currently contains the **Python foundation only**. It loads CSV
datasets, validates their structure, profiles missing values and numeric
columns, and detects duplicate rows. The frontend, HTTP API, reporting layer and
advanced repair engine are intentionally not part of this release.

## Features

- Defensive `Dataset` container that preserves the original data.
- CSV loader with encoding, separator, missing-value and number-format options.
- Structural validation with machine-readable issue codes and an optional
  fail-fast exception.
- Missing-value statistics at dataset and column level.
- Duplicate-row detection, including column subsets and first/last policies.
- Descriptive statistics for real-valued numeric columns.
- Combined, immutable dataset profiles.
- Full pytest suite and typed, `pyproject.toml`-based packaging.

## Requirements

- Python 3.11+
- NumPy
- pandas

## Installation

```bash
python -m pip install -e ".[dev]"
```

## Quick start

```python
from data_mendrix import (
    load_csv,
    missing_value_report,
    numeric_statistics,
    profile_dataset,
    validate_dataset,
)

dataset = load_csv("data/customers.csv", na_values=["N/A", "unknown"])

validation = validate_dataset(dataset)
validation.raise_for_errors()

profile = profile_dataset(dataset)
print(profile.row_count, profile.column_count)
print(profile.missing.missing_cell_fraction)
print(profile.duplicates.duplicate_row_count)
print(profile.numeric.column_names)

missing = missing_value_report(dataset)
for column in missing.columns_with_missing:
    print(column.column, column.missing_count)

numeric = numeric_statistics(dataset)
print(numeric.column("age").median)
```

`Dataset.frame` always returns a copy, and analysis functions never modify the
dataset. Missing values follow pandas' standard `NaN`/`None`/`pd.NA`/`NaT`
semantics; configure `na_values` on the loader to treat extra strings as missing.

## Architecture

```text
src/data_mendrix/
├── core/            Dataset container, CSV loader, exception hierarchy
├── validation/      Structural validation and issue reporting
├── detection/       Duplicate-row detection
├── profiling/       Missing-value, numeric and combined profiling
├── repair/          Reserved for the future repair engine (not implemented)
├── reporting/       Reserved for future report formats (not implemented)
└── experimentation/ Reserved for future experiments (not implemented)
```

Dependency direction is deliberately one-way: analysis modules consume
`core`, while `core` contains no data-quality logic. The loader therefore does
not reject structurally imperfect datasets; call `validate_dataset()` when a
fail-fast contract is appropriate.

## Development

```bash
python -m pytest
python -m pytest --cov
python -m ruff check .
python -m mypy
```

The test suite uses `pythonpath = ["src"]`, so it runs without installing the
package. Installing with `pip install -e ".[dev]"` enables the optional linting,
typing and coverage tools.

## License

MIT
