# DataMendrix

DataMendrix is an open-source Python toolkit for auditing, cleaning and preparing
tabular datasets for machine learning.

> **Status:** early-stage (`0.3.x`). This repository contains the Python
> foundation, detection engine and cleaning/repair engine: CSV loading, structural
> validation, missing-value statistics, duplicate-row detection, numeric
> descriptive statistics, combined profiling, univariate outlier detection,
> constant-feature detection, class-imbalance analysis, and previewable
> missing-value, duplicate-row and data-type repairs. Reporting formats, the HTTP
> API and frontend are not implemented yet.

## Features

- Defensive `Dataset` container that preserves the original data.
- CSV loader with encoding, separator, missing-value and number-format options.
- Structural validation with stable machine-readable issue codes and an optional
  fail-fast exception.
- Missing-value statistics at dataset and column level.
- Duplicate-row detection, including column subsets and first/last policies.
- Descriptive statistics for real-valued numeric columns.
- Combined, immutable dataset profiles.
- Univariate outlier detection using IQR, Z-score or modified Z-score (MAD),
  with affected row labels and per-value explanations.
- Constant and near-constant feature detection with a configurable dominance
  threshold.
- Binary and multiclass target-imbalance analysis with counts, percentages and
  configurable ratio/minority thresholds.
- Explicit repair plans for missing values, duplicate rows and data types.
- Repair previews, immutable change reports, affected rows/columns and safe
  conversion-error policies.
- Typed public API, pytest suite and `pyproject.toml`-based packaging.
- CI configured for Python 3.11 and 3.12.

## Requirements

- Python 3.11 or newer.
- NumPy and pandas, installed automatically as runtime dependencies.

CI is configured to test Python 3.11 and 3.12.

## Installation

### Development installation

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

Install DataMendrix in editable mode with the development tools:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For a runtime-only installation from a source checkout, use:

```bash
python -m pip install .
```

## Quick start

```python
from data_mendrix import (
    FillMissing,
    RemoveDuplicates,
    apply_repair,
    detect_outliers,
    load_csv,
    preview_repair,
    profile_dataset,
    validate_dataset,
)

dataset = load_csv("data/customers.csv", na_values=["N/A", "unknown"])

validation = validate_dataset(dataset)
if not validation.is_valid:
    print(validation.codes)
    validation.raise_for_errors()

profile = profile_dataset(dataset)
print(profile.row_count, profile.column_count)
print(profile.missing.missing_cell_count)
print(profile.duplicates.duplicate_row_count)
print(profile.numeric.column_names)

outliers = detect_outliers(dataset)
for column in outliers.columns_with_outliers:
    print(column.column, column.outlier_indices)

plan = [
    FillMissing(columns=["age"], strategy="median"),
    RemoveDuplicates(subset=["customer_id"]),
]
preview = preview_repair(dataset, plan)
print(preview.change_count, preview.affected_rows)

result = apply_repair(dataset, plan)
cleaned = result.dataset
print(cleaned.shape, result.report.issues)
```

## API examples

### Load a CSV file

```python
from data_mendrix import load_csv

dataset = load_csv(
    "data/export.csv",
    encoding="utf-8",
    na_values=["N/A", "unknown", "-"],
)

print(dataset.shape)
print(dataset.columns)
print(dataset.dtypes)
```

The loader supports `.csv` paths and already-open text streams. It performs I/O
and parsing only; malformed data is not silently repaired.

### Validate structure

```python
from data_mendrix import ValidationConfig, validate_dataset

config = ValidationConfig(
    min_rows=10,
    required_columns=("customer_id", "email"),
)
report = validate_dataset(dataset, config=config)

for issue in report.issues:
    print(issue.code.value, issue.message)

if not report.is_valid:
    report.raise_for_errors()
```

Validation checks shape, duplicate and blank column names, optional unnamed
columns, and required columns. It does not assess statistical quality or
semantic correctness.

### Profile a dataset

```python
from data_mendrix import profile_dataset

profile = profile_dataset(dataset)

print(f"rows={profile.row_count}, columns={profile.column_count}")
print(f"memory={profile.memory_usage_bytes} bytes")
print(f"missing cells={profile.missing.missing_cell_count}")
print(f"duplicate rows={profile.duplicates.duplicate_row_count}")

for column in profile.columns:
    print(column.name, column.dtype, column.unique_count)
```

### Detect duplicate rows

```python
from data_mendrix import detect_duplicates

all_duplicates = detect_duplicates(dataset)
key_duplicates = detect_duplicates(
    dataset,
    subset=["customer_id"],
    keep="last",
)

print(all_duplicates.duplicate_row_count)
print(key_duplicates.duplicate_indices)
```

Duplicate detection only reports rows; it never removes them.

### Preview and apply repairs

Repairs are expressed as an explicit, ordered plan. Previewing a plan runs it on
an internal copy and returns the same immutable report that applying it will
return. `apply_repair` also returns a new `Dataset`; the input dataset is never
modified.

```python
from data_mendrix import (
    ConvertTypes,
    DropMissing,
    FillMissing,
    RemoveDuplicates,
    apply_repair,
    preview_repair,
)

plan = [
    DropMissing(axis="rows", columns=["customer_id"], how="any"),
    FillMissing(columns=["age", "income"], strategy="median"),
    FillMissing(columns=["segment"], strategy="mode"),
    RemoveDuplicates(subset=["customer_id"], keep="first"),
    ConvertTypes(conversions={"age": "Int64"}, errors="coerce"),
]

preview = preview_repair(dataset, plan)
print(preview.row_count_before, preview.row_count_after)
print(preview.affected_rows, preview.affected_columns)
for operation in preview.operations:
    print(operation.action, operation.change_count, operation.issues)

result = apply_repair(dataset, plan)
cleaned = result.dataset
report = result.report
```

Operation classes validate their arguments when constructed:

| Operation | Purpose |
|---|---|
| `DropMissing` | Drop rows or columns using `any`/`all` missingness and an optional column scope |
| `FillMissing` | Fill with `mean`, `median`, `mode` or an explicit constant |
| `RemoveDuplicates` | Remove full-row or subset duplicates with `first`, `last` or `False` |
| `ConvertTypes` | Convert selected columns to pandas-compatible dtypes |

For single repairs, `drop_missing`, `fill_missing`, `remove_duplicates` and
`convert_dtypes` are convenience wrappers that return `RepairResult`. See
[docs/repair.md](docs/repair.md) for strategy defaults, error policies and report
semantics.

### Detect numeric outliers

```python
from data_mendrix import detect_outliers

report = detect_outliers(
    dataset,
    columns=["age", "income"],
    method="modified_zscore",
    threshold=3.5,
)

print(report.outlier_indices)
for column in report.columns_with_outliers:
    print(column.column, column.outlier_count)
    for evidence in column.evidence:
        print(evidence.row_index, evidence.value, evidence.reason)
```

The default method is `"iqr"`. Supported methods are `"iqr"`, `"zscore"` and
`"modified_zscore"`. Default thresholds are `1.5`, `3.0` and `3.5`
respectively. By default, all real-valued numeric columns are inspected; boolean
and complex columns are excluded.

### Detect constant and near-constant features

```python
from data_mendrix import detect_constant_features

report = detect_constant_features(dataset, dominance_threshold=0.98)

print(report.constant_column_names)
print(report.near_constant_column_names)
for column in report.near_constant_columns:
    print(column.column, column.dominant_value, column.dominant_fraction)
```

A near-constant column has a dominant non-missing value covering at least the
configured fraction of non-missing observations. Exact constant columns are
included in the near-constant collection.

### Analyse class imbalance

```python
from data_mendrix import analyze_class_imbalance

report = analyze_class_imbalance(
    dataset,
    target="churn",
    imbalance_ratio_threshold=2.0,
    minority_fraction_threshold=0.1,
)

print(report.class_count, report.imbalance_ratio, report.is_imbalanced)
for item in report.classes:
    print(item.label, item.count, item.percentage)
```

The target column must be selected explicitly. Missing target values are
excluded from class counts and percentages. The ratio condition uses
`majority_count / minority_count`; multiclass targets compare the largest and
smallest classes. See [docs/detection.md](docs/detection.md) for method formulas
and threshold semantics.

### Missing values and numeric statistics

```python
from data_mendrix import missing_value_report, numeric_statistics

missing = missing_value_report(dataset)
for column in missing.columns_with_missing:
    print(column.column, column.missing_count, column.missing_fraction)

numeric = numeric_statistics(dataset)
if "age" in numeric.column_names:
    age = numeric.column("age")
    print(age.mean, age.std, age.median, age.iqr)
```

### Work with an in-memory DataFrame

```python
import pandas as pd

from data_mendrix import Dataset, profile_dataset

dataset = Dataset(
    pd.DataFrame({"value": [1.0, 2.0, None]}),
    name="inline-example",
    source="generated",
)
print(profile_dataset(dataset).missing.missing_cell_count)
```

## Core API

| Area | Primary API | Report or value types |
|---|---|---|
| Container | `Dataset` | DataFrame copy, shape, name and source metadata |
| Loading | `load_csv` | `Dataset` |
| Validation | `validate_dataset` | `ValidationConfig`, `ValidationReport`, `ValidationIssue`, `ValidationCode` |
| Duplicates | `detect_duplicates` | `DuplicateReport` |
| Outliers | `detect_outliers` | `OutlierReport`, `ColumnOutlierReport`, `OutlierEvidence` |
| Constant features | `detect_constant_features` | `ConstantFeatureReport`, `ConstantColumnStats` |
| Class imbalance | `analyze_class_imbalance` | `ClassImbalanceReport`, `ClassCount` |
| Missing values | `missing_value_report` | `MissingValueReport`, `ColumnMissingStats` |
| Numeric statistics | `numeric_statistics` | `NumericStatistics`, `NumericColumnStatistics` |
| Combined profile | `profile_dataset` | `DatasetProfile`, `ColumnProfile` |
| Repair plan | `preview_repair`, `apply_repair` | `RepairReport`, `RepairOperationReport`, `RepairResult` |
| Missing-value repair | `drop_missing`, `fill_missing` | `RepairResult`, `ColumnChange`, `RepairIssue` |
| Duplicate repair | `remove_duplicates` | `RepairResult` |
| Data-type repair | `convert_dtypes` | `RepairResult`, `RepairConversionError` |
| Errors | `DataMendrixError` and subclasses | Typed library exceptions |

All report and operation objects are frozen dataclasses, and all analysis
functions are read-only. Repair functions never modify their input dataset; they
return a new dataset plus an immutable report.

## Data safety and semantics

- `Dataset` takes a defensive copy on construction.
- `Dataset.frame` returns another copy, so callers cannot mutate the internal
  frame through the public API.
- Analysis and preview functions never modify the dataset.
- Repair operations run on a defensive copy. `apply_repair` returns a new
  `Dataset` and a `RepairReport`; the original dataset remains unchanged.
- Repair plans are explicit and ordered. There are no implicit or automatic
  repairs, and every operation reports affected rows, affected columns and
  change counts.
- Missing values follow pandas semantics: `NaN`, `None`, `pandas.NA` and `NaT`.
  CSV fields matching pandas' default NA tokens are missing unless
  `keep_default_na=False` is passed to the loader.
- Duplicate rows use pandas equality semantics, including equal missing-value
  patterns.
- Outlier detection is univariate and numeric-only. Infinite values are flagged
  as outliers and excluded from the finite baseline used to calculate bounds.
- Constant-feature and class-imbalance analyses use non-missing observations for
  their fractions and percentages; missing values are reported separately.
- Detection thresholds are configurable heuristics, not automatic repair
  decisions. Reports identify values and rows but never modify the dataset.
- `mean`/`median` fills use finite, non-missing values and default to numeric
  columns. `mode` uses the first non-missing value in a tie. Fills with no usable
  baseline are reported as issues and leave the column unchanged.
- Conversion errors follow an explicit policy: `raise` (typed
  `RepairConversionError`), `skip` (record an issue and leave the column
  unchanged) or `coerce` (replace invalid values with missing where the target
  dtype supports missing values).
- The loader does not validate automatically. This allows structurally
  imperfect datasets to be loaded and audited before a fail-fast decision.

## Architecture

```text
src/data_mendrix/
├── core/            Dataset container, CSV loader, exception hierarchy
├── validation/      Structural validation and issue reporting
├── detection/       Duplicate, outlier, constant and imbalance detection
├── profiling/       Missing-value, numeric and combined profiling
├── repair/          Missing-value, duplicate and dtype repair
├── reporting/       Reserved for future report formats (not implemented)
└── experimentation/ Reserved for future experiments (not implemented)
```

Dependency direction is deliberately one-way:

```text
profiling ──> detection ──> core
profiling ──> core
repair ──> core
validation ──> core
```

`core` contains shared container, loading, exception and column-selection logic.
Public exports are assembled in `src/data_mendrix/__init__.py`. Reports and
operations are immutable snapshots rather than mutable analysis state.

## Known limitations

The current release is intentionally narrow:

- Only CSV paths and text streams are supported; the path must end in `.csv`.
- Files are loaded into memory with pandas; there is no out-of-core processing.
- Reporting formats, the HTTP API and frontend are not implemented.
- Validation is structural only; it does not enforce data types, semantic rules
  or target leakage. Statistical checks are separate read-only detection APIs.
- Repairs are explicit, in-memory transformations. There is no automatic repair
  recommendation, custom row-wise callback, rollback log or saved repair plan.
- Dtype repair delegates parsing to pandas. Numeric, datetime and timedelta
  coercion use pandas conversion rules, and conversions that would turn existing
  missing values into data are rejected. Duplicate labels in a selection are
  considered only once.
- With no considered columns (a zero-column dataset), `axis="rows"` with
  `how="all"` drops every row, matching `pandas.dropna`'s vacuous-truth
  semantics; `how="any"` drops nothing.
- Repair reports describe what a plan did; they do not prove that the repaired
  dataset is semantically correct for a downstream model.
- Outlier detection is univariate and numeric-only. It does not detect
  multivariate, contextual or time-series anomalies.
- Outlier thresholds are statistical heuristics. Flagged values are candidates
  for review, not automatically invalid data.
- Constant and class-imbalance thresholds are caller-configured heuristics.
  Missing target values are excluded from class percentages.
- Numeric statistics exclude booleans, complex values and non-numeric object
  dtypes such as `Decimal`, datetimes and timedeltas.
- Report column labels are stringified, so labels such as `0` and `"0"` cannot
  be distinguished in report lookups.
- pandas may rename duplicate CSV headers while parsing, so the original duplicate
  header may not appear as a `DUPLICATE_COLUMNS` validation issue.
- Mutable Python objects stored in object-typed cells are shared by pandas deep
  copies; CSV-loaded scalar data is unaffected.
- CI is configured for Python 3.11 and 3.12 only.

See [docs/detection.md](docs/detection.md) for detection method definitions,
[docs/repair.md](docs/repair.md) for repair semantics and
[docs/known-limitations.md](docs/known-limitations.md) for detailed limitations
and workarounds.

## Development

Install the development environment as described above, then run the same
quality gates used by CI:

```bash
python -m pytest
python -m pytest --cov --cov-report=term-missing
python -m ruff check .
python -m ruff format --check .
python -m mypy
```

Apply automatic formatting with:

```bash
python -m ruff format .
```

The test suite uses `pythonpath = ["src"]`, so it can run directly from a
checkout. Installing with `pip install -e ".[dev]"` provides pytest, coverage,
Ruff and mypy.

CI is configured to run pytest with coverage, Ruff linting, Ruff formatting
checks and mypy on Python 3.11 and 3.12 jobs. See `.github/workflows/ci.yml`.

Optional distribution build:

```bash
python -m pip install build
python -m build
```

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) before
opening a pull request. Use the GitHub issue templates for bug reports and
feature requests, and the pull request template for changes.

## License

MIT. See [LICENSE](LICENSE).
