# Known limitations

This document describes intentional boundaries of the current DataMendrix
foundation (`0.1.x`). They are not promises that the software will remain
limited in these ways; they clarify what is and is not validated today.

## Scope and maturity

- DataMendrix is in an early alpha stage. The current release focuses on
  inspecting datasets, not repairing them.
- The repair engine, reporting formats, HTTP API, frontend and experiment
  tracking are not implemented.
- The public API may evolve before a stable `1.0` release. Breaking changes
  should be documented in the pull request and release notes.

## Input formats and loading

- Path-based loading supports `.csv` files only. A path with another extension,
  including `.tsv` or `.csv.gz`, raises `UnsupportedFileTypeError`.
- Already-open text streams are accepted, but the library does not manage their
  lifetime or close them.
- The entire file is loaded into memory with pandas. There is no streaming,
  chunked or out-of-core CSV mode.
- Parsing uses pandas' automatic type inference. There is no user-supplied dtype
  or schema contract during loading.
- pandas may rename duplicate CSV headers while parsing. For example, `a,a`
  becomes `a,a.1`, so structural validation sees two unique labels and may not
  report the original duplicated header.
- A completely empty file raises `DatasetLoadError`. A header-only file can be
  loaded, but the default validation configuration reports `TOO_FEW_ROWS`
  because it requires at least one row.

## Missing values and duplicate rows

- Missing-value detection uses pandas semantics: `NaN`, `None`, `pandas.NA` and
  `NaT`.
- CSV fields that match pandas' default NA tokens are converted to missing
  values. Pass `keep_default_na=False` and explicit `na_values` when blank
  fields or custom sentinels must be preserved.
- Empty strings in a directly constructed `DataFrame` are not automatically
  treated as missing.
- Duplicate detection uses pandas' equality definition. Rows with identical
  missing-value patterns can therefore be duplicates.
- Duplicate reports identify row labels, not positions. With a non-unique index,
  the same label can appear more than once in `duplicate_indices`.
- Duplicate detection reports rows but does not drop, merge or repair them.

## Structural validation

- Validation is structural only. It checks row and column minimums, duplicate
  or blank column labels, optional unnamed columns and required column names.
- It does not validate data types, ranges, uniqueness constraints, referential
  integrity, outliers, semantic rules, PII or target leakage.
- The loader does not validate automatically. This is intentional so imperfect
  datasets can be loaded and audited without being rejected.
- `ValidationConfig` requires caller-supplied rules; it is not a schema
  language and does not infer constraints from the data.

## Numeric statistics and profiling

- Numeric statistics include real-valued numeric dtypes. Boolean, complex,
  datetime, timedelta and object columns such as `Decimal` are excluded.
- Missing values are excluded from numeric calculations. Infinite values are
  retained and counted separately through `infinite_count`; they can make
  statistics such as means and quartiles infinite or `NaN`.
- Statistics that need more observations than are available are `NaN`. For
  example, standard deviation needs two values, skewness needs three and
  kurtosis needs four.
- Standard deviation is the sample standard deviation (`ddof=1`).
- There is no outlier detection, robust statistics, distribution testing or
  automatic type coercion.
- Memory usage is pandas' deep memory estimate and includes the index. It is
  useful for comparison but is not an operating-system measurement.
- Profiles are point-in-time snapshots. They do not update when external data
  changes.
- Column names in reports are stringified. Labels `0` and `"0"` therefore share
  the same report name and cannot be distinguished through report lookups.
- The unique-count fallback for unhashable object cells uses pairwise equality
  and is quadratic. It is intended only for unusual, typically small in-memory
  columns.

## Dataset isolation

- `Dataset` takes a pandas deep copy on construction, and `Dataset.frame`
  returns another deep copy.
- pandas deep copies do not recursively copy mutable Python objects stored in
  object-typed cells. Lists, dictionaries and similar objects can therefore be
  shared between the original frame and a copy.
- CSV-loaded datasets contain scalar values, so this caveat normally does not
  apply to data read through `load_csv`.

## Environment and compatibility

- The package requires Python 3.11 or newer.
- Continuous integration is configured for Python 3.11 and 3.12. Other versions
  are not part of the current CI matrix even when they satisfy the package
  metadata.
- Supported pandas and NumPy versions follow the lower bounds in
  `pyproject.toml`. Behaviour may vary with substantially newer library
  releases, and CI does not currently run a full dependency-version matrix.
- CI is configured to run on Ubuntu Linux. Windows and macOS development is
  supported informally, including Windows path handling in the loader.

## Reporting and interoperability

- Reports are Python dataclasses; JSON, HTML, PDF and other serialization
  formats are not provided.
- There is no CLI, service endpoint or persisted profile format.
- There is no integration with experiment trackers, workflow orchestrators or
  ML pipelines yet.

If one of these limitations blocks your use case, open a feature request with
the problem, the expected workflow and the smallest useful API change. See
[CONTRIBUTING.md](../CONTRIBUTING.md).
