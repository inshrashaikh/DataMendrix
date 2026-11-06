# Known limitations

This document describes intentional boundaries of the current DataMendrix
foundation, detection engine and repair engine (`0.3.x`). They are not promises
that the software will remain limited in these ways; they clarify what is and is
not validated today.

## Scope and maturity

- DataMendrix is in an early alpha stage. The current release covers inspection
  and explicit, in-memory cleaning of tabular datasets.
- Reporting formats, the HTTP API, frontend and experiment tracking are not
  implemented.
- Repairs are caller-specified operations rather than automatic recommendations.
  The library does not infer a cleaning policy from the data.
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
  Explicit removal is available through `RemoveDuplicates` or
  `remove_duplicates`.

## Repair engine

- Repairs are explicit transformations. The library does not suggest which
  missing values, duplicates or data types to repair.
- A repair plan is applied in memory to a defensive copy. There is no streaming,
  chunked, out-of-core or transactional repair mode.
- Operations support pandas-compatible dtype targets, but there is no
  user-supplied per-value conversion callback or schema language.
- `mean` and `median` fills use finite non-missing values only and are computed
  in `float64`. Integers beyond `2**53` may lose precision in the fill value.
  If no finite baseline exists, the column is left unchanged and a `RepairIssue`
  is recorded.
- `mode` falls back to first-appearance order for ties and cannot fill a column
  whose values are all missing or whose most frequent value is incompatible with
  `fillna`.
- `constant` fills require a non-missing scalar. A fill that is incompatible with
  the column dtype is recorded as an issue rather than silently discarded.
- `coerce` conversion can only introduce missing values when the target dtype
  supports them. Exact integer and NumPy boolean targets may therefore raise
  `RepairConversionError` instead of producing missing values.
- Conversions that would turn existing missing values into non-missing data are
  rejected. Dtype repair delegates parsing semantics to pandas; it does not
  implement a separate CSV or date parser.
- Repair reports describe the plan that ran. They do not prove that the resulting
  dataset is semantically valid, unbiased or suitable for a particular model.
- Duplicate labels in a selection are considered only once. With no considered
  columns (a zero-column dataset), `axis="rows", how="all"` drops every row,
  matching `pandas.dropna`'s vacuous-truth semantics; `how="any"` drops
  nothing. Report row-label unions de-duplicate in plan order and treat all
  missing row labels as one; unhashable labels (e.g. lists in an object index)
  are compared pairwise.
- Timezone-aware datetime targets are coerced by localising timezone-naive
  results to the target timezone or converting timezone-aware results to it.
- There is no persisted repair history, undo log or rollback mechanism. Callers
  should retain the original dataset when they need to compare or revert.
- Repair does not implicitly validate its output. Run `validate_dataset`,
  detection APIs and task-specific checks after cleaning as needed.

## Structural validation

- Validation is structural only. It checks row and column minimums, duplicate
  or blank column labels, optional unnamed columns and required column names.
- It does not validate data types, ranges, uniqueness constraints, referential
  integrity, outliers, semantic rules, PII or target leakage.
- Statistical checks are separate read-only detection APIs. They produce
  evidence for review; they do not validate, repair or reject a dataset.
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
- There is no distribution testing, automatic type coercion or multivariate
  anomaly detection.
- Memory usage is pandas' deep memory estimate and includes the index. It is
  useful for comparison but is not an operating-system measurement.
- Profiles are point-in-time snapshots. They do not update when external data
  changes.
- Column names in reports are stringified. Labels `0` and `"0"` therefore share
  the same report name and cannot be distinguished through report lookups.
- The unique-count fallback for unhashable object cells uses pairwise equality
  and is quadratic. It is intended only for unusual, typically small in-memory
  columns.

## Detection engine

- Outlier detection is univariate and applies to real-valued numeric columns
  only. Boolean, complex, datetime, timedelta and non-numeric object columns are
  excluded by default; explicitly selecting a non-numeric column raises
  `InvalidParameterError`.
- There is no multivariate, contextual, time-series or model-based anomaly
  detection. Correlations and row-level combinations are not considered.
- Missing values are excluded from outlier baselines and evidence. Infinite
  values are always flagged as outliers and excluded from the finite baseline.
  If a column has no finite values, its bounds are `NaN` while infinities remain
  flagged.
- IQR uses linear quartiles. If the interquartile range is zero, any finite value
  outside the identical quartile bounds is flagged with an infinite score.
- Z-score uses the sample standard deviation (`ddof=1`) and needs at least two
  finite observations. With fewer observations, no finite Z-score outlier is
  reported.
- Modified Z-score uses an unscaled MAD and the conventional `0.6745`
  multiplier. If MAD is zero, finite values different from the median are
  flagged with an infinite score.
- Thresholds are configurable statistical heuristics. Outliers are review
  candidates, not automatically invalid values, and the library does not remove
  or replace them.
- Constant-feature detection excludes missing values from the dominance
  denominator. A column with no non-missing values is neither constant nor
  near-constant. Exact constants are also included in the near-constant
  collection.
- Constant-feature detection and class-imbalance analysis rely on pandas value
  counting, so unhashable object cells or class labels such as lists are not
  supported.
- Class-imbalance analysis requires an explicit, unique target column. Missing
  target values are excluded from counts, percentages and ratios, and are
  reported separately.
- Imbalance thresholds are caller-configured heuristics. At least two observed
  classes are required; a one-class target is not labelled imbalanced. A ratio
  equal to the ratio threshold and a minority fraction equal to the fraction
  threshold do not trigger detection.
- The detection engine does not drop constant features, remove outliers,
  resample classes or create train/test splits. Detection remains read-only;
  explicit cleaning is handled by the separate repair engine.

## Dataset isolation

- `Dataset` takes a pandas deep copy on construction, and `Dataset.frame`
  returns another deep copy. Repaired datasets follow the same rule.
- pandas deep copies do not recursively copy mutable Python objects stored in
  object-typed cells. Lists, dictionaries and similar objects can therefore be
  shared between the original frame and a copy, including the dataset returned
  by a repair.
- CSV-loaded datasets contain scalar values, so this caveat normally does not
  apply to data read through `load_csv`.

## Environment and compatibility

- The package requires Python 3.11 or newer.
- The test suite passes on Python 3.11, 3.12, 3.13 and 3.14 against both
  pandas 2.3.x and pandas 3.x. Continuous integration is configured for Python
  3.11 and 3.12 only, without a dependency-version matrix.
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
