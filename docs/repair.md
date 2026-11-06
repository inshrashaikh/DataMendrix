# Cleaning and repair engine

The `0.3.x` repair engine turns explicit cleaning decisions into a new
`Dataset`. Repairs are never implicit: every operation is a value object, every
plan is ordered, previews show the proposed changes, and application returns an
immutable report.

```python
from data_mendrix import (
    ConvertTypes,
    DropMissing,
    FillMissing,
    RemoveDuplicates,
    apply_repair,
    preview_repair,
)
```

See [known-limitations.md](known-limitations.md) for intentional scope
boundaries.

## Plans, previews and results

A plan is one operation or an iterable of operations:

```python
plan = [
    DropMissing(axis="rows", columns=["customer_id"], how="any"),
    FillMissing(columns=["age", "income"], strategy="median"),
    FillMissing(columns=["segment"], strategy="mode"),
    RemoveDuplicates(subset=["customer_id"], keep="first"),
    ConvertTypes(conversions={"age": "Int64"}, errors="coerce"),
]

preview = preview_repair(dataset, plan)
result = apply_repair(dataset, plan)
```

`preview_repair` runs the plan on an internal defensive copy and returns a
`RepairReport`. It does not return or expose the repaired data. `apply_repair`
runs the same plan and returns a `RepairResult` with two attributes:

- `result.dataset`: a new `Dataset` preserving `name` and `source` metadata.
- `result.report`: the same `RepairReport` that a preview produces.

The input dataset is never modified. Operations see the effects of earlier
operations in the plan, and an empty iterable produces an unchanged copy with an
empty report. If an operation raises, no partial `RepairResult` is returned and
the original dataset remains unchanged.

Single-operation conveniences are also available:

```python
result = fill_missing(dataset, columns=["age"], strategy="median")
result = drop_missing(dataset, columns=["customer_id"])
result = remove_duplicates(dataset, subset=["customer_id"])
result = convert_dtypes(dataset, conversions={"age": "Int64"})
```

## Missing-value repair

### Drop rows or columns

```python
DropMissing(
    axis="rows",  # "rows" or "columns"
    columns=None,  # optional label or iterable of labels
    how="any",  # "any" or "all"
)
```

`axis="rows"` removes rows according to missingness in the selected columns.
`axis="columns"` removes columns that satisfy the missingness rule; unselected
columns are retained even when they contain missing values.

- `how="any"` means at least one considered value is missing.
- `how="all"` means every considered value is missing.
- `columns=None` considers every column.
- Duplicate labels in `columns` are only considered once.
- When no columns are considered (a zero-column dataset with
  `columns=None`), `how="any"` drops nothing and `how="all"` drops every row,
  matching `pandas.DataFrame.dropna`'s vacuous-truth semantics.

Missing values follow pandas semantics, including `NaN`, `None`, `pandas.NA`
and `NaT`. Empty datasets and zero-column selections are handled without error.

### Fill missing values

```python
FillMissing(
    columns=None,  # optional label or iterable of labels
    strategy="mean",  # "mean", "median", "mode" or "constant"
    value=None,  # required only for "constant"
)
```

| Strategy | Default scope when `columns=None` | Fill value |
|---|---|---|
| `mean` | numeric columns | arithmetic mean of finite non-missing values |
| `median` | numeric columns | median of finite non-missing values |
| `mode` | every column | most frequent non-missing value |
| `constant` | every column | the explicit `value` |

Rules and edge cases:

- `mean` and `median` explicitly selected non-numeric columns raise
  `InvalidParameterError`. A non-numeric column is not changed by a default
  mean/median fill.
- Infinite values are excluded from the finite baseline used by mean and median.
  If no finite values remain, the column is left unchanged and a `RepairIssue` is
  recorded.
- `mode` uses the first value in source order when multiple values tie. If every
  value is missing, the column is left unchanged and an issue is recorded.
- `value` must be a non-missing scalar for `constant`; it must be `None` for all
  other strategies. Incompatible fill values are recorded as issues rather than
  silently modifying the column.
- A fill can change a column's dtype; the report records before/after dtypes.
- Columns with no missing values are no-ops and do not appear as affected
  columns.

## Duplicate removal

```python
RemoveDuplicates(
    subset=None,  # optional label or iterable of labels
    keep="first",  # "first", "last" or False
)
```

- `subset=None` compares whole rows.
- `keep="first"` keeps the first occurrence of each comparison key.
- `keep="last"` keeps the last occurrence.
- `keep=False` removes every row that participates in a duplicate group.

Missing values compare as equal, so identical missing-value patterns can form
duplicate groups. Reports list the removed row labels and the compared columns
only when a change occurs.

## Data-type repair

```python
ConvertTypes(
    conversions={"age": "Int64", "joined": "datetime64[ns]"},
    errors="raise",  # "raise", "skip" or "coerce"
)
```

`conversions` maps column labels to pandas-compatible targets such as `"int64"`,
`"float64"`, `"Int64"`, `"string"`, `"category"`, `"boolean"`,
`"datetime64[ns]"` or NumPy dtype objects. Targets are validated before any
conversion is attempted.

| `errors` | Behaviour on a failed conversion |
|---|---|
| `raise` | Raise `RepairConversionError` with the column and target attributes. The original pandas exception is chained as the cause. |
| `skip` | Leave that column unchanged, record a `RepairIssue`, and continue with other conversions. |
| `coerce` | Replace values that cannot be cast with missing values, then apply the target dtype. |

Coercion limits:

- `coerce` can only use a target dtype that can represent missing values. For
  example, `"Int64"` can represent missing values but exact `"int64"` cannot; the
  latter raises `RepairConversionError` when coercion would introduce a missing
  value.
- Existing missing values must remain missing after conversion. A conversion
  that would silently turn an existing missing value into data is rejected.
- Numeric, datetime and timedelta targets use pandas' conversion rules. Other
  targets use a per-value fallback.
- Timezone-aware datetime targets are supported: `errors="coerce"` parses
  values with `pandas.to_datetime`, then localises timezone-naive results to
  the target timezone or converts timezone-aware results to it.
- A conversion that leaves both dtype and values unchanged is a no-op and is not
  reported as changed.

`RepairConversionError` derives from both `RepairError` and `ValueError`, so
callers can catch the library base class or an existing `ValueError` idiom.

## Report semantics

`RepairReport` describes an entire plan. Its main properties are:

- `operations`: one `RepairOperationReport` per operation, in plan order.
- `changed`: whether any operation changed data or schema.
- `change_count`: the sum of each operation's primary change count.
- `affected_rows`: the union of affected row labels, in plan order.
- `affected_columns`: the union of affected column names, in plan order.
- `issues`: all non-fatal issues across the plan.
- `row_count_before`, `row_count_after`, `column_count_before`,
  `column_count_after`, plus delta properties.

Each `RepairOperationReport` records:

- `action`: a stable `RepairAction` value.
- `change_count`: rows removed, columns removed, cells filled or columns
  converted, depending on the operation.
- `affected_rows` and `affected_columns`.
- `column_changes`: `ColumnChange` entries with before/after dtypes, fill values
  and cell counts.
- `issues`: non-fatal problems such as an undefined fill statistic or a skipped
  conversion. `error_type` reports a stable, backend-independent class name:
  DataMendrix errors keep their own name and third-party exceptions are mapped
  to the nearest built-in exception (for example `"ValueError"`).
- `detail`: a human-readable summary of the operation scope and policy.

All report and operation objects are frozen dataclasses with slots. Collections
are tuples. Reports are snapshots: changing the original DataFrame after a repair
does not alter a returned report or repaired dataset.

## Edge cases

- Duplicate labels in a selection (e.g. `columns=["a", "a"]`) are considered
  only once; explicit duplicate labels on the frame itself are rejected as
  ambiguous. Row-label unions in reports are de-duplicated in plan order,
  treating all missing row labels as one and supporting unhashable labels
  (compared pairwise).
- When no columns are considered (a zero-column dataset with `columns=None`),
  `axis="rows", how="all"` drops every row, matching `pandas.dropna`'s
  vacuous-truth semantics; `how="any"` drops nothing.
- Empty datasets are supported by all repair operations.
- `NaN`, `None`, `pandas.NA`, `NaT`, and infinities are handled according to the
  rules above.
- Unknown columns raise `ColumnNotFoundError`; ambiguous duplicate column labels
  raise `InvalidParameterError`.
- Empty column selections and invalid operation options raise
  `InvalidParameterError`.
- A non-`Dataset` input raises `InvalidDatasetError`.
- Invalid operation objects in a plan raise `InvalidParameterError`.

## Related documentation

- [Detection engine](detection.md)
- [Known limitations](known-limitations.md)
- [README](../README.md)
