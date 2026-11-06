# Detection engine

The detection engine introduced in `0.2.x` adds read-only checks for numeric
outliers, constant or near-constant features, and target-class imbalance. Every
function returns an immutable report and never removes, replaces or repairs
values. Explicit cleaning is provided separately by the
[repair engine](repair.md).

```python
from data_mendrix import (
    analyze_class_imbalance,
    detect_constant_features,
    detect_outliers,
)
```

See [known-limitations.md](known-limitations.md) for intentional scope
boundaries.

## Outlier detection

```python
report = detect_outliers(
    dataset,
    columns=None,
    method="iqr",
    threshold=None,
)
```

`columns` accepts one label or an iterable of labels. `None` selects every
real-valued numeric column. Boolean, complex and non-numeric columns are
excluded by default. Explicitly selecting a non-numeric column raises
`InvalidParameterError`.

### Methods and default thresholds

| Method | Default threshold | Centre | Scale |
|---|---:|---|---|
| `iqr` | `1.5` | median | interquartile range |
| `zscore` | `3.0` | mean | sample standard deviation (`ddof=1`) |
| `modified_zscore` | `3.5` | median | median absolute deviation (MAD) |

Thresholds must be positive finite numbers. A `threshold=None` selects the
method default.

### IQR

For finite observations, linear quartiles `Q1` and `Q3` define the
interquartile range:

```text
IQR = Q3 - Q1
lower = Q1 - threshold * IQR
upper = Q3 + threshold * IQR
```

A value below `lower` or above `upper` is flagged. IQR evidence scores measure
the distance from the relevant quartile in IQR units:

```text
below: (Q1 - value) / IQR
above: (value - Q3) / IQR
```

If `IQR` is zero, any finite value outside the identical quartile bounds is
flagged with an infinite score.

### Z-score

The Z-score method uses the mean and sample standard deviation of finite
observations:

```text
lower = mean - threshold * std
upper = mean + threshold * std
score = abs(value - mean) / std
```

At least two finite observations are needed for a standard deviation. With fewer
observations, no finite Z-score outlier is reported.

### Modified Z-score (MAD)

The modified Z-score uses robust location and scale:

```text
MAD = median(abs(value - median))
score = 0.6745 * abs(value - median) / MAD
lower = median - (threshold / 0.6745) * MAD
upper = median + (threshold / 0.6745) * MAD
```

When `MAD` is zero, any finite value different from the median is flagged with
an infinite score. This is useful for a mostly constant column with a spike.

### Evidence and edge cases

`OutlierReport` exposes:

- `method` and `threshold`;
- the union of affected row labels in dataset order through `outlier_indices`;
- one `ColumnOutlierReport` per analysed column;
- `columns_with_outliers` and `outlier_cell_count` summaries.

Each `ColumnOutlierReport` includes observation counts, finite baseline, bounds,
outlier labels and an `evidence` tuple. Each `OutlierEvidence` contains the row
label, value, method-specific score, direction, bounds and a human-readable
reason.

Semantics:

- `NaN`, `None`, `pandas.NA` and `NaT` are missing and excluded from analysis.
- `inf` and `-inf` are always flagged as outliers and excluded from the finite
  baseline used to calculate centre, scale and bounds.
- If a column has no finite baseline, bounds are `NaN`, but infinite values are
  still flagged.
- Empty columns, all-missing columns, singleton columns and constant columns do
  not crash. No outlier is reported unless the method can establish a meaningful
  deviation.
- Reports identify row labels. A non-unique index can therefore contain repeated
  labels.
- No rows or values are removed.

## Constant and near-constant features

```python
report = detect_constant_features(
    dataset,
    columns=None,
    dominance_threshold=0.95,
)
```

All dtypes are eligible. Missing values are excluded from the non-missing
denominator, dominant count and dominant fraction.

For each column:

- `unique_count` counts distinct non-missing values;
- `dominant_value` is the most frequent non-missing value, with first occurrence
  used as the deterministic tie-break;
- `dominant_fraction = dominant_count / non_missing_count`;
- `is_constant` is true when there is exactly one distinct non-missing value;
- `is_near_constant` is true when the dominant fraction is at least
  `dominance_threshold`.

`dominance_threshold` must be in `(0, 1]`. The default `0.95` means a value must
cover at least 95% of non-missing observations. Exact constants also satisfy the
near-constant rule, so `near_constant_columns` includes `constant_columns`.

A column with no non-missing values is neither constant nor near-constant.
Missingness is reported separately through `missing_count`.

## Class imbalance

```python
report = analyze_class_imbalance(
    dataset,
    target="churn",
    imbalance_ratio_threshold=2.0,
    minority_fraction_threshold=0.1,
)
```

The target is required and must identify exactly one column. Reports include:

- counts and percentages for every observed class, ordered from most to least
  frequent; ties retain first-appearance order;
- `class_count`, `is_binary` and `is_multiclass`;
- `majority_class`, `minority_class`, `majority_fraction` and
  `minority_fraction`;
- `imbalance_ratio = majority_count / minority_count`;
- the configured thresholds and the derived `is_imbalanced` flag.

Missing target values are excluded from class counts, percentages and ratios.
They are reported as `missing_count` instead.

`is_imbalanced` requires at least two observed classes and is true when either:

```text
imbalance_ratio > imbalance_ratio_threshold
minority_fraction < minority_fraction_threshold
```

`imbalance_ratio_threshold` must be greater than `1.0` and defaults to `2.0`.
`minority_fraction_threshold` must be in `[0, 0.5]` and defaults to `0.1`. Set it
to `0.0` to disable the minority-fraction condition. Equality does not trigger
either condition.

A target with one observed class is not labelled imbalanced; it is better
described as a constant target and is covered by constant-feature detection.
An empty or all-missing target produces no classes, `NaN` fractions and ratio,
and `is_imbalanced == False`.

## Interpretation

Thresholds are configurable statistical heuristics. They do not establish that a
value is erroneous, that a feature should be dropped, or that a dataset must be
rebalanced. Use the reports as evidence for review and keep domain requirements
and downstream model behaviour in mind.

Reports and analysis are read-only. Removing outliers, dropping constant
features and resampling classes are outside the detection engine. Missing-value,
duplicate and dtype cleaning is handled by the separate
[repair engine](repair.md).
