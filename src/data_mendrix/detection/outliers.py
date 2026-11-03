"""Univariate outlier detection for numeric columns.

Three complementary methods are supported:

- ``iqr`` uses the interquartile range and Tukey-style fences.
- ``zscore`` compares absolute standardised deviations from the mean.
- ``modified_zscore`` uses the median absolute deviation (MAD), which is less
  sensitive to extreme values than the mean and standard deviation.

Detection is read-only. The reports contain affected row labels and evidence
explaining every flagged value.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from numbers import Real
from typing import Literal

import numpy as np
import pandas as pd

from data_mendrix.core.dataset import Dataset
from data_mendrix.core.exceptions import (
    ColumnNotFoundError,
    InvalidDatasetError,
    InvalidParameterError,
)
from data_mendrix.detection._columns import resolve_column_selection, unique_column_position

OutlierMethod = Literal["iqr", "zscore", "modified_zscore"]
OutlierDirection = Literal["below", "above"]

_DEFAULT_THRESHOLDS: dict[OutlierMethod, float] = {
    "iqr": 1.5,
    "zscore": 3.0,
    "modified_zscore": 3.5,
}
_METHOD_LABELS: dict[OutlierMethod, str] = {
    "iqr": "IQR",
    "zscore": "Z-score",
    "modified_zscore": "modified Z-score (MAD)",
}
_MAD_SCALE = 0.6745


@dataclass(frozen=True, slots=True)
class OutlierEvidence:
    """Explanation for one value flagged by an outlier method.

    Attributes
    ----------
    row_index:
        Index label of the affected row.
    value:
        The observed numeric value, including ``inf`` or ``-inf``.
    score:
        Method-specific score. For IQR this is the distance from the relevant
        quartile in IQR units; for Z-score and modified Z-score it is the
        absolute score used for thresholding. Infinite values have ``inf``.
    direction:
        Whether the value is below or above the centre of the distribution.
    lower_bound:
        Lower decision bound, or ``nan`` when no finite baseline exists.
    upper_bound:
        Upper decision bound, or ``nan`` when no finite baseline exists.
    reason:
        Human-readable explanation suitable for reports and logs.
    """

    row_index: Hashable
    value: float
    score: float
    direction: OutlierDirection
    lower_bound: float
    upper_bound: float
    reason: str


@dataclass(frozen=True, slots=True)
class ColumnOutlierReport:
    """Outlier statistics and evidence for one numeric column."""

    column: str
    method: OutlierMethod
    threshold: float
    row_count: int
    observed_count: int
    finite_count: int
    missing_count: int
    infinite_count: int
    center: float
    scale: float
    lower_bound: float
    upper_bound: float
    outlier_count: int
    outlier_indices: tuple[Hashable, ...]
    evidence: tuple[OutlierEvidence, ...]

    @property
    def has_outliers(self) -> bool:
        """Whether at least one value was flagged."""
        return self.outlier_count > 0

    @property
    def outlier_fraction(self) -> float:
        """Fraction of non-missing observations that were flagged."""
        if self.observed_count == 0:
            return 0.0
        return self.outlier_count / self.observed_count


@dataclass(frozen=True, slots=True)
class OutlierReport:
    """Outlier results for every analysed column in a dataset."""

    method: OutlierMethod
    threshold: float
    row_count: int
    columns: tuple[ColumnOutlierReport, ...]
    outlier_indices: tuple[Hashable, ...]

    @property
    def has_outliers(self) -> bool:
        """Whether any analysed column contains an outlier."""
        return bool(self.outlier_indices)

    @property
    def column_names(self) -> tuple[str, ...]:
        """Names of the columns included in the report."""
        return tuple(column.column for column in self.columns)

    @property
    def columns_with_outliers(self) -> tuple[ColumnOutlierReport, ...]:
        """Column reports containing at least one flagged value."""
        return tuple(column for column in self.columns if column.has_outliers)

    @property
    def outlier_cell_count(self) -> int:
        """Number of flagged column values across all analysed columns."""
        return sum(column.outlier_count for column in self.columns)

    def column(self, name: Hashable) -> ColumnOutlierReport:
        """Return the report for one numeric column by label.

        Non-string labels are matched using the string representation stored in
        the report.

        Raises
        ------
        ColumnNotFoundError
            If the label is not a numeric column in the report.
        """
        label = str(name)
        for column in self.columns:
            if column.column == label:
                return column
        raise ColumnNotFoundError((name,))


def detect_outliers(
    dataset: Dataset,
    *,
    columns: Hashable | Iterable[Hashable] | None = None,
    method: OutlierMethod = "iqr",
    threshold: float | None = None,
) -> OutlierReport:
    """Detect univariate numeric outliers without modifying the dataset.

    Parameters
    ----------
    dataset:
        Dataset to inspect.
    columns:
        Optional column label or iterable of numeric labels. By default every
        eligible numeric column is analysed. Boolean and complex columns are
        excluded. Explicitly selecting a non-numeric column raises
        :class:`InvalidParameterError`.
    method:
        ``"iqr"``, ``"zscore"`` or ``"modified_zscore"``. Defaults to ``"iqr"``.
    threshold:
        Method-specific positive threshold. Defaults are ``1.5`` for IQR,
        ``3.0`` for Z-score and ``3.5`` for modified Z-score.

    Returns
    -------
    OutlierReport
        Immutable per-column statistics, row labels and evidence. Missing values
        are excluded from calculations. Infinities are reported separately and
        flagged as outliers; they are excluded from the finite baseline used to
        derive bounds.

    Raises
    ------
    InvalidDatasetError
        If ``dataset`` is not a :class:`~data_mendrix.core.dataset.Dataset`.
    InvalidParameterError
        If a parameter is invalid or an explicitly selected column is not
        numeric.
    ColumnNotFoundError
        If a selected column does not exist.

    Examples
    --------
    >>> report = detect_outliers(dataset, method="iqr", threshold=1.5)
    >>> report.outlier_indices
    (12, 48)
    """
    if not isinstance(dataset, Dataset):
        raise InvalidDatasetError(
            f"detect_outliers() expects a Dataset, got {type(dataset).__name__}."
        )
    if not isinstance(method, str) or method not in _DEFAULT_THRESHOLDS:
        raise InvalidParameterError("method must be one of: 'iqr', 'zscore', 'modified_zscore'.")

    validated_method = method
    validated_threshold = _resolve_threshold(validated_method, threshold)
    frame = dataset._view()
    positions = _resolve_outlier_positions(frame, columns)

    row_mask = np.zeros(frame.shape[0], dtype=bool)
    reports: list[ColumnOutlierReport] = []
    for position in positions:
        report, evidence_positions = _analyze_outlier_column(
            frame.iloc[:, position],
            column=str(frame.columns[position]),
            method=validated_method,
            threshold=validated_threshold,
        )
        reports.append(report)
        for evidence_position in evidence_positions:
            row_mask[evidence_position] = True

    return OutlierReport(
        method=validated_method,
        threshold=validated_threshold,
        row_count=int(frame.shape[0]),
        columns=tuple(reports),
        outlier_indices=tuple(frame.index[row_mask].tolist()),
    )


def _resolve_outlier_positions(
    frame: pd.DataFrame,
    columns: Hashable | Iterable[Hashable] | None,
) -> list[int]:
    """Resolve numeric column positions to analyse."""
    selection = resolve_column_selection(frame, columns)
    if selection is None:
        return [
            position
            for position in range(frame.shape[1])
            if _is_numeric_candidate(frame.iloc[:, position])
        ]

    positions: list[int] = []
    non_numeric: list[str] = []
    for label in selection:
        position = unique_column_position(frame, label)
        if _is_numeric_candidate(frame.iloc[:, position]):
            positions.append(position)
        else:
            non_numeric.append(str(label))

    if non_numeric:
        formatted = ", ".join(repr(column) for column in non_numeric)
        raise InvalidParameterError(
            f"Outlier detection requires numeric columns; not numeric: {formatted}."
        )

    return positions


def _is_numeric_candidate(series: pd.Series) -> bool:
    """Return whether a series is eligible for numeric outlier detection."""
    dtype = series.dtype
    return bool(
        pd.api.types.is_numeric_dtype(dtype)
        and not pd.api.types.is_bool_dtype(dtype)
        and not pd.api.types.is_complex_dtype(dtype)
    )


def _resolve_threshold(method: OutlierMethod, threshold: float | None) -> float:
    """Resolve and validate a method-specific outlier threshold."""
    if threshold is None:
        return _DEFAULT_THRESHOLDS[method]
    if isinstance(threshold, bool) or not isinstance(threshold, Real):
        raise InvalidParameterError("threshold must be a positive finite number.")
    value = float(threshold)
    if not math.isfinite(value) or value <= 0.0:
        raise InvalidParameterError("threshold must be a positive finite number.")
    return value


def _analyze_outlier_column(
    series: pd.Series,
    *,
    column: str,
    method: OutlierMethod,
    threshold: float,
) -> tuple[ColumnOutlierReport, tuple[int, ...]]:
    """Analyse one numeric series and return its report and row positions."""
    values = series.to_numpy(dtype="float64", na_value=np.nan)
    missing_mask = np.isnan(values)
    infinite_mask = np.isinf(values)
    finite_mask = ~missing_mask & ~infinite_mask
    finite_values = values[finite_mask]

    center, scale, lower_bound, upper_bound, q1, q3 = _baseline(
        finite_values,
        method=method,
        threshold=threshold,
    )

    evidence: list[OutlierEvidence] = []
    evidence_positions: list[int] = []
    for position, value in enumerate(values):
        if missing_mask[position]:
            continue

        if infinite_mask[position]:
            infinite_direction: OutlierDirection = "above" if value > 0 else "below"
            numeric_value = float(value)
            evidence.append(
                OutlierEvidence(
                    row_index=series.index[position],
                    value=numeric_value,
                    score=math.inf,
                    direction=infinite_direction,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    reason=(
                        f"value {numeric_value!r} is non-finite and is treated as an outlier "
                        f"by {_METHOD_LABELS[method]}."
                    ),
                )
            )
            evidence_positions.append(position)
            continue

        direction = _outlier_direction(value, lower_bound, upper_bound)
        if direction is None:
            continue

        numeric_value = float(value)
        score = _outlier_score(
            numeric_value,
            direction=direction,
            method=method,
            center=center,
            scale=scale,
            q1=q1,
            q3=q3,
        )
        evidence.append(
            OutlierEvidence(
                row_index=series.index[position],
                value=numeric_value,
                score=score,
                direction=direction,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                reason=_outlier_reason(
                    numeric_value,
                    direction=direction,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    score=score,
                    threshold=threshold,
                    method=method,
                ),
            )
        )
        evidence_positions.append(position)

    return (
        ColumnOutlierReport(
            column=column,
            method=method,
            threshold=threshold,
            row_count=int(values.size),
            observed_count=int(values.size - missing_mask.sum()),
            finite_count=int(finite_values.size),
            missing_count=int(missing_mask.sum()),
            infinite_count=int(infinite_mask.sum()),
            center=center,
            scale=scale,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            outlier_count=len(evidence),
            outlier_indices=tuple(evidence_item.row_index for evidence_item in evidence),
            evidence=tuple(evidence),
        ),
        tuple(evidence_positions),
    )


def _baseline(
    finite_values: np.ndarray,
    *,
    method: OutlierMethod,
    threshold: float,
) -> tuple[float, float, float, float, float, float]:
    """Return centre, scale, bounds and quartiles for a finite baseline."""
    if finite_values.size == 0:
        nan = math.nan
        return nan, nan, nan, nan, nan, nan

    if method == "iqr":
        q1, q3 = (float(value) for value in np.quantile(finite_values, [0.25, 0.75]))
        center = float(np.median(finite_values))
        scale = q3 - q1
        return center, scale, q1 - threshold * scale, q3 + threshold * scale, q1, q3

    if method == "zscore":
        center = float(np.mean(finite_values))
        scale = float(np.std(finite_values, ddof=1)) if finite_values.size >= 2 else math.nan
        if not math.isfinite(scale):
            return center, scale, math.nan, math.nan, math.nan, math.nan
        return (
            center,
            scale,
            center - threshold * scale,
            center + threshold * scale,
            math.nan,
            math.nan,
        )

    center = float(np.median(finite_values))
    scale = float(np.median(np.abs(finite_values - center)))
    spread = threshold * scale / _MAD_SCALE if math.isfinite(scale) else math.nan
    return center, scale, center - spread, center + spread, math.nan, math.nan


def _outlier_direction(
    value: float,
    lower_bound: float,
    upper_bound: float,
) -> OutlierDirection | None:
    """Classify a finite value against the decision bounds."""
    if value < lower_bound:
        return "below"
    if value > upper_bound:
        return "above"
    return None


def _outlier_score(
    value: float,
    *,
    direction: OutlierDirection,
    method: OutlierMethod,
    center: float,
    scale: float,
    q1: float,
    q3: float,
) -> float:
    """Return the method-specific score for a flagged value."""
    if method == "iqr":
        if scale <= 0.0 or not math.isfinite(scale):
            return math.inf
        return (q1 - value) / scale if direction == "below" else (value - q3) / scale

    if method == "zscore":
        if scale <= 0.0 or not math.isfinite(scale):
            return math.inf
        return abs(value - center) / scale

    if scale <= 0.0 or not math.isfinite(scale):
        return math.inf
    return _MAD_SCALE * abs(value - center) / scale


def _outlier_reason(
    value: float,
    *,
    direction: OutlierDirection,
    lower_bound: float,
    upper_bound: float,
    score: float,
    threshold: float,
    method: OutlierMethod,
) -> str:
    """Build a stable human-readable explanation for a finite outlier."""
    if direction == "below":
        relation = f"below the lower bound {lower_bound!r}"
    else:
        relation = f"above the upper bound {upper_bound!r}"
    return (
        f"value {value!r} is {relation}; {_METHOD_LABELS[method]} score "
        f"{score!r} exceeds or meets threshold {threshold!r}."
    )
