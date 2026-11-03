"""Constant and near-constant feature detection."""

from __future__ import annotations

import math
from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from numbers import Real

import pandas as pd

from data_mendrix.core.dataset import Dataset
from data_mendrix.core.exceptions import (
    ColumnNotFoundError,
    InvalidDatasetError,
    InvalidParameterError,
)
from data_mendrix.detection._columns import resolve_column_selection, unique_column_position


@dataclass(frozen=True, slots=True)
class ConstantColumnStats:
    """Constantness statistics for one column.

    Missing values are excluded from ``non_missing_count``, ``unique_count``
    and ``dominant_fraction``. A column with no non-missing values is neither
    constant nor near-constant; missingness is reported separately.
    """

    column: str
    non_missing_count: int
    missing_count: int
    unique_count: int
    dominant_value: Hashable | None
    dominant_count: int
    dominant_fraction: float
    is_constant: bool
    is_near_constant: bool

    @property
    def has_missing(self) -> bool:
        """Whether the column contains missing values."""
        return self.missing_count > 0


@dataclass(frozen=True, slots=True)
class ConstantFeatureReport:
    """Constant and near-constant results for a dataset."""

    dominance_threshold: float
    row_count: int
    columns: tuple[ConstantColumnStats, ...]

    @property
    def column_names(self) -> tuple[str, ...]:
        """Names of the columns included in the report."""
        return tuple(column.column for column in self.columns)

    @property
    def constant_columns(self) -> tuple[ConstantColumnStats, ...]:
        """Columns with exactly one distinct non-missing value."""
        return tuple(column for column in self.columns if column.is_constant)

    @property
    def near_constant_columns(self) -> tuple[ConstantColumnStats, ...]:
        """Columns at or above the dominance threshold, including constants."""
        return tuple(column for column in self.columns if column.is_near_constant)

    @property
    def constant_column_names(self) -> tuple[str, ...]:
        """Names of exactly constant columns."""
        return tuple(column.column for column in self.constant_columns)

    @property
    def near_constant_column_names(self) -> tuple[str, ...]:
        """Names of near-constant columns, including exact constants."""
        return tuple(column.column for column in self.near_constant_columns)

    @property
    def has_constant_features(self) -> bool:
        """Whether any near-constant or constant column was found."""
        return bool(self.near_constant_columns)

    def column(self, name: Hashable) -> ConstantColumnStats:
        """Return statistics for one column by label.

        Non-string labels are matched using the string representation stored in
        the report.

        Raises
        ------
        ColumnNotFoundError
            If no column with the given label exists.
        """
        label = str(name)
        for column in self.columns:
            if column.column == label:
                return column
        raise ColumnNotFoundError((name,))


def detect_constant_features(
    dataset: Dataset,
    *,
    columns: Hashable | Iterable[Hashable] | None = None,
    dominance_threshold: float = 0.95,
) -> ConstantFeatureReport:
    """Detect constant and near-constant columns without modifying the dataset.

    A column is exactly constant when it has one distinct non-missing value.
    It is near-constant when its most frequent non-missing value accounts for at
    least ``dominance_threshold`` of all non-missing values. Exact constants
    therefore also satisfy the near-constant rule.

    Parameters
    ----------
    dataset:
        Dataset to inspect.
    columns:
        Optional label or iterable of labels. By default all columns are
        inspected, regardless of dtype.
    dominance_threshold:
        Required minimum fraction for the dominant non-missing value. Must be in
        the interval ``(0, 1]``. Defaults to ``0.95``.

    Returns
    -------
    ConstantFeatureReport
        Immutable per-column statistics and filtered column collections.

    Raises
    ------
    InvalidDatasetError
        If ``dataset`` is not a :class:`~data_mendrix.core.dataset.Dataset`.
    InvalidParameterError
        If the threshold, column selection or selected label is invalid.
    ColumnNotFoundError
        If a selected column does not exist.

    Examples
    --------
    >>> report = detect_constant_features(dataset, dominance_threshold=0.95)
    >>> report.near_constant_column_names
    ('country',)
    """
    if not isinstance(dataset, Dataset):
        raise InvalidDatasetError(
            f"detect_constant_features() expects a Dataset, got {type(dataset).__name__}."
        )
    validated_threshold = _validate_dominance_threshold(dominance_threshold)
    frame = dataset._view()
    positions = _resolve_constant_positions(frame, columns)

    columns_report = tuple(
        _column_stats(
            frame.iloc[:, position],
            column=str(frame.columns[position]),
            dominance_threshold=validated_threshold,
        )
        for position in positions
    )

    return ConstantFeatureReport(
        dominance_threshold=validated_threshold,
        row_count=int(frame.shape[0]),
        columns=columns_report,
    )


def _resolve_constant_positions(
    frame: pd.DataFrame,
    columns: Hashable | Iterable[Hashable] | None,
) -> list[int]:
    """Resolve column positions for constant-feature detection."""
    selection = resolve_column_selection(frame, columns)
    if selection is None:
        return list(range(frame.shape[1]))
    return [unique_column_position(frame, label) for label in selection]


def _validate_dominance_threshold(value: float) -> float:
    """Validate the dominance fraction."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise InvalidParameterError("dominance_threshold must be in the interval (0, 1].")
    threshold = float(value)
    if not math.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        raise InvalidParameterError("dominance_threshold must be in the interval (0, 1].")
    return threshold


def _column_stats(
    series: pd.Series,
    *,
    column: str,
    dominance_threshold: float,
) -> ConstantColumnStats:
    """Build constantness statistics for one column."""
    non_missing = series.dropna()
    non_missing_count = int(non_missing.size)
    counts = non_missing.value_counts(sort=False)
    unique_count = int(counts.size)

    if unique_count == 0:
        dominant_value: Hashable | None = None
        dominant_count = 0
        dominant_fraction = 0.0
    else:
        dominant_value = counts.idxmax()
        dominant_count = int(counts.max())
        dominant_fraction = dominant_count / non_missing_count

    return ConstantColumnStats(
        column=column,
        non_missing_count=non_missing_count,
        missing_count=int(series.isna().sum()),
        unique_count=unique_count,
        dominant_value=dominant_value,
        dominant_count=dominant_count,
        dominant_fraction=dominant_fraction,
        is_constant=unique_count == 1,
        is_near_constant=non_missing_count > 0 and dominant_fraction >= dominance_threshold,
    )
