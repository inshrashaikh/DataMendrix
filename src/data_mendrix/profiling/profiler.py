"""Combined dataset profiling.

A profile is a point-in-time, read-only snapshot that composes schema details,
missing-value statistics, numeric descriptive statistics and duplicate-row
detection into one report.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from data_mendrix.core.dataset import Dataset
from data_mendrix.core.exceptions import ColumnNotFoundError, InvalidDatasetError
from data_mendrix.detection.duplicates import DuplicateReport, detect_duplicates
from data_mendrix.profiling.missing import MissingValueReport, missing_value_report
from data_mendrix.profiling.numeric import NumericStatistics, numeric_statistics


@dataclass(frozen=True, slots=True)
class ColumnProfile:
    """Schema and completeness details for one column."""

    name: str
    dtype: str
    non_missing_count: int
    missing_count: int
    missing_fraction: float
    unique_count: int


@dataclass(frozen=True, slots=True)
class DatasetProfile:
    """Comprehensive read-only profile of a dataset."""

    name: str
    source: str | None
    row_count: int
    column_count: int
    memory_usage_bytes: int
    columns: tuple[ColumnProfile, ...]
    missing: MissingValueReport
    numeric: NumericStatistics
    duplicates: DuplicateReport

    @property
    def numeric_column_names(self) -> tuple[str, ...]:
        """Names of columns included in the numeric statistics section."""
        return self.numeric.column_names

    def column(self, name: Hashable) -> ColumnProfile:
        """Return the profile for one column by label.

        Non-string labels are matched using the string representation stored in
        the report.

        Raises
        ------
        ColumnNotFoundError
            If no column with the given label exists.
        """
        label = str(name)
        for column in self.columns:
            if column.name == label:
                return column
        raise ColumnNotFoundError((name,))


def profile_dataset(dataset: Dataset) -> DatasetProfile:
    """Build a full profile of a dataset.

    Memory usage is measured with ``DataFrame.memory_usage(deep=True)`` and
    includes the index. Unique counts ignore missing values. The dataset itself
    is never modified.

    Parameters
    ----------
    dataset:
        Dataset to profile.

    Returns
    -------
    DatasetProfile
        Immutable snapshot combining schema, missing values, numeric statistics
        and duplicate rows.

    Raises
    ------
    InvalidDatasetError
        If ``dataset`` is not a :class:`~data_mendrix.core.dataset.Dataset`.

    Examples
    --------
    >>> profile = profile_dataset(dataset)
    >>> profile.row_count, profile.column_count
    (100, 8)
    >>> profile.missing.missing_cell_fraction
    0.0425
    """
    if not isinstance(dataset, Dataset):
        raise InvalidDatasetError(
            f"profile_dataset() expects a Dataset, got {type(dataset).__name__}."
        )

    frame = dataset._view()
    row_count, column_count = frame.shape
    missing = missing_value_report(dataset)

    columns: list[ColumnProfile] = []
    for position, label in enumerate(frame.columns):
        missing_stats = missing.columns[position]
        columns.append(
            ColumnProfile(
                name=str(label),
                dtype=str(frame.dtypes.iloc[position]),
                non_missing_count=missing_stats.non_missing_count,
                missing_count=missing_stats.missing_count,
                missing_fraction=missing_stats.missing_fraction,
                unique_count=_unique_count(frame.iloc[:, position]),
            )
        )

    return DatasetProfile(
        name=dataset.name,
        source=dataset.source,
        row_count=row_count,
        column_count=column_count,
        memory_usage_bytes=int(frame.memory_usage(deep=True).sum()),
        columns=tuple(columns),
        missing=missing,
        numeric=numeric_statistics(dataset),
        duplicates=detect_duplicates(dataset),
    )


def _unique_count(series: pd.Series) -> int:
    """Count non-missing distinct values, including unhashable object cells."""
    non_missing = series.dropna()
    try:
        return int(non_missing.nunique())
    except TypeError:
        # pandas cannot hash list/dict cells. Fall back to equality comparison;
        # this is quadratic but only applies to unusual, typically small columns.
        distinct: list[object] = []
        for value in non_missing:
            if not any(_values_equal(value, existing) for existing in distinct):
                distinct.append(value)
        return len(distinct)


def _values_equal(left: object, right: object) -> bool:
    """Compare two possibly array-like values without ambiguity errors."""
    try:
        result = left == right
    except (TypeError, ValueError):
        return False
    if isinstance(result, bool):
        return result
    try:
        return bool(np.all(result))
    except (TypeError, ValueError):
        return False
