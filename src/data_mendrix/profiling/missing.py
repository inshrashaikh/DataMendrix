"""Missing-value statistics.

Missing values follow pandas' standard definition: ``NaN``, ``None``,
``pandas.NA`` and ``NaT``. Empty strings are not treated as missing unless the
loader is configured to parse them as such via ``na_values``.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass

from data_mendrix.core.dataset import Dataset
from data_mendrix.core.exceptions import ColumnNotFoundError, InvalidDatasetError


@dataclass(frozen=True, slots=True)
class ColumnMissingStats:
    """Missing-value counts for a single column."""

    column: str
    missing_count: int
    non_missing_count: int
    missing_fraction: float

    @property
    def total_count(self) -> int:
        """Total number of cells in the column."""
        return self.missing_count + self.non_missing_count

    @property
    def has_missing(self) -> bool:
        """Whether the column contains at least one missing value."""
        return self.missing_count > 0


@dataclass(frozen=True, slots=True)
class MissingValueReport:
    """Dataset-wide and per-column missing-value statistics."""

    row_count: int
    column_count: int
    total_cell_count: int
    missing_cell_count: int
    missing_cell_fraction: float
    rows_with_missing: int
    complete_row_count: int
    columns: tuple[ColumnMissingStats, ...]

    @property
    def is_complete(self) -> bool:
        """Whether the dataset has no missing cells at all."""
        return self.missing_cell_count == 0

    @property
    def columns_with_missing(self) -> tuple[ColumnMissingStats, ...]:
        """Per-column statistics for columns that contain missing values."""
        return tuple(column for column in self.columns if column.has_missing)

    def column(self, name: Hashable) -> ColumnMissingStats:
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


def missing_value_report(dataset: Dataset) -> MissingValueReport:
    """Compute missing-value statistics for a dataset.

    Parameters
    ----------
    dataset:
        Dataset to inspect. The dataset is not modified.

    Returns
    -------
    MissingValueReport
        Counts and fractions at dataset and column level. When the dataset has
        no cells, all fractions are ``0.0`` rather than undefined.

    Raises
    ------
    InvalidDatasetError
        If ``dataset`` is not a :class:`~data_mendrix.core.dataset.Dataset`.

    Examples
    --------
    >>> report = missing_value_report(dataset)
    >>> report.missing_cell_fraction
    0.125
    """
    if not isinstance(dataset, Dataset):
        raise InvalidDatasetError(
            f"missing_value_report() expects a Dataset, got {type(dataset).__name__}."
        )

    frame = dataset._view()
    row_count, column_count = frame.shape
    missing_mask = frame.isna()
    total_cell_count = row_count * column_count
    missing_cell_count = int(missing_mask.to_numpy().sum())

    columns: list[ColumnMissingStats] = []
    for position, label in enumerate(frame.columns):
        column_missing = int(missing_mask.iloc[:, position].sum())
        non_missing = row_count - column_missing
        columns.append(
            ColumnMissingStats(
                column=str(label),
                missing_count=column_missing,
                non_missing_count=non_missing,
                missing_fraction=(column_missing / row_count if row_count else 0.0),
            )
        )

    rows_with_missing = int(missing_mask.any(axis=1).sum())

    return MissingValueReport(
        row_count=row_count,
        column_count=column_count,
        total_cell_count=total_cell_count,
        missing_cell_count=missing_cell_count,
        missing_cell_fraction=(missing_cell_count / total_cell_count if total_cell_count else 0.0),
        rows_with_missing=rows_with_missing,
        complete_row_count=row_count - rows_with_missing,
        columns=tuple(columns),
    )
