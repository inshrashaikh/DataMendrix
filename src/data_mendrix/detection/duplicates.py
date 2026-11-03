"""Duplicate-row detection.

Duplicate detection is read-only and uses pandas' definition of equality:
two missing values compare as equal, so rows with identical ``NaN`` patterns can
be duplicates. Reports identify redundant rows relative to the selected
``keep`` policy without removing anything.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from typing import Literal

from data_mendrix.core.dataset import Dataset
from data_mendrix.core.exceptions import InvalidDatasetError, InvalidParameterError
from data_mendrix.detection._columns import resolve_column_selection

KeepPolicy = Literal["first", "last"]


@dataclass(frozen=True, slots=True)
class DuplicateReport:
    """Summary of duplicate rows in a dataset.

    Attributes
    ----------
    row_count:
        Total number of rows inspected.
    unique_row_count:
        Number of distinct row values.
    duplicate_row_count:
        Number of redundant rows, i.e. rows beyond the first (or last)
        occurrence of each distinct row value.
    duplicate_group_count:
        Number of distinct row values that occur more than once.
    duplicate_indices:
        Index labels of the redundant rows, in dataset order.
    """

    row_count: int
    unique_row_count: int
    duplicate_row_count: int
    duplicate_group_count: int
    duplicate_indices: tuple[Hashable, ...]

    @property
    def has_duplicates(self) -> bool:
        """Whether at least one redundant row exists."""
        return self.duplicate_row_count > 0

    @property
    def duplicate_fraction(self) -> float:
        """Fraction of rows that are redundant, between 0 and 1.

        An empty dataset is defined to have a duplicate fraction of ``0.0``.
        """
        if self.row_count == 0:
            return 0.0
        return self.duplicate_row_count / self.row_count


def detect_duplicates(
    dataset: Dataset,
    *,
    subset: Hashable | Iterable[Hashable] | None = None,
    keep: KeepPolicy = "first",
) -> DuplicateReport:
    """Find duplicate rows in a dataset without removing them.

    Parameters
    ----------
    dataset:
        Dataset to inspect.
    subset:
        Optional column label or iterable of labels to compare. By default all
        columns are compared, which detects fully duplicated rows.
    keep:
        Which occurrence to treat as original. ``"first"`` marks later
        occurrences as duplicates; ``"last"`` marks earlier occurrences.

    Returns
    -------
    DuplicateReport
        Counts and index labels describing the duplicates.

    Raises
    ------
    InvalidDatasetError
        If ``dataset`` is not a :class:`~data_mendrix.core.dataset.Dataset`.
    InvalidParameterError
        If ``subset`` is empty or ``keep`` is not ``"first"`` or ``"last"``.
    ColumnNotFoundError
        If a requested subset column does not exist.

    Examples
    --------
    >>> report = detect_duplicates(dataset)
    >>> report.duplicate_row_count
    3
    """
    if not isinstance(dataset, Dataset):
        raise InvalidDatasetError(
            f"detect_duplicates() expects a Dataset, got {type(dataset).__name__}."
        )
    if keep not in ("first", "last"):
        raise InvalidParameterError("keep must be either 'first' or 'last'.")

    frame = dataset._view()
    columns = resolve_column_selection(frame, subset, parameter="subset")
    duplicate_mask = frame.duplicated(subset=columns, keep=keep)
    duplicate_row_count = int(duplicate_mask.sum())

    if duplicate_row_count == 0:
        duplicate_group_count = 0
    else:
        all_duplicate_mask = frame.duplicated(subset=columns, keep=False)
        compared = (
            frame.loc[all_duplicate_mask]
            if columns is None
            else frame.loc[all_duplicate_mask, columns]
        )
        duplicate_group_count = int(compared.drop_duplicates().shape[0])

    return DuplicateReport(
        row_count=int(frame.shape[0]),
        unique_row_count=int(frame.shape[0]) - duplicate_row_count,
        duplicate_row_count=duplicate_row_count,
        duplicate_group_count=duplicate_group_count,
        duplicate_indices=tuple(frame.index[duplicate_mask].tolist()),
    )
