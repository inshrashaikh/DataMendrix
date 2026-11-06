"""Immutable result and report models for dataset repair."""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, cast

import numpy as np
import pandas as pd

from data_mendrix.core.dataset import Dataset


class RepairAction(StrEnum):
    """Stable machine-readable identifiers for repair operations."""

    DROP_MISSING_ROWS = "drop_missing_rows"
    DROP_MISSING_COLUMNS = "drop_missing_columns"
    FILL_MISSING = "fill_missing"
    REMOVE_DUPLICATES = "remove_duplicates"
    CONVERT_DTYPES = "convert_dtypes"


@dataclass(frozen=True, slots=True)
class RepairIssue:
    """A non-fatal problem encountered while planning or applying a repair.

    ``error_type`` is a stable, backend-independent class name: DataMendrix
    errors keep their own name (for example ``"RepairConversionError"``) and
    third-party exceptions are reported as the nearest built-in exception in
    their inheritance chain (for example ``"ValueError"``).
    """

    column: str | None
    message: str
    error_type: str | None = None


@dataclass(frozen=True, slots=True)
class ColumnChange:
    """Cell- or dtype-level changes for one affected column."""

    column: str
    change_count: int
    before_dtype: str
    after_dtype: str
    fill_value: object | None = None
    detail: str = ""


@dataclass(frozen=True, slots=True)
class RepairOperationReport:
    """Immutable report for one repair operation.

    ``change_count`` is the operation's primary change unit: rows removed for
    row/duplicate operations, columns removed for column drops, cells filled for
    missing-value fills, and columns converted for dtype repairs.
    """

    action: RepairAction
    row_count_before: int
    row_count_after: int
    column_count_before: int
    column_count_after: int
    affected_rows: tuple[Hashable, ...] = ()
    affected_columns: tuple[str, ...] = ()
    change_count: int = 0
    column_changes: tuple[ColumnChange, ...] = ()
    issues: tuple[RepairIssue, ...] = ()
    detail: str = ""

    @property
    def changed(self) -> bool:
        """Whether the operation changed data or schema."""
        return self.change_count > 0

    @property
    def affected_row_count(self) -> int:
        """Number of row labels explicitly affected."""
        return len(self.affected_rows)

    @property
    def affected_column_count(self) -> int:
        """Number of column labels affected or used for a row-level decision."""
        return len(self.affected_columns)

    @property
    def row_count_delta(self) -> int:
        """Change in row count; negative values mean rows were removed."""
        return self.row_count_after - self.row_count_before

    @property
    def column_count_delta(self) -> int:
        """Change in column count; negative values mean columns were removed."""
        return self.column_count_after - self.column_count_before


@dataclass(frozen=True, slots=True)
class RepairReport:
    """Immutable combined report for a repair plan or preview."""

    dataset_name: str
    operations: tuple[RepairOperationReport, ...]
    row_count_before: int
    row_count_after: int
    column_count_before: int
    column_count_after: int

    @property
    def operation_count(self) -> int:
        """Number of operations in the plan."""
        return len(self.operations)

    @property
    def changed(self) -> bool:
        """Whether any operation changed data or schema."""
        return any(operation.changed for operation in self.operations)

    @property
    def change_count(self) -> int:
        """Sum of each operation's primary change count."""
        return sum(operation.change_count for operation in self.operations)

    @property
    def affected_rows(self) -> tuple[Hashable, ...]:
        """Union of row labels affected across operations, in plan order."""
        rows = (row for operation in self.operations for row in operation.affected_rows)
        return _unique_values(rows)

    @property
    def affected_columns(self) -> tuple[str, ...]:
        """Union of column names affected across operations, in plan order."""
        columns: list[str] = []
        seen: set[str] = set()
        for operation in self.operations:
            for column in operation.affected_columns:
                if column not in seen:
                    seen.add(column)
                    columns.append(column)
        return tuple(columns)

    @property
    def issues(self) -> tuple[RepairIssue, ...]:
        """All non-fatal issues recorded by the plan."""
        return tuple(issue for operation in self.operations for issue in operation.issues)

    @property
    def has_issues(self) -> bool:
        """Whether at least one non-fatal issue was recorded."""
        return bool(self.issues)

    @property
    def row_count_delta(self) -> int:
        """Change in row count across the whole plan."""
        return self.row_count_after - self.row_count_before

    @property
    def column_count_delta(self) -> int:
        """Change in column count across the whole plan."""
        return self.column_count_after - self.column_count_before


@dataclass(frozen=True, slots=True)
class RepairResult:
    """A repaired dataset together with its immutable report."""

    dataset: Dataset
    report: RepairReport


def _unique_values(values: Iterable[Hashable]) -> tuple[Hashable, ...]:
    """Return labels in first-seen order without requiring hashability."""
    unique: list[Hashable] = []
    seen: set[Hashable] = set()
    unhashable_unique: list[Hashable] = []
    seen_missing = False

    for value in values:
        if _is_missing_scalar(value):
            if seen_missing:
                continue
            seen_missing = True
            unique.append(value)
            continue
        try:
            hash(value)
        except TypeError:
            if any(_values_equal(value, existing) for existing in unhashable_unique):
                continue
            unhashable_unique.append(value)
            unique.append(value)
            continue
        if value in seen:
            continue
        seen.add(value)
        unique.append(value)

    return tuple(unique)


def _is_missing_scalar(value: object) -> bool:
    """Return whether a value is a scalar missing-value sentinel."""
    try:
        result = pd.isna(cast(Any, value))
    except Exception:
        return False
    return isinstance(result, (bool, np.bool_)) and bool(result)


def _values_equal(left: object, right: object) -> bool:
    """Compare two possibly array-like values without ambiguity errors."""
    try:
        result = left == right
    except Exception:
        return False
    if isinstance(result, bool):
        return result
    try:
        return bool(np.all(result))
    except Exception:
        return False
