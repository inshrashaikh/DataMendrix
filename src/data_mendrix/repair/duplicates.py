"""Duplicate-row removal."""

from __future__ import annotations

import pandas as pd

from data_mendrix.core.columns import resolve_column_selection, unique_column_position
from data_mendrix.repair.models import RepairAction, RepairOperationReport
from data_mendrix.repair.operations import RemoveDuplicates


def apply_remove_duplicates(
    frame: pd.DataFrame,
    operation: RemoveDuplicates,
) -> tuple[pd.DataFrame, RepairOperationReport]:
    """Remove duplicate rows and return the new frame and report."""
    row_count_before, column_count_before = (int(value) for value in frame.shape)
    selection = resolve_column_selection(frame, operation.subset, parameter="subset")
    subset_positions = (
        None
        if selection is None
        else list(
            dict.fromkeys(
                unique_column_position(frame, label, parameter="subset") for label in selection
            )
        )
    )

    duplicate_mask = frame.duplicated(subset=selection, keep=operation.keep)
    duplicate_count = int(duplicate_mask.sum())
    result = frame.loc[~duplicate_mask].copy()
    affected_columns: tuple[str, ...] = ()
    if duplicate_count:
        affected_columns = (
            tuple(str(column) for column in frame.columns)
            if subset_positions is None
            else tuple(str(frame.columns[position]) for position in subset_positions)
        )
    scope = (
        "all columns"
        if selection is None
        else "[" + ", ".join(repr(label) for label in selection) + "]"
    )

    return result, RepairOperationReport(
        action=RepairAction.REMOVE_DUPLICATES,
        row_count_before=row_count_before,
        row_count_after=int(result.shape[0]),
        column_count_before=column_count_before,
        column_count_after=int(result.shape[1]),
        affected_rows=tuple(frame.index[duplicate_mask].tolist()),
        affected_columns=affected_columns,
        change_count=duplicate_count,
        detail=f"keep={operation.keep!r}, subset={scope}",
    )
