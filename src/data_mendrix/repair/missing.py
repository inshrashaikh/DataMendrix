"""Missing-value repair operations."""

from __future__ import annotations

import math
import warnings
from collections.abc import Hashable
from typing import Any, cast

import numpy as np
import pandas as pd

from data_mendrix.core.columns import resolve_column_selection, unique_column_position
from data_mendrix.core.exceptions import InvalidParameterError
from data_mendrix.repair._errors import builtin_error_type
from data_mendrix.repair.models import (
    ColumnChange,
    RepairAction,
    RepairIssue,
    RepairOperationReport,
)
from data_mendrix.repair.operations import DropMissing, FillMissing

_NO_VALUE = object()


def apply_drop_missing(
    frame: pd.DataFrame,
    operation: DropMissing,
) -> tuple[pd.DataFrame, RepairOperationReport]:
    """Apply a missing-value drop and return the new frame and report."""
    row_count_before, column_count_before = (int(value) for value in frame.shape)
    selection = resolve_column_selection(frame, operation.columns)
    positions = (
        list(range(frame.shape[1]))
        if selection is None
        else list(dict.fromkeys(unique_column_position(frame, label) for label in selection))
    )
    scope = _scope_detail(selection)

    if operation.axis == "rows":
        if positions:
            subset = frame.iloc[:, positions]
            missing_mask = (
                subset.isna().all(axis=1) if operation.how == "all" else subset.isna().any(axis=1)
            )
            dropped_row_positions = [
                int(position) for position in np.flatnonzero(missing_mask.to_numpy(dtype=bool))
            ]
            result = frame.iloc[~missing_mask.to_numpy(dtype=bool)].copy()
        elif operation.how == "all":
            # With no columns to inspect, the "all considered values are missing"
            # condition is vacuously true for every row.
            dropped_row_positions = list(range(frame.shape[0]))
            result = frame.iloc[0:0].copy()
        else:
            dropped_row_positions = []
            result = frame.copy()

        affected_columns: tuple[str, ...] = ()
        if dropped_row_positions:
            dropped_subset = frame.iloc[dropped_row_positions, positions]
            column_flags = dropped_subset.isna().any(axis=0).to_numpy(dtype=bool)
            affected_columns = tuple(
                str(frame.columns[position])
                for position, flagged in zip(positions, column_flags, strict=True)
                if flagged
            )

        return result, _build_report(
            action=RepairAction.DROP_MISSING_ROWS,
            frame_before_rows=row_count_before,
            frame_before_columns=column_count_before,
            result=result,
            affected_rows=tuple(frame.index[dropped_row_positions].tolist()),
            affected_columns=affected_columns,
            change_count=len(dropped_row_positions),
            detail=f"axis='rows', how='{operation.how}', scope={scope}",
        )

    if positions:
        subset = frame.iloc[:, positions]
        column_mask = (
            subset.isna().all(axis=0) if operation.how == "all" else subset.isna().any(axis=0)
        )
        dropped_column_positions = [
            position
            for position, dropped in zip(positions, column_mask.to_numpy(dtype=bool), strict=True)
            if dropped
        ]
    else:
        dropped_column_positions = []

    dropped_set = set(dropped_column_positions)
    keep_positions = [position for position in range(frame.shape[1]) if position not in dropped_set]
    result = frame.iloc[:, keep_positions].copy()
    affected_columns = tuple(str(frame.columns[position]) for position in dropped_column_positions)

    return result, _build_report(
        action=RepairAction.DROP_MISSING_COLUMNS,
        frame_before_rows=row_count_before,
        frame_before_columns=column_count_before,
        result=result,
        affected_columns=affected_columns,
        change_count=len(dropped_column_positions),
        detail=f"axis='columns', how='{operation.how}', scope={scope}",
    )


def apply_fill_missing(
    frame: pd.DataFrame,
    operation: FillMissing,
) -> tuple[pd.DataFrame, RepairOperationReport]:
    """Apply a missing-value fill and return the new frame and report."""
    row_count_before, column_count_before = (int(value) for value in frame.shape)
    selection = resolve_column_selection(frame, operation.columns)
    positions = (
        list(range(frame.shape[1]))
        if selection is None
        else list(dict.fromkeys(unique_column_position(frame, label) for label in selection))
    )

    if selection is not None and operation.strategy in ("mean", "median"):
        non_numeric = [
            str(frame.columns[position])
            for position in positions
            if not _is_numeric_candidate(frame.iloc[:, position])
        ]
        if non_numeric:
            formatted = ", ".join(repr(column) for column in non_numeric)
            raise InvalidParameterError(
                f"{operation.strategy} fill requires numeric columns; not numeric: {formatted}."
            )

    row_mask = np.zeros(frame.shape[0], dtype=bool)
    column_changes: list[ColumnChange] = []
    issues: list[RepairIssue] = []
    total_filled = 0

    for position in positions:
        series = frame.iloc[:, position]
        column = str(frame.columns[position])
        if operation.strategy in ("mean", "median") and not _is_numeric_candidate(series):
            continue

        missing_count = int(series.isna().sum())
        if missing_count == 0:
            continue

        fill_value = _resolve_fill_value(series, operation, column=column, issues=issues)
        if fill_value is _NO_VALUE:
            continue

        before_dtype = str(series.dtype)
        try:
            # ``infer_objects`` keeps object columns' dtype stable across
            # pandas versions. The downcasting happens inside ``fillna`` in
            # pandas 2 and is explicit via ``infer_objects``; suppress the
            # deprecation warning because the explicit inference is the
            # sanctioned migration path.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                filled = series.fillna(cast(Any, fill_value)).infer_objects()
        except Exception as exc:
            issues.append(
                RepairIssue(
                    column=column,
                    message=(
                        f"fill value {fill_value!r} is not compatible with the column; "
                        f"column left unchanged: {exc}"
                    ),
                    error_type=builtin_error_type(exc),
                )
            )
            continue

        filled_count = missing_count - int(filled.isna().sum())
        if filled_count <= 0:
            issues.append(
                RepairIssue(
                    column=column,
                    message="fill value did not replace any missing values.",
                )
            )
            continue

        frame.isetitem(position, filled.array)
        row_mask |= series.isna().to_numpy(dtype=bool)
        total_filled += filled_count
        column_changes.append(
            ColumnChange(
                column=column,
                change_count=filled_count,
                before_dtype=before_dtype,
                after_dtype=str(frame.iloc[:, position].dtype),
                fill_value=fill_value,
                detail=f"strategy='{operation.strategy}'",
            )
        )

    result = frame
    return result, _build_report(
        action=RepairAction.FILL_MISSING,
        frame_before_rows=row_count_before,
        frame_before_columns=column_count_before,
        result=result,
        affected_rows=tuple(frame.index[row_mask].tolist()),
        affected_columns=tuple(change.column for change in column_changes),
        change_count=total_filled,
        column_changes=tuple(column_changes),
        issues=tuple(issues),
        detail=f"strategy='{operation.strategy}', scope={_scope_detail(selection)}",
    )


def _resolve_fill_value(
    series: pd.Series,
    operation: FillMissing,
    *,
    column: str,
    issues: list[RepairIssue],
) -> object:
    """Return a fill value or the internal no-value sentinel."""
    if operation.strategy == "constant":
        return operation.value

    if operation.strategy in ("mean", "median"):
        values = series.to_numpy(dtype="float64", na_value=np.nan)
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            issues.append(
                RepairIssue(
                    column=column,
                    message=(
                        f"{operation.strategy} is undefined because the column has no finite "
                        "non-missing values."
                    ),
                )
            )
            return _NO_VALUE
        fill_value = float(
            np.mean(finite_values) if operation.strategy == "mean" else np.median(finite_values)
        )
        if not math.isfinite(fill_value):
            issues.append(
                RepairIssue(
                    column=column,
                    message=(
                        f"{operation.strategy} produced a non-finite fill value; column left "
                        "unchanged."
                    ),
                )
            )
            return _NO_VALUE
        return fill_value

    mode = _mode_value(series)
    if mode is None:
        issues.append(
            RepairIssue(
                column=column,
                message="mode is undefined because the column has no non-missing values.",
            )
        )
        return _NO_VALUE
    value, _ = mode
    return value


def _mode_value(series: pd.Series) -> tuple[object, int] | None:
    """Return the most frequent non-missing value and its count."""
    non_missing = series.dropna()
    if non_missing.empty:
        return None

    try:
        counts = non_missing.value_counts(sort=False)
    except Exception:
        values = non_missing.tolist()
        best_value = values[0]
        best_count = 0
        for value in values:
            count = sum(1 for other in values if _values_equal(value, other))
            if count > best_count:
                best_value = value
                best_count = count
        return best_value, best_count

    if counts.empty:
        return None
    return counts.idxmax(), int(counts.max())


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


def _is_numeric_candidate(series: pd.Series) -> bool:
    """Return whether a series is eligible for mean or median filling."""
    dtype = series.dtype
    return bool(
        pd.api.types.is_numeric_dtype(dtype)
        and not pd.api.types.is_bool_dtype(dtype)
        and not pd.api.types.is_complex_dtype(dtype)
    )


def _scope_detail(selection: list[Hashable] | None) -> str:
    """Format a validated column selection for a human-readable report."""
    if selection is None:
        return "all columns"
    return "[" + ", ".join(repr(label) for label in selection) + "]"


def _build_report(
    *,
    action: RepairAction,
    frame_before_rows: int,
    frame_before_columns: int,
    result: pd.DataFrame,
    affected_rows: tuple[Hashable, ...] = (),
    affected_columns: tuple[str, ...] = (),
    change_count: int = 0,
    column_changes: tuple[ColumnChange, ...] = (),
    issues: tuple[RepairIssue, ...] = (),
    detail: str = "",
) -> RepairOperationReport:
    """Build a repair operation report from before/after frame dimensions."""
    return RepairOperationReport(
        action=action,
        row_count_before=frame_before_rows,
        row_count_after=int(result.shape[0]),
        column_count_before=frame_before_columns,
        column_count_after=int(result.shape[1]),
        affected_rows=affected_rows,
        affected_columns=affected_columns,
        change_count=change_count,
        column_changes=column_changes,
        issues=issues,
        detail=detail,
    )
