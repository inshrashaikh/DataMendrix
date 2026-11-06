"""Data-type repair with explicit conversion error policies."""

from __future__ import annotations

from collections.abc import Hashable
from typing import Any, cast

import numpy as np
import pandas as pd

from data_mendrix.core.columns import unique_column_position
from data_mendrix.core.exceptions import (
    InvalidParameterError,
    RepairConversionError,
)
from data_mendrix.repair._errors import builtin_error_type
from data_mendrix.repair.models import (
    ColumnChange,
    RepairAction,
    RepairIssue,
    RepairOperationReport,
)
from data_mendrix.repair.operations import ConversionErrorPolicy, ConvertTypes

_CONVERSION_EXCEPTIONS = (Exception,)


def apply_convert_types(
    frame: pd.DataFrame,
    operation: ConvertTypes,
) -> tuple[pd.DataFrame, RepairOperationReport]:
    """Apply dtype conversions and return the new frame and report."""
    row_count_before, column_count_before = (int(value) for value in frame.shape)
    resolved: list[tuple[Hashable, int, object]] = []

    for label, target in operation.conversions.items():
        position = unique_column_position(frame, label)
        _validate_target_dtype(label, target)
        resolved.append((label, position, target))

    row_mask = np.zeros(frame.shape[0], dtype=bool)
    column_changes: list[ColumnChange] = []
    affected_columns: list[str] = []
    issues: list[RepairIssue] = []

    for label, position, target in resolved:
        series = frame.iloc[:, position]
        column = str(frame.columns[position])
        before_dtype = str(series.dtype)

        try:
            converted, coerced_positions = _convert_series(series, target, operation.errors)
        except _CONVERSION_EXCEPTIONS as exc:
            if operation.errors == "skip":
                issues.append(
                    RepairIssue(
                        column=column,
                        message=f"conversion to {target!r} failed; column left unchanged: {exc}",
                        error_type=builtin_error_type(exc),
                    )
                )
                continue
            if isinstance(exc, RepairConversionError):
                raise
            raise RepairConversionError(
                f"Could not convert column {column!r} to {target!r}: {exc}",
                column=label,
                target=target,
            ) from exc

        after_dtype = str(converted.dtype)
        if before_dtype == after_dtype and converted.equals(series):
            continue

        for coerced_position in coerced_positions:
            row_mask[coerced_position] = True

        frame.isetitem(position, converted.array)
        affected_columns.append(column)
        detail = (
            f"{len(coerced_positions)} invalid value(s) coerced to missing"
            if coerced_positions
            else "dtype converted"
        )
        column_changes.append(
            ColumnChange(
                column=column,
                change_count=int(converted.size),
                before_dtype=before_dtype,
                after_dtype=after_dtype,
                detail=detail,
            )
        )

    return frame, RepairOperationReport(
        action=RepairAction.CONVERT_DTYPES,
        row_count_before=row_count_before,
        row_count_after=int(frame.shape[0]),
        column_count_before=column_count_before,
        column_count_after=int(frame.shape[1]),
        affected_rows=tuple(frame.index[row_mask].tolist()),
        affected_columns=tuple(affected_columns),
        change_count=len(affected_columns),
        column_changes=tuple(column_changes),
        issues=tuple(issues),
        detail=f"errors='{operation.errors}'",
    )


def _validate_target_dtype(column: Hashable, target: object) -> None:
    """Validate a requested dtype before any data is changed."""
    if target is None:
        raise InvalidParameterError(f"Target dtype for column {column!r} must not be None.")
    try:
        pd.api.types.pandas_dtype(target)
    except _CONVERSION_EXCEPTIONS as exc:
        raise InvalidParameterError(
            f"Target dtype for column {column!r} is invalid: {target!r}."
        ) from exc


def _convert_series(
    series: pd.Series,
    target: object,
    errors: ConversionErrorPolicy,
) -> tuple[pd.Series, tuple[int, ...]]:
    """Convert one series, returning the result and coerced row positions."""
    try:
        converted = _astype(series, target)
    except _CONVERSION_EXCEPTIONS:
        if errors != "coerce":
            raise
    else:
        _ensure_missing_preserved(
            series.isna().to_numpy(dtype=bool),
            converted,
            target,
            column=str(series.name),
        )
        return converted, ()

    dtype = pd.api.types.pandas_dtype(target)
    original_missing = series.isna().to_numpy(dtype=bool)

    if (
        pd.api.types.is_integer_dtype(dtype)
        or pd.api.types.is_float_dtype(dtype)
        or pd.api.types.is_complex_dtype(dtype)
    ):
        try:
            coerced = pd.to_numeric(series, errors="coerce")
        except _CONVERSION_EXCEPTIONS as exc:
            raise _conversion_error(series, target, str(exc)) from exc
        return _cast_coerced(
            coerced,
            target,
            original_missing,
            column=str(series.name),
        )

    if pd.api.types.is_datetime64_any_dtype(dtype):
        try:
            coerced = pd.to_datetime(series, errors="coerce")
        except _CONVERSION_EXCEPTIONS as exc:
            raise _conversion_error(series, target, str(exc)) from exc
        target_tz = getattr(dtype, "tz", None)
        if target_tz is not None:
            coerced = (
                coerced.dt.tz_localize(target_tz)
                if coerced.dt.tz is None
                else coerced.dt.tz_convert(target_tz)
            )
        return _cast_coerced(
            coerced,
            target,
            original_missing,
            column=str(series.name),
        )

    if pd.api.types.is_timedelta64_dtype(dtype):
        try:
            coerced = pd.to_timedelta(series, errors="coerce")
        except _CONVERSION_EXCEPTIONS as exc:
            raise _conversion_error(series, target, str(exc)) from exc
        return _cast_coerced(
            coerced,
            target,
            original_missing,
            column=str(series.name),
        )

    return _coerce_generic(series, target, original_missing)


def _cast_coerced(
    coerced: pd.Series,
    target: object,
    original_missing: np.ndarray,
    *,
    column: str,
) -> tuple[pd.Series, tuple[int, ...]]:
    """Cast a pandas-coerced series and count newly missing values."""
    try:
        converted = _astype(coerced, target)
    except _CONVERSION_EXCEPTIONS as exc:
        raise _conversion_error(
            coerced,
            target,
            "invalid values were replaced with missing, but the target dtype cannot "
            f"represent missing values: {exc}",
            column=column,
        ) from exc

    converted_missing = converted.isna().to_numpy(dtype=bool)
    _ensure_missing_preserved(
        original_missing,
        converted,
        target,
        column=column,
    )
    failed_positions = np.flatnonzero(converted_missing & ~original_missing)
    return converted, tuple(int(position) for position in failed_positions)


def _coerce_generic(
    series: pd.Series,
    target: object,
    original_missing: np.ndarray,
) -> tuple[pd.Series, tuple[int, ...]]:
    """Fallback coercion for dtypes without a vectorised pandas converter."""
    values: list[object] = []
    failed_positions: list[int] = []

    for position, value in enumerate(series.tolist()):
        if original_missing[position]:
            values.append(np.nan)
            continue
        try:
            values.append(_astype(pd.Series([value]), target).iloc[0])
        except _CONVERSION_EXCEPTIONS:
            values.append(np.nan)
            failed_positions.append(position)

    candidate = pd.Series(
        values,
        index=series.index,
        name=series.name,
        dtype=object,
    )
    try:
        converted = _astype(candidate, target)
    except _CONVERSION_EXCEPTIONS as exc:
        raise _conversion_error(series, target, str(exc)) from exc

    _ensure_missing_preserved(
        original_missing,
        converted,
        target,
        column=str(series.name),
    )
    if failed_positions:
        converted_missing = converted.isna().to_numpy(dtype=bool)
        if not all(converted_missing[position] for position in failed_positions):
            raise _conversion_error(
                series,
                target,
                "invalid values could not be represented as missing values by the target dtype",
            )
    return converted, tuple(failed_positions)


def _ensure_missing_preserved(
    original_missing: np.ndarray,
    converted: pd.Series,
    target: object,
    *,
    column: str,
) -> None:
    """Reject conversions that would turn existing missing values into data."""
    if not original_missing.any():
        return
    converted_missing = converted.isna().to_numpy(dtype=bool)
    if not bool(converted_missing[original_missing].all()):
        raise RepairConversionError(
            f"Could not convert column {column!r} to {target!r}: the target dtype "
            "cannot represent existing missing values.",
            column=column,
            target=target,
        )


def _astype(series: pd.Series, target: object) -> pd.Series:
    """Cast a series while accepting the full pandas dtype surface."""
    return series.astype(cast(Any, target))


def _conversion_error(
    series: pd.Series,
    target: object,
    detail: str,
    *,
    column: str | None = None,
) -> RepairConversionError:
    """Build a typed conversion error for an internal coercion failure."""
    label = str(series.name) if column is None else column
    return RepairConversionError(
        f"Could not convert column {label!r} to {target!r}: {detail}",
        column=label,
        target=target,
    )
