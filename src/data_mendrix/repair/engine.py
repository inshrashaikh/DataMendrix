"""Repair planning, preview and application APIs."""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping

import pandas as pd

from data_mendrix.core.dataset import Dataset
from data_mendrix.core.exceptions import InvalidDatasetError, InvalidParameterError
from data_mendrix.repair.dtypes import apply_convert_types
from data_mendrix.repair.duplicates import apply_remove_duplicates
from data_mendrix.repair.missing import apply_drop_missing, apply_fill_missing
from data_mendrix.repair.models import RepairReport, RepairResult
from data_mendrix.repair.operations import (
    ConversionErrorPolicy,
    ConvertTypes,
    DropAxis,
    DropHow,
    DropMissing,
    FillMissing,
    FillStrategy,
    RemoveDuplicates,
    RepairKeepPolicy,
    RepairOperation,
)

_OPERATION_TYPES = (DropMissing, FillMissing, RemoveDuplicates, ConvertTypes)


def preview_repair(
    dataset: Dataset,
    operations: RepairOperation | Iterable[RepairOperation],
) -> RepairReport:
    """Preview a repair plan without returning or exposing a repaired dataset.

    The plan is executed on an internal defensive copy so later operations see
    the effects of earlier ones. The original dataset is never modified. Use
    :func:`apply_repair` to obtain the repaired :class:`Dataset` and the same
    report.

    Parameters
    ----------
    dataset:
        Dataset to repair.
    operations:
        One operation or an iterable of operations, applied in order.

    Returns
    -------
    RepairReport
        Immutable counts, affected row/column labels, per-column changes and
        explicit issues for the proposed plan.

    Raises
    ------
    InvalidDatasetError
        If ``dataset`` is not a :class:`~data_mendrix.core.dataset.Dataset`.
    InvalidParameterError
        If an operation object or operation parameter is invalid.
    ColumnNotFoundError
        If an operation selects a column that does not exist.
    RepairConversionError
        If dtype conversion fails under ``errors="raise"``.
    """
    if not isinstance(dataset, Dataset):
        raise InvalidDatasetError(
            f"preview_repair() expects a Dataset, got {type(dataset).__name__}."
        )
    resolved = _normalize_operations(operations)
    _, report = _run_operations(dataset, resolved)
    return report


def apply_repair(
    dataset: Dataset,
    operations: RepairOperation | Iterable[RepairOperation],
) -> RepairResult:
    """Apply a repair plan and return a new dataset with an immutable report.

    Parameters
    ----------
    dataset:
        Dataset to repair. It is never modified.
    operations:
        One operation or an iterable of operations, applied in order. An empty
        iterable produces an unchanged copy with an empty report.

    Returns
    -------
    RepairResult
        A new :class:`Dataset` preserving ``name`` and ``source`` metadata, plus
        a :class:`RepairReport` describing every operation.

    Raises
    ------
    InvalidDatasetError
        If ``dataset`` is not a :class:`~data_mendrix.core.dataset.Dataset`.
    InvalidParameterError
        If an operation object or operation parameter is invalid.
    ColumnNotFoundError
        If an operation selects a column that does not exist.
    RepairConversionError
        If dtype conversion fails under ``errors="raise"``.
    """
    if not isinstance(dataset, Dataset):
        raise InvalidDatasetError(
            f"apply_repair() expects a Dataset, got {type(dataset).__name__}."
        )
    resolved = _normalize_operations(operations)
    frame, report = _run_operations(dataset, resolved)
    repaired = Dataset(frame, name=dataset.name, source=dataset.source)
    return RepairResult(dataset=repaired, report=report)


def drop_missing(
    dataset: Dataset,
    *,
    axis: DropAxis = "rows",
    columns: Hashable | Iterable[Hashable] | None = None,
    how: DropHow = "any",
) -> RepairResult:
    """Drop rows or columns containing missing values.

    This is a convenience wrapper around
    :class:`~data_mendrix.repair.operations.DropMissing` and
    :func:`apply_repair`.
    """
    return apply_repair(
        dataset,
        DropMissing(axis=axis, columns=columns, how=how),
    )


def fill_missing(
    dataset: Dataset,
    *,
    columns: Hashable | Iterable[Hashable] | None = None,
    strategy: FillStrategy = "mean",
    value: object | None = None,
) -> RepairResult:
    """Fill missing values using a statistic or explicit constant.

    This is a convenience wrapper around
    :class:`~data_mendrix.repair.operations.FillMissing` and
    :func:`apply_repair`.
    """
    return apply_repair(
        dataset,
        FillMissing(columns=columns, strategy=strategy, value=value),
    )


def remove_duplicates(
    dataset: Dataset,
    *,
    subset: Hashable | Iterable[Hashable] | None = None,
    keep: RepairKeepPolicy = "first",
) -> RepairResult:
    """Remove duplicate rows using a full-row or subset comparison.

    This is a convenience wrapper around
    :class:`~data_mendrix.repair.operations.RemoveDuplicates` and
    :func:`apply_repair`.
    """
    return apply_repair(
        dataset,
        RemoveDuplicates(subset=subset, keep=keep),
    )


def convert_dtypes(
    dataset: Dataset,
    *,
    conversions: Mapping[Hashable, object],
    errors: ConversionErrorPolicy = "raise",
) -> RepairResult:
    """Convert selected columns to requested pandas-compatible dtypes.

    This is a convenience wrapper around
    :class:`~data_mendrix.repair.operations.ConvertTypes` and
    :func:`apply_repair`.
    """
    return apply_repair(
        dataset,
        ConvertTypes(conversions=conversions, errors=errors),
    )


def _normalize_operations(
    operations: RepairOperation | Iterable[RepairOperation],
) -> tuple[RepairOperation, ...]:
    """Normalise a single operation or iterable into a tuple of operations."""
    if isinstance(operations, _OPERATION_TYPES):
        return (operations,)
    if isinstance(operations, (str, bytes)):
        raise InvalidParameterError(
            "operations must be a repair operation or an iterable of repair operations."
        )
    try:
        resolved = tuple(operations)
    except TypeError as exc:
        raise InvalidParameterError(
            "operations must be a repair operation or an iterable of repair operations."
        ) from exc

    for operation in resolved:
        if not isinstance(operation, _OPERATION_TYPES):
            raise InvalidParameterError(
                "operations must contain only DropMissing, FillMissing, "
                "RemoveDuplicates or ConvertTypes instances."
            )
    return resolved


def _run_operations(
    dataset: Dataset,
    operations: tuple[RepairOperation, ...],
) -> tuple[pd.DataFrame, RepairReport]:
    """Run a validated plan on a defensive copy and build its report."""
    frame = dataset._view().copy(deep=True)
    row_count_before = int(frame.shape[0])
    column_count_before = int(frame.shape[1])
    reports = []

    for operation in operations:
        if isinstance(operation, DropMissing):
            frame, operation_report = apply_drop_missing(frame, operation)
        elif isinstance(operation, FillMissing):
            frame, operation_report = apply_fill_missing(frame, operation)
        elif isinstance(operation, RemoveDuplicates):
            frame, operation_report = apply_remove_duplicates(frame, operation)
        else:
            frame, operation_report = apply_convert_types(frame, operation)
        reports.append(operation_report)

    return frame, RepairReport(
        dataset_name=dataset.name,
        operations=tuple(reports),
        row_count_before=row_count_before,
        row_count_after=int(frame.shape[0]),
        column_count_before=column_count_before,
        column_count_after=int(frame.shape[1]),
    )
