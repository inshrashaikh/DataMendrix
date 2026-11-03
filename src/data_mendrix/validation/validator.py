"""Structural dataset validation.

The validator checks the shape and schema of a dataset. It deliberately does
not judge data quality such as missing values, outliers or duplicate rows;
those belong to the profiling and detection modules and are reported separately.
"""

from __future__ import annotations

import re

from data_mendrix.core.dataset import Dataset
from data_mendrix.core.exceptions import InvalidDatasetError, InvalidParameterError
from data_mendrix.validation.models import (
    ValidationCode,
    ValidationConfig,
    ValidationIssue,
    ValidationReport,
)

_UNNAMED_COLUMN_PATTERN = re.compile(r"^Unnamed: \d+$")


def validate_dataset(
    dataset: Dataset,
    config: ValidationConfig | None = None,
) -> ValidationReport:
    """Validate the structure of a dataset without modifying it.

    Parameters
    ----------
    dataset:
        Dataset to inspect.
    config:
        Optional validation rules. Defaults to
        :class:`~data_mendrix.validation.models.ValidationConfig`, which requires
        at least one row and one column.

    Returns
    -------
    ValidationReport
        Every issue found, in a deterministic order. An empty report means the
        dataset passed.

    Raises
    ------
    InvalidDatasetError
        If ``dataset`` is not a :class:`~data_mendrix.core.dataset.Dataset`.
    InvalidParameterError
        If ``config`` is not a :class:`ValidationConfig`.

    Examples
    --------
    >>> report = validate_dataset(dataset)
    >>> if not report.is_valid:
    ...     print(report.codes)
    """
    if not isinstance(dataset, Dataset):
        raise InvalidDatasetError(
            f"validate_dataset() expects a Dataset, got {type(dataset).__name__}."
        )
    if config is None:
        config = ValidationConfig()
    elif not isinstance(config, ValidationConfig):
        raise InvalidParameterError(
            f"config must be a ValidationConfig instance or None, got {type(config).__name__}."
        )

    frame = dataset._view()
    row_count, column_count = frame.shape
    issues: list[ValidationIssue] = []

    if row_count < config.min_rows:
        issues.append(
            ValidationIssue(
                code=ValidationCode.TOO_FEW_ROWS,
                message=(
                    f"Dataset has {row_count} row(s), but at least {config.min_rows} is required."
                ),
            )
        )

    if column_count < config.min_columns:
        issues.append(
            ValidationIssue(
                code=ValidationCode.TOO_FEW_COLUMNS,
                message=(
                    f"Dataset has {column_count} column(s), but at least "
                    f"{config.min_columns} is required."
                ),
            )
        )

    duplicate_labels = frame.columns[frame.columns.duplicated()].unique()
    for label in duplicate_labels:
        issues.append(
            ValidationIssue(
                code=ValidationCode.DUPLICATE_COLUMNS,
                message=f"Column label {label!r} appears more than once.",
                column=str(label),
            )
        )

    for label in frame.columns:
        if isinstance(label, str) and not label.strip():
            issues.append(
                ValidationIssue(
                    code=ValidationCode.BLANK_COLUMN_NAME,
                    message="Dataset contains a blank column name.",
                    column=label,
                )
            )
        elif (
            config.forbid_unnamed_columns
            and isinstance(label, str)
            and _UNNAMED_COLUMN_PATTERN.fullmatch(label)
        ):
            issues.append(
                ValidationIssue(
                    code=ValidationCode.UNNAMED_COLUMN,
                    message=(f"Column {label!r} was generated from a blank CSV header."),
                    column=label,
                )
            )

    for required in config.required_columns:
        if required not in frame.columns:
            issues.append(
                ValidationIssue(
                    code=ValidationCode.MISSING_REQUIRED_COLUMN,
                    message=f"Required column {required!r} is missing.",
                    column=required,
                )
            )

    return ValidationReport(tuple(issues))
