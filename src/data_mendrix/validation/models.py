"""Value objects for dataset validation.

Validation is expressed as a report of immutable issues rather than raising on
the first problem. This lets an auditing workflow present every structural
problem at once, while :meth:`ValidationReport.raise_for_errors` offers an
explicit fail-fast option.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from data_mendrix.core.exceptions import (
    DatasetValidationError,
    InvalidParameterError,
)


class ValidationCode(StrEnum):
    """Stable machine-readable identifiers for validation issues."""

    TOO_FEW_ROWS = "too_few_rows"
    TOO_FEW_COLUMNS = "too_few_columns"
    DUPLICATE_COLUMNS = "duplicate_columns"
    BLANK_COLUMN_NAME = "blank_column_name"
    MISSING_REQUIRED_COLUMN = "missing_required_column"
    UNNAMED_COLUMN = "unnamed_column"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """A single structural problem found in a dataset.

    Attributes
    ----------
    code:
        Stable identifier that programs can branch on.
    message:
        Human-readable explanation suitable for logs or reports.
    column:
        The affected column label, when the issue concerns one column.
    """

    code: ValidationCode
    message: str
    column: str | None = None


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    """Rules used by :func:`data_mendrix.validation.validate_dataset`.

    Parameters
    ----------
    min_rows:
        Minimum number of rows required. Defaults to 1 so header-only files are
        rejected unless the caller explicitly opts out.
    min_columns:
        Minimum number of columns required.
    required_columns:
        Column labels that must be present. Duplicate or blank labels are
        rejected as configuration errors.
    forbid_unnamed_columns:
        Whether columns created from blank CSV headers (``"Unnamed: 0"`` and
        similar) should be reported as issues.

    Raises
    ------
    InvalidParameterError
        If any configuration value is invalid.
    """

    min_rows: int = 1
    min_columns: int = 1
    required_columns: tuple[str, ...] = ()
    forbid_unnamed_columns: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(self.min_rows, int)
            or isinstance(self.min_rows, bool)
            or self.min_rows < 0
        ):
            raise InvalidParameterError("min_rows must be a non-negative integer.")
        if (
            not isinstance(self.min_columns, int)
            or isinstance(self.min_columns, bool)
            or self.min_columns < 0
        ):
            raise InvalidParameterError("min_columns must be a non-negative integer.")
        if not isinstance(self.forbid_unnamed_columns, bool):
            raise InvalidParameterError("forbid_unnamed_columns must be a boolean.")
        if isinstance(self.required_columns, str):
            raise InvalidParameterError(
                "required_columns must be an iterable of column names, not a string."
            )

        try:
            required = tuple(self.required_columns)
        except TypeError as exc:
            raise InvalidParameterError("required_columns must be an iterable of strings.") from exc

        for column in required:
            if not isinstance(column, str) or not column.strip():
                raise InvalidParameterError("required_columns must contain non-empty strings.")
        if len(set(required)) != len(required):
            raise InvalidParameterError("required_columns must not contain duplicates.")

        object.__setattr__(self, "required_columns", required)


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """The complete set of structural issues found in a dataset."""

    issues: tuple[ValidationIssue, ...] = ()

    @property
    def is_valid(self) -> bool:
        """Whether no validation issues were found."""
        return not self.issues

    @property
    def codes(self) -> tuple[ValidationCode, ...]:
        """Issue codes in report order, useful for assertions and summaries."""
        return tuple(issue.code for issue in self.issues)

    def raise_for_errors(self) -> None:
        """Raise :class:`DatasetValidationError` if any issues were found.

        Raises
        ------
        DatasetValidationError
            If :attr:`issues` is non-empty. The exception carries the same
            issues that are available on this report.
        """
        if self.is_valid:
            return

        summary = "; ".join(issue.message for issue in self.issues)
        raise DatasetValidationError(
            f"Dataset validation failed with {len(self.issues)} issue(s): {summary}",
            self.issues,
        )
