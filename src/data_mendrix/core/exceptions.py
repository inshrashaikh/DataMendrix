"""Exception hierarchy used across DataMendrix.

Every library-specific error derives from :class:`DataMendrixError`, which lets
callers catch a single base class without swallowing unrelated Python errors.
Where it makes sense, exceptions also inherit from an appropriate built-in
exception (``ValueError`` or ``KeyError``) so existing error-handling idioms keep
working.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data_mendrix.validation.models import ValidationIssue


class DataMendrixError(Exception):
    """Base class for all DataMendrix-specific exceptions."""


class DatasetLoadError(DataMendrixError):
    """Raised when a dataset cannot be read from its source.

    This covers missing or inaccessible files, undecodable content, empty input
    and malformed CSV syntax. The original low-level exception, when available,
    is chained as the cause.
    """


class UnsupportedFileTypeError(DatasetLoadError):
    """Raised when a path does not point to a supported file type.

    The initial loader intentionally supports only ``.csv`` files.
    """


class InvalidDatasetError(DataMendrixError, ValueError):
    """Raised when an object cannot be used as a dataset.

    Typical causes are passing something other than a :class:`pandas.DataFrame`
    or :class:`~data_mendrix.core.dataset.Dataset` where one is required.
    """


class DatasetValidationError(DataMendrixError):
    """Raised when a dataset fails structural validation.

    The individual :class:`~data_mendrix.validation.models.ValidationIssue`
    objects that caused the failure are available through :attr:`issues`.
    """

    def __init__(
        self,
        message: str,
        issues: Sequence[ValidationIssue] = (),
    ) -> None:
        super().__init__(message)
        self.issues: tuple[ValidationIssue, ...] = tuple(issues)


class ColumnNotFoundError(DataMendrixError, KeyError):
    """Raised when one or more requested columns do not exist in a dataset.

    The missing labels are available through :attr:`columns`.
    """

    def __init__(self, columns: Iterable[Hashable]) -> None:
        self.columns: tuple[str, ...] = tuple(str(column) for column in columns)
        formatted = ", ".join(repr(column) for column in self.columns)
        super().__init__(f"Column(s) not found in dataset: {formatted}")


class InvalidParameterError(DataMendrixError, ValueError):
    """Raised when a public function or option receives an invalid parameter."""


class RepairError(DataMendrixError):
    """Base class for repair operations that cannot be completed."""


class RepairConversionError(RepairError, ValueError):
    """Raised when a requested dtype conversion fails.

    The offending column label and target dtype are available through
    :attr:`column` and :attr:`target`. The original pandas exception, if any, is
    chained as the cause.
    """

    def __init__(self, message: str, *, column: Hashable, target: object) -> None:
        super().__init__(message)
        self.column: str = str(column)
        self.target: str = str(target)
