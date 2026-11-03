"""Immutable dataset container.

DataMendrix never analyses or repairs a caller's ``DataFrame`` in place. The
:class:`Dataset` container takes a defensive copy on construction and exposes
the frame only as another copy. Internal modules use :meth:`Dataset._view`,
which is read-only by contract, to avoid copying large frames repeatedly.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from data_mendrix.core.exceptions import InvalidDatasetError, InvalidParameterError


class Dataset:
    """A named, source-aware wrapper around a pandas ``DataFrame``.

    The wrapper is immutable: its metadata cannot be reassigned and the wrapped
    frame cannot be retrieved for in-place mutation. Analysis functions therefore
    never alter the original data, and callers can safely reuse the same
    ``DataFrame`` elsewhere.

    Copying follows pandas semantics: scalar values are isolated, but mutable
    Python objects stored in object-typed cells are shared. CSV-loaded datasets
    contain scalar values, so this does not affect their isolation.

    Parameters
    ----------
    frame:
        The tabular data to wrap. A deep copy is taken immediately, so later
        mutations of ``frame`` do not affect the dataset.
    name:
        Human-readable dataset name used in reports. Must be a non-empty string.
    source:
        Optional source location, typically a file path. Stored as a string.

    Raises
    ------
    InvalidDatasetError
        If ``frame`` is not a :class:`pandas.DataFrame`.
    InvalidParameterError
        If ``name`` is empty or ``source`` has an unsupported type.

    Examples
    --------
    >>> dataset = Dataset(pd.DataFrame({"value": [1, 2, 3]}), name="demo")
    >>> dataset.shape
    (3, 1)
    >>> dataset.frame["value"].tolist()
    [1, 2, 3]
    """

    __slots__ = ("_frame", "_name", "_source")

    _frame: pd.DataFrame
    _name: str
    _source: str | None

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        name: str = "dataset",
        source: str | Path | None = None,
    ) -> None:
        if not isinstance(frame, pd.DataFrame):
            raise InvalidDatasetError(
                f"Dataset data must be a pandas DataFrame, got {type(frame).__name__}."
            )
        if not isinstance(name, str) or not name.strip():
            raise InvalidParameterError("Dataset name must be a non-empty string.")

        source_label: str | None
        if source is None:
            source_label = None
        elif isinstance(source, (str, Path)):
            source_label = str(source)
        else:
            raise InvalidParameterError(
                "Dataset source must be a string, pathlib.Path or None, "
                f"got {type(source).__name__}."
            )

        object.__setattr__(self, "_frame", frame.copy(deep=True))
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_source", source_label)

    @property
    def frame(self) -> pd.DataFrame:
        """Return a defensive deep copy of the underlying data."""
        return self._frame.copy(deep=True)

    @property
    def name(self) -> str:
        """Human-readable dataset name."""
        return self._name

    @property
    def source(self) -> str | None:
        """Original source location, if the dataset was loaded from one."""
        return self._source

    @property
    def shape(self) -> tuple[int, int]:
        """Tuple of ``(row_count, column_count)``."""
        return self._frame.shape

    @property
    def row_count(self) -> int:
        """Number of rows in the dataset."""
        return int(self._frame.shape[0])

    @property
    def column_count(self) -> int:
        """Number of columns in the dataset."""
        return int(self._frame.shape[1])

    @property
    def columns(self) -> pd.Index:
        """Column labels in their original order."""
        return self._frame.columns.copy()

    @property
    def dtypes(self) -> pd.Series:
        """Column dtypes keyed by column label."""
        return self._frame.dtypes.copy()

    @property
    def empty(self) -> bool:
        """Whether the dataset contains no rows or no columns."""
        return bool(self._frame.empty)

    def _view(self) -> pd.DataFrame:
        """Return the internal frame for read-only use by library modules.

        This method is private on purpose. Callers outside DataMendrix must use
        :attr:`frame`; mutating the returned object would break the guarantee
        that the original dataset is preserved.
        """
        return self._frame

    def __setattr__(self, name: str, value: object) -> None:
        """Prevent reassignment of dataset metadata or data."""
        raise AttributeError("Dataset instances are immutable; construct a new Dataset instead.")

    def __len__(self) -> int:
        """Return the number of rows."""
        return self.row_count

    def __repr__(self) -> str:
        """Return a concise, developer-friendly representation."""
        return f"Dataset(name={self._name!r}, shape={self.shape}, source={self._source!r})"
