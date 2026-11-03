"""CSV loading utilities.

The loader performs I/O and parsing only. It does not validate dataset quality
or coerce column types beyond pandas' normal CSV inference; that separation
lets users load and audit imperfect data instead of having it rejected.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import IO, TypeAlias

import pandas as pd

from data_mendrix.core.dataset import Dataset
from data_mendrix.core.exceptions import (
    DatasetLoadError,
    InvalidParameterError,
    UnsupportedFileTypeError,
)

CsvSource: TypeAlias = str | Path | IO[str]
NaValues: TypeAlias = str | Sequence[str] | Mapping[str, Sequence[str]]


def load_csv(
    source: CsvSource,
    *,
    encoding: str = "utf-8",
    separator: str = ",",
    na_values: NaValues | None = None,
    keep_default_na: bool = True,
    comment: str | None = None,
    skipinitialspace: bool = False,
    thousands: str | None = None,
    decimal: str = ".",
) -> Dataset:
    """Load a CSV dataset without modifying the source.

    Parameters
    ----------
    source:
        Path to a ``.csv`` file, or an already-open text stream. Streams are
        read from their current position and are not closed by this function.
    encoding:
        Text encoding used to decode the file. Defaults to UTF-8.
    separator:
        Single-character field separator. Use ``"\\t"`` for TSV content.
    na_values:
        Additional strings that pandas should treat as missing values. May be a
        single string, a sequence of strings, or a mapping of column name to
        strings.
    keep_default_na:
        Whether to keep pandas' built-in missing-value tokens (such as ``NA``
        and an empty field) in addition to ``na_values``.
    comment:
        Optional single character that starts a comment line.
    skipinitialspace:
        Whether to skip spaces after the separator.
    thousands:
        Optional single-character thousands separator, for example ``","``.
    decimal:
        Single character used as the decimal point.

    Returns
    -------
    Dataset
        A dataset containing a defensive copy of the parsed data.

    Raises
    ------
    InvalidParameterError
        If a loader option has an invalid type or value.
    UnsupportedFileTypeError
        If a path does not end in ``.csv``.
    DatasetLoadError
        If the source cannot be read, is empty, has no columns, cannot be
        decoded, or contains malformed CSV syntax.

    Examples
    --------
    >>> dataset = load_csv("data.csv")
    >>> dataset.shape
    (100, 8)
    """
    _validate_options(
        encoding=encoding,
        separator=separator,
        na_values=na_values,
        keep_default_na=keep_default_na,
        comment=comment,
        skipinitialspace=skipinitialspace,
        thousands=thousands,
        decimal=decimal,
    )

    read_source, name, source_label = _resolve_source(source)
    source_display = repr(source_label or name)

    try:
        frame = pd.read_csv(
            read_source,
            encoding=encoding,
            sep=separator,
            na_values=na_values,
            keep_default_na=keep_default_na,
            comment=comment,
            skipinitialspace=skipinitialspace,
            thousands=thousands,
            decimal=decimal,
        )
    except pd.errors.EmptyDataError as exc:
        raise DatasetLoadError(f"CSV source {source_display} contains no data.") from exc
    except pd.errors.ParserError as exc:
        raise DatasetLoadError(f"Could not parse CSV source {source_display}: {exc}") from exc
    except UnicodeDecodeError as exc:
        raise DatasetLoadError(
            f"Could not decode CSV source {source_display} using encoding {encoding!r}."
        ) from exc
    except OSError as exc:
        raise DatasetLoadError(f"Could not read CSV source {source_display}: {exc}") from exc

    if frame.shape[1] == 0:
        raise DatasetLoadError(f"CSV source {source_display} produced a dataset with no columns.")

    return Dataset(frame, name=name, source=source_label)


def _resolve_source(source: CsvSource) -> tuple[CsvSource, str, str | None]:
    """Resolve a user-supplied source into something pandas can read.

    Returns the source to pass to pandas, a dataset name, and an optional
    source label (the path as a string).
    """
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser()
        if path.suffix.lower() != ".csv":
            raise UnsupportedFileTypeError(
                f"Unsupported dataset file type {path.suffix!r}; only '.csv' is supported."
            )
        if not path.exists():
            raise DatasetLoadError(f"CSV file not found: {path}")
        if not path.is_file():
            raise DatasetLoadError(f"CSV source is not a file: {path}")
        return path, path.stem or "dataset", str(path)

    if hasattr(source, "read"):
        stream_name = getattr(source, "name", None)
        name = _name_from_stream(stream_name)
        source_label = str(stream_name) if isinstance(stream_name, (str, Path)) else None
        return source, name, source_label

    raise InvalidParameterError(
        f"CSV source must be a file path or a readable text stream, got {type(source).__name__}."
    )


def _name_from_stream(stream_name: object) -> str:
    """Derive a dataset name from a stream's optional ``name`` attribute."""
    if isinstance(stream_name, (str, Path)):
        stem = Path(stream_name).stem
        return stem or "dataset"
    return "dataset"


def _validate_options(
    *,
    encoding: str,
    separator: str,
    na_values: NaValues | None,
    keep_default_na: bool,
    comment: str | None,
    skipinitialspace: bool,
    thousands: str | None,
    decimal: str,
) -> None:
    """Validate loader options before any I/O takes place."""
    if not isinstance(encoding, str) or not encoding:
        raise InvalidParameterError("encoding must be a non-empty string.")
    _require_single_character(separator, "separator", forbidden="\r\n")
    _require_single_character(decimal, "decimal", forbidden="\r\n")
    if comment is not None:
        _require_single_character(comment, "comment", forbidden="\r\n")
    if thousands is not None:
        _require_single_character(thousands, "thousands", forbidden="\r\n")
    if not isinstance(keep_default_na, bool):
        raise InvalidParameterError("keep_default_na must be a boolean.")
    if not isinstance(skipinitialspace, bool):
        raise InvalidParameterError("skipinitialspace must be a boolean.")
    _validate_na_values(na_values)


def _require_single_character(
    value: object,
    parameter: str,
    *,
    forbidden: str,
) -> None:
    """Raise when an option is not exactly one allowed character."""
    if not isinstance(value, str) or len(value) != 1 or value in forbidden:
        raise InvalidParameterError(
            f"{parameter} must be a single character that is not a line break."
        )


def _validate_na_values(na_values: NaValues | None) -> None:
    """Validate the ``na_values`` option against its supported shapes."""
    if na_values is None or isinstance(na_values, str):
        return
    if isinstance(na_values, Mapping):
        for column, tokens in na_values.items():
            if not isinstance(column, str):
                raise InvalidParameterError(
                    "na_values mapping keys must be column names (strings)."
                )
            if isinstance(tokens, str):
                continue
            if not isinstance(tokens, Sequence) or not all(
                isinstance(token, str) for token in tokens
            ):
                raise InvalidParameterError(
                    "na_values mapping values must be strings or sequences of strings."
                )
        return
    if isinstance(na_values, Sequence) and all(isinstance(token, str) for token in na_values):
        return
    raise InvalidParameterError(
        "na_values must be None, a string, a sequence of strings, "
        "or a mapping of column names to strings."
    )
