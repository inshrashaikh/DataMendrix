"""Shared column-selection helpers for detection APIs."""

from __future__ import annotations

from collections.abc import Hashable, Iterable
from typing import cast

import pandas as pd

from data_mendrix.core.exceptions import ColumnNotFoundError, InvalidParameterError


def resolve_column_selection(
    frame: pd.DataFrame,
    selection: Hashable | Iterable[Hashable] | None,
    *,
    parameter: str = "columns",
) -> list[Hashable] | None:
    """Normalise and validate a column label or iterable of labels.

    ``None`` means "use the caller's default selection". A tuple that matches an
    existing column label is treated as a single label, which supports columns
    with tuple labels and MultiIndex columns.
    """
    if selection is None:
        return None

    if isinstance(selection, str) or not isinstance(selection, Iterable):
        columns: list[Hashable] = [selection]
    elif _is_existing_label(frame, selection):
        columns = [cast(Hashable, selection)]
    else:
        try:
            columns = list(selection)
        except TypeError as exc:
            raise InvalidParameterError(
                f"{parameter} must be a column label or an iterable of labels."
            ) from exc

    if not columns:
        raise InvalidParameterError(f"{parameter} must contain at least one column.")

    missing = tuple(column for column in columns if not _is_existing_label(frame, column))
    if missing:
        raise ColumnNotFoundError(missing)

    return columns


def unique_column_position(
    frame: pd.DataFrame,
    label: Hashable,
    *,
    parameter: str = "column",
) -> int:
    """Return the position of a column label, rejecting ambiguous labels."""
    try:
        location = frame.columns.get_loc(label)
    except (KeyError, TypeError) as exc:
        raise ColumnNotFoundError((label,)) from exc

    if not isinstance(location, int):
        raise InvalidParameterError(
            f"{parameter} {label!r} is not unique; select a column with a unique label."
        )

    return location


def _is_existing_label(frame: pd.DataFrame, label: object) -> bool:
    """Return whether an object is a column label, tolerating unhashable input."""
    try:
        return label in frame.columns
    except TypeError:
        return False
