"""Immutable operation specifications for the repair engine.

Operations are explicit value objects. They describe a proposed change but do
not hold or modify a dataset; :func:`data_mendrix.repair.apply_repair` and
:func:`data_mendrix.repair.preview_repair` execute them against a dataset.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from data_mendrix.core.exceptions import InvalidParameterError

DropAxis = Literal["rows", "columns"]
DropHow = Literal["any", "all"]
FillStrategy = Literal["mean", "median", "mode", "constant"]
RepairKeepPolicy = Literal["first", "last", False]
ConversionErrorPolicy = Literal["raise", "skip", "coerce"]

_DROP_AXES = ("rows", "columns")
_DROP_HOWS = ("any", "all")
_FILL_STRATEGIES = ("mean", "median", "mode", "constant")
_CONVERSION_ERROR_POLICIES = ("raise", "skip", "coerce")


@dataclass(frozen=True, slots=True)
class DropMissing:
    """Drop rows or columns that contain missing values.

    Parameters
    ----------
    axis:
        ``"rows"`` or ``"columns"``. Defaults to ``"rows"``.
    columns:
        Optional label or iterable of labels limiting the columns considered
        when deciding what to drop. With ``axis="columns"``, only selected
        columns are candidates for removal; unselected columns are retained.
        ``None`` considers every column.
    how:
        ``"any"`` drops a row or column with at least one missing value;
        ``"all"`` drops it only when every considered value is missing.
        Defaults to ``"any"``.
    """

    axis: DropAxis = "rows"
    columns: Hashable | Iterable[Hashable] | None = None
    how: DropHow = "any"

    def __post_init__(self) -> None:
        if self.axis not in _DROP_AXES:
            raise InvalidParameterError("axis must be either 'rows' or 'columns'.")
        if self.how not in _DROP_HOWS:
            raise InvalidParameterError("how must be either 'any' or 'all'.")
        object.__setattr__(self, "columns", _freeze_columns(self.columns))


@dataclass(frozen=True, slots=True)
class FillMissing:
    """Fill missing values using a statistic or an explicit constant.

    Parameters
    ----------
    columns:
        Optional label or iterable of labels. ``None`` uses strategy-specific
        defaults: numeric columns for ``"mean"`` and ``"median"``; every column
        for ``"mode"`` and ``"constant"``.
    strategy:
        ``"mean"``, ``"median"``, ``"mode"`` or ``"constant"``. Defaults to
        ``"mean"``.
    value:
        Required non-missing scalar when ``strategy="constant"``. Must be
        ``None`` for every other strategy.
    """

    columns: Hashable | Iterable[Hashable] | None = None
    strategy: FillStrategy = "mean"
    value: object | None = None

    def __post_init__(self) -> None:
        if self.strategy not in _FILL_STRATEGIES:
            raise InvalidParameterError(
                "strategy must be one of: 'mean', 'median', 'mode', 'constant'."
            )
        if self.strategy == "constant":
            if self.value is None or not pd.api.types.is_scalar(self.value):
                raise InvalidParameterError(
                    "value must be a non-missing scalar when strategy='constant'."
                )
            if _is_missing_scalar(self.value):
                raise InvalidParameterError(
                    "value must be a non-missing scalar when strategy='constant'."
                )
        elif self.value is not None:
            raise InvalidParameterError("value is only valid when strategy='constant'.")
        object.__setattr__(self, "columns", _freeze_columns(self.columns))


@dataclass(frozen=True, slots=True)
class RemoveDuplicates:
    """Remove duplicate rows using a full-row or subset comparison.

    Parameters
    ----------
    subset:
        Optional label or iterable of labels to compare. ``None`` compares whole
        rows.
    keep:
        ``"first"`` keeps the first occurrence, ``"last"`` keeps the last, and
        ``False`` removes every occurrence of a duplicated row. Defaults to
        ``"first"``.
    """

    subset: Hashable | Iterable[Hashable] | None = None
    keep: RepairKeepPolicy = "first"

    def __post_init__(self) -> None:
        if self.keep not in ("first", "last") and self.keep is not False:
            raise InvalidParameterError("keep must be 'first', 'last' or False.")
        object.__setattr__(self, "subset", _freeze_columns(self.subset))


@dataclass(frozen=True, slots=True)
class ConvertTypes:
    """Convert selected columns to requested pandas-compatible dtypes.

    Parameters
    ----------
    conversions:
        Mapping of column label to target dtype. Targets are passed to pandas
        ``astype`` and may be strings, Python types, NumPy dtypes or pandas
        extension dtypes.
    errors:
        ``"raise"`` raises :class:`RepairConversionError` on the first failed
        conversion; ``"skip"`` leaves that column unchanged and records a
        :class:`RepairIssue`; ``"coerce"`` replaces values that cannot be cast
        with missing values where the target dtype can represent them. Defaults
        to ``"raise"``.
    """

    conversions: Mapping[Hashable, object]
    errors: ConversionErrorPolicy = "raise"

    def __post_init__(self) -> None:
        if not isinstance(self.conversions, Mapping) or not self.conversions:
            raise InvalidParameterError("conversions must be a non-empty mapping.")
        if self.errors not in _CONVERSION_ERROR_POLICIES:
            raise InvalidParameterError("errors must be 'raise', 'skip' or 'coerce'.")
        object.__setattr__(
            self,
            "conversions",
            MappingProxyType(dict(self.conversions)),
        )


RepairOperation = DropMissing | FillMissing | RemoveDuplicates | ConvertTypes


def _freeze_columns(
    selection: Hashable | Iterable[Hashable] | None,
) -> Hashable | Iterable[Hashable] | None:
    """Copy mutable selections into tuples to keep operation objects stable."""
    if selection is None or isinstance(selection, str) or not isinstance(selection, Iterable):
        return selection
    if isinstance(selection, (tuple, frozenset)):
        return selection
    try:
        return tuple(selection)
    except TypeError:
        return selection


def _is_missing_scalar(value: object) -> bool:
    """Return whether a value is a scalar missing-value sentinel."""
    try:
        result = pd.isna(cast(Any, value))
    except Exception:
        return False
    return isinstance(result, (bool, np.bool_)) and bool(result)
