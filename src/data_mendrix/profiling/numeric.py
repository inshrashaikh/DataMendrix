"""Descriptive statistics for numeric columns.

Statistics use pandas' standard non-missing semantics and sample standard
deviation (``ddof=1``). Boolean and complex columns are excluded: booleans are
categorical flags, and complex values have no meaningful ordering for quartiles.
Infinite values are retained in the calculations but counted separately so
callers can decide whether to treat them as data-quality issues.
"""

from __future__ import annotations

import math
from collections.abc import Hashable
from dataclasses import dataclass, fields
from typing import cast

import numpy as np
import pandas as pd

from data_mendrix.core.dataset import Dataset
from data_mendrix.core.exceptions import ColumnNotFoundError, InvalidDatasetError


@dataclass(frozen=True, slots=True, eq=False)
class NumericColumnStatistics:
    """Descriptive statistics for one numeric column.

    ``count`` counts non-missing values, including infinities. When fewer values
    are available than a statistic requires, that statistic is ``nan``. In
    particular, ``std`` needs two values, ``skewness`` needs three, and
    ``kurtosis`` needs four. Quartiles use pandas' default linear interpolation;
    skewness and excess (Fisher) kurtosis follow pandas' definitions.

    Equality treats ``nan`` in the same field on both sides as equal, so two
    reports built from identical data compare equal even when a statistic is
    undefined. All other fields compare with the usual ``==`` semantics.
    """

    column: str
    count: int
    missing_count: int
    infinite_count: int
    mean: float
    std: float
    minimum: float
    first_quartile: float
    median: float
    third_quartile: float
    maximum: float
    iqr: float
    skewness: float
    kurtosis: float

    def __eq__(self, other: object) -> bool:
        if other.__class__ is not NumericColumnStatistics:
            return NotImplemented
        return all(
            _nan_aware_equal(getattr(self, field.name), getattr(other, field.name))
            for field in fields(self)
        )

    def __hash__(self) -> int:
        values = tuple(_hashable_nan(getattr(self, field.name)) for field in fields(self))
        return hash(values)


_NAN_HASH_SENTINEL = ("<nan>",)


def _nan_aware_equal(left: object, right: object) -> bool:
    """Compare two field values, treating paired ``nan`` floats as equal."""
    if (
        isinstance(left, float)
        and isinstance(right, float)
        and math.isnan(left)
        and math.isnan(right)
    ):
        return True
    return bool(left == right)


def _hashable_nan(value: object) -> object:
    """Canonicalise ``nan`` so equal statistics hash equally."""
    if isinstance(value, float) and math.isnan(value):
        return _NAN_HASH_SENTINEL
    return value


@dataclass(frozen=True, slots=True)
class NumericStatistics:
    """Numeric descriptive statistics for every eligible column."""

    columns: tuple[NumericColumnStatistics, ...]

    @property
    def column_names(self) -> tuple[str, ...]:
        """Names of the columns included in the report, in dataset order."""
        return tuple(column.column for column in self.columns)

    @property
    def has_numeric_columns(self) -> bool:
        """Whether at least one numeric column was found."""
        return bool(self.columns)

    def column(self, name: Hashable) -> NumericColumnStatistics:
        """Return statistics for one numeric column by label.

        Non-string labels are matched using the string representation stored in
        the report.

        Raises
        ------
        ColumnNotFoundError
            If the label is not a numeric column in the report.
        """
        label = str(name)
        for column in self.columns:
            if column.column == label:
                return column
        raise ColumnNotFoundError((name,))


def numeric_statistics(dataset: Dataset) -> NumericStatistics:
    """Compute descriptive statistics for numeric columns.

    Parameters
    ----------
    dataset:
        Dataset to inspect. The dataset is not modified.
    Returns
    -------
    NumericStatistics
        One entry per real-valued numeric column. Columns with an all-missing
        numeric dtype are included, with ``count`` equal to zero and numeric
        statistics equal to ``nan``.

    Raises
    ------
    InvalidDatasetError
        If ``dataset`` is not a :class:`~data_mendrix.core.dataset.Dataset`.

    Examples
    --------
    >>> stats = numeric_statistics(dataset)
    >>> stats.column("price").median
    19.99
    """
    if not isinstance(dataset, Dataset):
        raise InvalidDatasetError(
            f"numeric_statistics() expects a Dataset, got {type(dataset).__name__}."
        )

    frame = dataset._view()
    columns: list[NumericColumnStatistics] = []

    for position, label in enumerate(frame.columns):
        series = frame.iloc[:, position]
        dtype = series.dtype
        if (
            not pd.api.types.is_numeric_dtype(dtype)
            or pd.api.types.is_bool_dtype(dtype)
            or pd.api.types.is_complex_dtype(dtype)
        ):
            continue

        columns.append(
            _column_statistics(
                column=str(label),
                values=series.dropna(),
                missing_count=int(series.isna().sum()),
            )
        )

    return NumericStatistics(tuple(columns))


def _column_statistics(
    *,
    column: str,
    values: pd.Series,
    missing_count: int,
) -> NumericColumnStatistics:
    """Build statistics for one non-missing numeric series."""
    count = int(values.size)
    infinite_count = int(np.isinf(values.to_numpy(dtype="float64")).sum())

    if count == 0:
        return NumericColumnStatistics(
            column=column,
            count=0,
            missing_count=missing_count,
            infinite_count=0,
            mean=math.nan,
            std=math.nan,
            minimum=math.nan,
            first_quartile=math.nan,
            median=math.nan,
            third_quartile=math.nan,
            maximum=math.nan,
            iqr=math.nan,
            skewness=math.nan,
            kurtosis=math.nan,
        )

    # Infinities are retained and counted separately. Arithmetic involving them
    # legitimately produces ``inf``/``nan``; suppress NumPy's expected warnings
    # so a profile of such data stays quiet.
    with np.errstate(invalid="ignore", divide="ignore"):
        quantiles = values.quantile([0.25, 0.5, 0.75]).tolist()
        mean = float(values.mean())
        std = float(values.std())
        minimum = float(values.min())
        maximum = float(values.max())
        skewness = float(cast(float, values.skew())) if count > 2 else math.nan
        kurtosis = float(cast(float, values.kurt())) if count > 3 else math.nan

    first_quartile, median, third_quartile = (float(quantile) for quantile in quantiles)

    return NumericColumnStatistics(
        column=column,
        count=count,
        missing_count=missing_count,
        infinite_count=infinite_count,
        mean=mean,
        std=std,
        minimum=minimum,
        first_quartile=first_quartile,
        median=median,
        third_quartile=third_quartile,
        maximum=maximum,
        iqr=third_quartile - first_quartile,
        skewness=skewness,
        kurtosis=kurtosis,
    )
