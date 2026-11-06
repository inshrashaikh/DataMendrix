"""Tests for numeric descriptive statistics."""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from data_mendrix import (
    ColumnNotFoundError,
    Dataset,
    InvalidDatasetError,
    numeric_statistics,
)


def test_numeric_statistics_for_float_column() -> None:
    frame = pd.DataFrame(
        {
            "value": [1.0, 2.0, 3.0, 4.0, np.nan],
            "label": ["a", "b", "c", "d", "e"],
        }
    )

    stats = numeric_statistics(Dataset(frame))

    assert stats.column_names == ("value",)
    value = stats.column("value")
    assert value.count == 4
    assert value.missing_count == 1
    assert value.infinite_count == 0
    assert value.mean == pytest.approx(2.5)
    assert value.std == pytest.approx(1.2909944487358056)
    assert value.minimum == pytest.approx(1.0)
    assert value.first_quartile == pytest.approx(1.75)
    assert value.median == pytest.approx(2.5)
    assert value.third_quartile == pytest.approx(3.25)
    assert value.maximum == pytest.approx(4.0)
    assert value.iqr == pytest.approx(1.5)
    assert value.skewness == pytest.approx(0.0)
    assert value.kurtosis == pytest.approx(-1.2)


def test_nullable_integer_columns_are_supported() -> None:
    frame = pd.DataFrame(
        {
            "nullable": pd.array([1, 2, None, 4, 5], dtype="Int64"),
        }
    )

    stats = numeric_statistics(Dataset(frame)).column("nullable")

    assert stats.count == 4
    assert stats.missing_count == 1
    assert stats.mean == pytest.approx(3.0)
    assert stats.std == pytest.approx(1.8257418583505538)
    assert stats.first_quartile == pytest.approx(1.75)
    assert stats.median == pytest.approx(3.0)
    assert stats.third_quartile == pytest.approx(4.25)
    assert stats.iqr == pytest.approx(2.5)
    assert stats.skewness == pytest.approx(0.0)
    assert stats.kurtosis == pytest.approx(-3.3)


def test_non_numeric_and_boolean_columns_are_excluded() -> None:
    frame = pd.DataFrame(
        {
            "label": ["a", "b", "c"],
            "flag": [True, False, True],
            "when": pd.date_range("2024-01-01", periods=3),
        }
    )

    stats = numeric_statistics(Dataset(frame))

    assert not stats.has_numeric_columns
    assert stats.columns == ()
    assert stats.column_names == ()


def test_all_missing_numeric_column_is_included() -> None:
    stats = numeric_statistics(Dataset(pd.DataFrame({"x": [np.nan, np.nan, np.nan]}))).column("x")

    assert stats.count == 0
    assert stats.missing_count == 3
    assert stats.infinite_count == 0
    for value in (
        stats.mean,
        stats.std,
        stats.minimum,
        stats.first_quartile,
        stats.median,
        stats.third_quartile,
        stats.maximum,
        stats.iqr,
        stats.skewness,
        stats.kurtosis,
    ):
        assert math.isnan(value)


def test_constant_column_has_zero_dispersion() -> None:
    stats = numeric_statistics(Dataset(pd.DataFrame({"x": [5.0, 5.0, 5.0, 5.0, 5.0]}))).column("x")

    assert stats.mean == pytest.approx(5.0)
    assert stats.std == pytest.approx(0.0)
    assert stats.minimum == pytest.approx(5.0)
    assert stats.maximum == pytest.approx(5.0)
    assert stats.iqr == pytest.approx(0.0)


def test_infinite_values_are_counted() -> None:
    stats = numeric_statistics(Dataset(pd.DataFrame({"x": [1.0, np.inf, 3.0]}))).column("x")

    assert stats.count == 3
    assert stats.infinite_count == 1
    assert stats.minimum == pytest.approx(1.0)
    assert math.isinf(stats.maximum)


def test_insufficient_values_yield_nan_for_skew_and_kurtosis() -> None:
    stats = numeric_statistics(Dataset(pd.DataFrame({"x": [1.0, 2.0]}))).column("x")

    assert stats.count == 2
    assert stats.std == pytest.approx(0.7071067811865476)
    assert math.isnan(stats.skewness)
    assert math.isnan(stats.kurtosis)


def test_numeric_statistics_reject_wrong_dataset_type() -> None:
    with pytest.raises(InvalidDatasetError, match="expects a Dataset"):
        numeric_statistics(cast(Any, pd.DataFrame({"a": [1]})))


def test_statistics_equality_treats_nan_as_equal() -> None:
    left = numeric_statistics(Dataset(pd.DataFrame({"x": [1.0, 2.0]})))
    right = numeric_statistics(Dataset(pd.DataFrame({"x": [1.0, 2.0]})))

    assert left == right
    assert hash(left.column("x")) == hash(right.column("x"))


def test_statistics_equality_detects_real_differences() -> None:
    left = numeric_statistics(Dataset(pd.DataFrame({"x": [1.0, 2.0]})))
    right = numeric_statistics(Dataset(pd.DataFrame({"x": [1.0, 3.0]})))

    assert left != right
    assert left.column("x") != right.column("x")


def test_unknown_numeric_column_raises() -> None:
    stats = numeric_statistics(Dataset(pd.DataFrame({"a": [1]})))

    with pytest.raises(ColumnNotFoundError) as exc_info:
        stats.column("b")

    assert exc_info.value.columns == ("b",)


def test_numeric_column_lookup_accepts_non_string_label() -> None:
    stats = numeric_statistics(Dataset(pd.DataFrame({0: [1.0, 2.0]}))).column(0)

    assert stats.column == "0"
    assert stats.mean == pytest.approx(1.5)
