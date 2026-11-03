"""Tests for constant and near-constant feature detection."""

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
    InvalidParameterError,
    detect_constant_features,
)


def _constant_dataset() -> Dataset:
    return Dataset(
        pd.DataFrame(
            {
                "constant": [5, 5, 5, 5],
                "near": [1, 1, 1, 2],
                "with_missing": [1, 1, np.nan, 1],
                "mixed": ["a", "b", "c", None],
            }
        )
    )


def test_constant_and_near_constant_columns_are_reported() -> None:
    report = detect_constant_features(_constant_dataset())

    assert report.row_count == 4
    assert report.dominance_threshold == 0.95
    assert report.column_names == ("constant", "near", "with_missing", "mixed")
    assert report.constant_column_names == ("constant", "with_missing")
    assert report.near_constant_column_names == ("constant", "with_missing")
    assert report.has_constant_features

    constant = report.column("constant")
    assert constant.non_missing_count == 4
    assert constant.missing_count == 0
    assert constant.unique_count == 1
    assert constant.dominant_value == 5
    assert constant.dominant_count == 4
    assert constant.dominant_fraction == 1.0
    assert constant.is_constant
    assert constant.is_near_constant
    assert not constant.has_missing

    with_missing = report.column("with_missing")
    assert with_missing.non_missing_count == 3
    assert with_missing.missing_count == 1
    assert with_missing.unique_count == 1
    assert with_missing.is_constant
    assert with_missing.is_near_constant
    assert with_missing.has_missing

    near = report.column("near")
    assert near.unique_count == 2
    assert near.dominant_fraction == pytest.approx(0.75)
    assert not near.is_constant
    assert not near.is_near_constant


def test_configurable_dominance_threshold_captures_near_constant_column() -> None:
    report = detect_constant_features(_constant_dataset(), dominance_threshold=0.7)

    near = report.column("near")
    assert report.dominance_threshold == 0.7
    assert near.is_near_constant
    assert near.dominant_value == 1
    assert near.dominant_count == 3
    assert report.near_constant_column_names == ("constant", "near", "with_missing")


def test_threshold_of_one_only_captures_exact_constants() -> None:
    report = detect_constant_features(_constant_dataset(), dominance_threshold=1.0)

    assert report.near_constant_column_names == ("constant", "with_missing")


def test_all_missing_column_is_not_constant() -> None:
    dataset = Dataset(pd.DataFrame({"missing": [np.nan, np.nan, np.nan]}))

    report = detect_constant_features(dataset)
    column = report.column("missing")

    assert column.non_missing_count == 0
    assert column.missing_count == 3
    assert column.unique_count == 0
    assert column.dominant_value is None
    assert column.dominant_count == 0
    assert column.dominant_fraction == 0.0
    assert not column.is_constant
    assert not column.is_near_constant
    assert not report.has_constant_features


def test_empty_dataset_is_safe() -> None:
    dataset = Dataset(pd.DataFrame({"value": pd.Series([], dtype="object")}))

    report = detect_constant_features(dataset)
    column = report.column("value")

    assert report.row_count == 0
    assert column.non_missing_count == 0
    assert column.missing_count == 0
    assert column.unique_count == 0
    assert not column.is_constant


def test_single_row_is_constant() -> None:
    report = detect_constant_features(Dataset(pd.DataFrame({"value": ["only"]})))

    column = report.column("value")
    assert column.unique_count == 1
    assert column.is_constant
    assert column.is_near_constant


def test_no_columns_is_safe() -> None:
    dataset = Dataset(pd.DataFrame(index=range(3)))

    report = detect_constant_features(dataset)

    assert report.row_count == 3
    assert report.columns == ()
    assert report.column_names == ()
    assert not report.has_constant_features


def test_infinite_values_can_be_constant() -> None:
    dataset = Dataset(pd.DataFrame({"value": [np.inf, np.inf, np.inf]}))

    report = detect_constant_features(dataset)
    column = report.column("value")

    assert column.unique_count == 1
    assert column.dominant_value == np.inf
    assert column.is_constant


def test_column_selection_accepts_single_and_non_string_labels() -> None:
    dataset = Dataset(pd.DataFrame({0: [1, 1, 1], 1: [1, 2, 3]}))

    report = detect_constant_features(dataset, columns=0)

    assert report.column_names == ("0",)
    assert report.column(0).column == "0"
    assert report.column_names == ("0",)


def test_unknown_column_raises() -> None:
    with pytest.raises(ColumnNotFoundError) as exc_info:
        detect_constant_features(_constant_dataset(), columns=["missing"])

    assert exc_info.value.columns == ("missing",)


def test_empty_column_selection_raises() -> None:
    with pytest.raises(InvalidParameterError, match="at least one"):
        detect_constant_features(_constant_dataset(), columns=[])


def test_ambiguous_duplicate_column_label_raises() -> None:
    dataset = Dataset(pd.DataFrame([[1, 2], [3, 4]], columns=["x", "x"]))

    with pytest.raises(InvalidParameterError, match="not unique"):
        detect_constant_features(dataset, columns="x")


def test_report_column_lookup_raises_for_unknown_label() -> None:
    report = detect_constant_features(_constant_dataset())

    with pytest.raises(ColumnNotFoundError):
        report.column("missing")


@pytest.mark.parametrize("threshold", [0, -0.1, 1.1, math.nan, math.inf, True, "0.9"])
def test_invalid_threshold_raises(threshold: Any) -> None:
    with pytest.raises(InvalidParameterError, match="dominance_threshold"):
        detect_constant_features(
            _constant_dataset(),
            dominance_threshold=cast(Any, threshold),
        )


def test_constant_detection_rejects_wrong_dataset_type() -> None:
    with pytest.raises(InvalidDatasetError, match="expects a Dataset"):
        detect_constant_features(cast(Any, pd.DataFrame({"value": [1]})))
