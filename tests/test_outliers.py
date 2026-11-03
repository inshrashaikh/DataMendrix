"""Tests for univariate outlier detection."""

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
    detect_outliers,
)


def _outlier_dataset() -> Dataset:
    return Dataset(
        pd.DataFrame(
            {
                "value": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0],
                "label": ["a", "b", "c", "d", "e", "f"],
            }
        )
    )


def test_iqr_detects_high_outlier_with_evidence() -> None:
    report = detect_outliers(_outlier_dataset())

    assert report.method == "iqr"
    assert report.threshold == 1.5
    assert report.row_count == 6
    assert report.column_names == ("value",)
    assert report.has_outliers
    assert report.outlier_indices == (5,)
    assert report.outlier_cell_count == 1

    column = report.column("value")
    assert column.observed_count == 6
    assert column.finite_count == 6
    assert column.missing_count == 0
    assert column.infinite_count == 0
    assert column.center == pytest.approx(3.5)
    assert column.scale == pytest.approx(2.5)
    assert column.lower_bound == pytest.approx(-1.5)
    assert column.upper_bound == pytest.approx(8.5)
    assert column.outlier_count == 1
    assert column.outlier_indices == (5,)
    assert column.outlier_fraction == pytest.approx(1 / 6)
    assert report.columns_with_outliers == (column,)

    evidence = column.evidence[0]
    assert evidence.row_index == 5
    assert evidence.value == pytest.approx(100.0)
    assert evidence.direction == "above"
    assert evidence.score == pytest.approx(38.1)
    assert evidence.lower_bound == pytest.approx(-1.5)
    assert evidence.upper_bound == pytest.approx(8.5)
    assert evidence.reason.startswith("value 100.0 is above the upper bound")


def test_iqr_detects_low_outlier() -> None:
    dataset = Dataset(pd.DataFrame({"value": [-100.0, 1.0, 2.0, 3.0, 4.0, 5.0]}))

    report = detect_outliers(dataset)
    column = report.column("value")

    assert column.outlier_indices == (0,)
    assert column.evidence[0].direction == "below"
    assert column.evidence[0].score == pytest.approx(40.5)
    assert "below the lower bound" in column.evidence[0].reason


def test_iqr_threshold_is_configurable() -> None:
    report = detect_outliers(_outlier_dataset(), threshold=40.0)

    assert report.threshold == 40.0
    assert not report.has_outliers
    assert report.column("value").outlier_count == 0


def test_zscore_detects_outlier_with_configured_threshold() -> None:
    report = detect_outliers(_outlier_dataset(), method="zscore", threshold=2.0)

    column = report.column("value")
    assert report.method == "zscore"
    assert report.threshold == 2.0
    assert report.outlier_indices == (5,)
    assert column.center == pytest.approx(115 / 6)
    assert column.evidence[0].direction == "above"
    assert column.evidence[0].score > 2.0


def test_zscore_uses_default_threshold() -> None:
    dataset = Dataset(pd.DataFrame({"value": [0.0] * 10 + [100.0]}))

    report = detect_outliers(dataset, method="zscore")

    assert report.threshold == 3.0
    assert report.outlier_indices == (10,)
    assert report.column("value").evidence[0].score > 3.0


def test_modified_zscore_detects_outlier_using_mad() -> None:
    report = detect_outliers(_outlier_dataset(), method="modified_zscore")

    column = report.column("value")
    assert report.method == "modified_zscore"
    assert report.threshold == 3.5
    assert report.outlier_indices == (5,)
    assert column.center == pytest.approx(3.5)
    assert column.scale == pytest.approx(1.5)
    assert column.lower_bound == pytest.approx(-4.2835, abs=1e-4)
    assert column.upper_bound == pytest.approx(11.2835, abs=1e-4)
    assert column.evidence[0].score == pytest.approx(43.3928, abs=1e-4)


def test_modified_zscore_handles_zero_mad() -> None:
    dataset = Dataset(pd.DataFrame({"value": [5.0] * 10 + [100.0]}))

    report = detect_outliers(dataset, method="modified_zscore")
    column = report.column("value")

    assert column.scale == 0.0
    assert column.lower_bound == column.upper_bound == 5.0
    assert column.outlier_indices == (10,)
    assert math.isinf(column.evidence[0].score)


def test_iqr_handles_zero_dispersion() -> None:
    dataset = Dataset(pd.DataFrame({"value": [5.0] * 10 + [100.0]}))

    report = detect_outliers(dataset)
    column = report.column("value")

    assert column.scale == 0.0
    assert column.outlier_indices == (10,)
    assert math.isinf(column.evidence[0].score)


def test_missing_values_are_excluded_and_index_labels_are_reported() -> None:
    dataset = Dataset(
        pd.DataFrame(
            {"value": [1.0, 2.0, np.nan, 3.0, 4.0, 5.0, 100.0]},
            index=list("abcdefg"),
        )
    )

    report = detect_outliers(dataset)
    column = report.column("value")

    assert column.missing_count == 1
    assert column.observed_count == 6
    assert column.finite_count == 6
    assert report.outlier_indices == ("g",)
    assert column.evidence[0].row_index == "g"


def test_infinities_are_flagged_and_excluded_from_baseline() -> None:
    dataset = Dataset(pd.DataFrame({"value": [1.0, 2.0, 3.0, 4.0, np.inf, -np.inf]}))

    report = detect_outliers(dataset)
    column = report.column("value")

    assert column.observed_count == 6
    assert column.finite_count == 4
    assert column.infinite_count == 2
    assert column.outlier_indices == (4, 5)
    assert [item.direction for item in column.evidence] == ["above", "below"]
    assert all(math.isinf(item.score) for item in column.evidence)
    assert column.lower_bound == pytest.approx(-0.5)
    assert column.upper_bound == pytest.approx(5.5)


def test_all_infinite_values_are_flagged_without_a_finite_baseline() -> None:
    dataset = Dataset(pd.DataFrame({"value": [np.inf, -np.inf]}))

    report = detect_outliers(dataset)
    column = report.column("value")

    assert column.finite_count == 0
    assert math.isnan(column.center)
    assert math.isnan(column.lower_bound)
    assert math.isnan(column.upper_bound)
    assert column.outlier_indices == (0, 1)


def test_all_missing_values_produce_no_outliers() -> None:
    dataset = Dataset(pd.DataFrame({"value": [np.nan, np.nan]}))

    report = detect_outliers(dataset)
    column = report.column("value")

    assert column.observed_count == 0
    assert column.missing_count == 2
    assert column.outlier_count == 0
    assert column.outlier_fraction == 0.0
    assert math.isnan(column.center)


def test_empty_numeric_column_is_safe() -> None:
    dataset = Dataset(pd.DataFrame({"value": pd.Series([], dtype="float64")}))

    report = detect_outliers(dataset)
    column = report.column("value")

    assert report.row_count == 0
    assert column.row_count == 0
    assert column.outlier_count == 0
    assert math.isnan(column.center)


@pytest.mark.parametrize("values", [[42.0], [1.0, 2.0], [5.0, 5.0, 5.0]])
def test_small_or_constant_samples_are_safe(values: list[float]) -> None:
    report = detect_outliers(Dataset(pd.DataFrame({"value": values})))

    assert report.column("value").outlier_count == 0


def test_outlier_indices_are_unioned_across_columns() -> None:
    dataset = Dataset(
        pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0],
                "b": [-100.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                "label": ["a", "b", "c", "d", "e", "f"],
            }
        )
    )

    report = detect_outliers(dataset)

    assert report.column_names == ("a", "b")
    assert report.outlier_indices == (0, 5)
    assert report.outlier_cell_count == 2
    assert tuple(column.column for column in report.columns_with_outliers) == ("a", "b")


def test_non_numeric_columns_are_excluded_by_default() -> None:
    dataset = Dataset(pd.DataFrame({"label": ["a", "b"], "flag": [True, False]}))

    report = detect_outliers(dataset)

    assert report.columns == ()
    assert report.column_names == ()
    assert not report.has_outliers
    with pytest.raises(ColumnNotFoundError):
        report.column("label")


def test_report_column_lookup_raises_for_unknown_label() -> None:
    report = detect_outliers(_outlier_dataset())

    with pytest.raises(ColumnNotFoundError):
        report.column("missing")


def test_explicit_non_numeric_column_raises() -> None:
    dataset = Dataset(pd.DataFrame({"value": [1.0, 2.0], "label": ["a", "b"]}))

    with pytest.raises(InvalidParameterError, match="numeric columns"):
        detect_outliers(dataset, columns="label")


def test_boolean_column_is_not_numeric_for_outlier_detection() -> None:
    dataset = Dataset(pd.DataFrame({"flag": [True, False, True, False]}))

    with pytest.raises(InvalidParameterError, match="numeric columns"):
        detect_outliers(dataset, columns="flag")


def test_column_selection_accepts_a_single_non_string_label() -> None:
    dataset = Dataset(pd.DataFrame({0: [1.0, 2.0, 3.0, 4.0, 5.0, 100.0], 1: ["a"] * 6}))

    report = detect_outliers(dataset, columns=0)

    assert report.column_names == ("0",)
    assert report.column(0).column == "0"


def test_unknown_column_raises() -> None:
    with pytest.raises(ColumnNotFoundError) as exc_info:
        detect_outliers(_outlier_dataset(), columns=["missing"])

    assert exc_info.value.columns == ("missing",)


def test_empty_column_selection_raises() -> None:
    with pytest.raises(InvalidParameterError, match="at least one"):
        detect_outliers(_outlier_dataset(), columns=[])


def test_ambiguous_duplicate_column_label_raises() -> None:
    dataset = Dataset(pd.DataFrame([[1.0, 2.0], [2.0, 3.0]], columns=["x", "x"]))

    with pytest.raises(InvalidParameterError, match="not unique"):
        detect_outliers(dataset, columns="x")


@pytest.mark.parametrize("method", ["MAD", "", None, 1])
def test_invalid_method_raises(method: Any) -> None:
    with pytest.raises(InvalidParameterError, match="method"):
        detect_outliers(_outlier_dataset(), method=cast(Any, method))


@pytest.mark.parametrize("threshold", [0, -1, math.nan, math.inf, True, "3"])
def test_invalid_threshold_raises(threshold: Any) -> None:
    with pytest.raises(InvalidParameterError, match="positive finite"):
        detect_outliers(_outlier_dataset(), threshold=cast(Any, threshold))


class _BrokenIterable:
    def __iter__(self) -> None:
        raise TypeError("not iterable after all")


def test_broken_column_iterable_raises_parameter_error() -> None:
    with pytest.raises(InvalidParameterError, match="iterable of labels"):
        detect_outliers(_outlier_dataset(), columns=cast(Any, _BrokenIterable()))


def test_outlier_detection_rejects_wrong_dataset_type() -> None:
    with pytest.raises(InvalidDatasetError, match="expects a Dataset"):
        detect_outliers(cast(Any, pd.DataFrame({"value": [1.0]})))
