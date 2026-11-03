"""Tests for class-imbalance analysis."""

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
    analyze_class_imbalance,
)


def _dataset(labels: list[object]) -> Dataset:
    return Dataset(
        pd.DataFrame(
            {
                "target": labels,
                "feature": list(range(len(labels))),
            }
        )
    )


def test_binary_imbalance_reports_counts_and_ratios() -> None:
    report = analyze_class_imbalance(_dataset(["A"] * 9 + ["B"]), target="target")

    assert report.target == "target"
    assert report.row_count == 10
    assert report.observed_count == 10
    assert report.missing_count == 0
    assert report.class_count == 2
    assert report.is_binary
    assert not report.is_multiclass
    assert [(item.label, item.count) for item in report.classes] == [("A", 9), ("B", 1)]
    assert [item.percentage for item in report.classes] == pytest.approx([0.9, 0.1])
    assert report.majority_class is not None
    assert report.majority_class.label == "A"
    assert report.minority_class is not None
    assert report.minority_class.label == "B"
    assert report.majority_fraction == pytest.approx(0.9)
    assert report.minority_fraction == pytest.approx(0.1)
    assert report.imbalance_ratio == pytest.approx(9.0)
    assert report.imbalance_ratio_threshold == 2.0
    assert report.minority_fraction_threshold == 0.1
    assert report.is_imbalanced


def test_balanced_binary_target_is_not_imbalanced() -> None:
    report = analyze_class_imbalance(_dataset(["A"] * 5 + ["B"] * 5), target="target")

    assert [item.label for item in report.classes] == ["A", "B"]
    assert report.imbalance_ratio == pytest.approx(1.0)
    assert report.majority_fraction == pytest.approx(0.5)
    assert report.minority_fraction == pytest.approx(0.5)
    assert not report.is_imbalanced


def test_multiclass_imbalance_is_supported() -> None:
    report = analyze_class_imbalance(_dataset(["A"] * 6 + ["B"] * 3 + ["C"]), target="target")

    assert report.class_count == 3
    assert not report.is_binary
    assert report.is_multiclass
    assert [(item.label, item.count) for item in report.classes] == [("A", 6), ("B", 3), ("C", 1)]
    assert report.majority_fraction == pytest.approx(0.6)
    assert report.minority_fraction == pytest.approx(0.1)
    assert report.imbalance_ratio == pytest.approx(6.0)
    assert report.is_imbalanced


def test_missing_target_values_are_excluded_from_percentages() -> None:
    report = analyze_class_imbalance(
        _dataset(["A", "A", "B", None, np.nan]),
        target="target",
    )

    assert report.row_count == 5
    assert report.observed_count == 3
    assert report.missing_count == 2
    assert [(item.label, item.count) for item in report.classes] == [("A", 2), ("B", 1)]
    assert [item.percentage for item in report.classes] == pytest.approx([2 / 3, 1 / 3])
    assert report.imbalance_ratio == pytest.approx(2.0)
    assert not report.is_imbalanced


def test_one_class_is_not_classified_as_imbalanced() -> None:
    report = analyze_class_imbalance(_dataset(["A"] * 4), target="target")

    assert report.class_count == 1
    assert not report.is_binary
    assert not report.is_multiclass
    assert report.majority_class is not None
    assert report.minority_class is not None
    assert report.majority_class.label == "A"
    assert report.minority_class.label == "A"
    assert report.majority_fraction == pytest.approx(1.0)
    assert report.minority_fraction == pytest.approx(1.0)
    assert report.imbalance_ratio == pytest.approx(1.0)
    assert not report.is_imbalanced


def test_all_missing_target_is_safe() -> None:
    report = analyze_class_imbalance(_dataset([np.nan, None]), target="target")

    assert report.observed_count == 0
    assert report.missing_count == 2
    assert report.class_count == 0
    assert report.classes == ()
    assert report.majority_class is None
    assert report.minority_class is None
    assert math.isnan(report.majority_fraction)
    assert math.isnan(report.minority_fraction)
    assert math.isnan(report.imbalance_ratio)
    assert not report.is_imbalanced


def test_empty_target_is_safe() -> None:
    dataset = Dataset(pd.DataFrame({"target": pd.Series([], dtype="object")}))

    report = analyze_class_imbalance(dataset, target="target")

    assert report.row_count == 0
    assert report.observed_count == 0
    assert report.class_count == 0
    assert not report.is_imbalanced


def test_imbalance_thresholds_are_configurable() -> None:
    dataset = _dataset(["A"] * 3 + ["B"] * 2)

    default_report = analyze_class_imbalance(dataset, target="target")
    ratio_report = analyze_class_imbalance(
        dataset,
        target="target",
        imbalance_ratio_threshold=1.4,
    )
    fraction_report = analyze_class_imbalance(
        dataset,
        target="target",
        imbalance_ratio_threshold=10.0,
        minority_fraction_threshold=0.5,
    )
    disabled_report = analyze_class_imbalance(
        dataset,
        target="target",
        imbalance_ratio_threshold=10.0,
        minority_fraction_threshold=0.0,
    )

    assert default_report.imbalance_ratio == pytest.approx(1.5)
    assert not default_report.is_imbalanced
    assert ratio_report.is_imbalanced
    assert fraction_report.is_imbalanced
    assert not disabled_report.is_imbalanced


def test_infinite_target_labels_are_supported() -> None:
    report = analyze_class_imbalance(
        _dataset([np.inf, np.inf, -np.inf]),
        target="target",
    )

    assert report.class_count == 2
    assert report.classes[0].label == np.inf
    assert report.classes[0].count == 2
    assert report.classes[1].label == -np.inf
    assert report.imbalance_ratio == pytest.approx(2.0)


def test_non_string_target_label_is_supported() -> None:
    dataset = Dataset(pd.DataFrame({0: [1, 1, 2], 1: ["a", "b", "c"]}))

    report = analyze_class_imbalance(dataset, target=0)

    assert report.target == "0"
    assert [(item.label, item.count) for item in report.classes] == [(1, 2), (2, 1)]


def test_unknown_target_raises() -> None:
    with pytest.raises(ColumnNotFoundError) as exc_info:
        analyze_class_imbalance(_dataset(["A"]), target="missing")

    assert exc_info.value.columns == ("missing",)


def test_ambiguous_target_label_raises() -> None:
    dataset = Dataset(pd.DataFrame([[1, 2], [3, 4]], columns=["target", "target"]))

    with pytest.raises(InvalidParameterError, match="not unique"):
        analyze_class_imbalance(dataset, target="target")


@pytest.mark.parametrize("threshold", [1, 0, -1, math.nan, math.inf, True, "2"])
def test_invalid_ratio_threshold_raises(threshold: Any) -> None:
    with pytest.raises(InvalidParameterError, match="imbalance_ratio_threshold"):
        analyze_class_imbalance(
            _dataset(["A", "B"]),
            target="target",
            imbalance_ratio_threshold=cast(Any, threshold),
        )


@pytest.mark.parametrize("threshold", [-0.1, 0.6, 1, math.nan, math.inf, True, "0.1"])
def test_invalid_minority_threshold_raises(threshold: Any) -> None:
    with pytest.raises(InvalidParameterError, match="minority_fraction_threshold"):
        analyze_class_imbalance(
            _dataset(["A", "B"]),
            target="target",
            minority_fraction_threshold=cast(Any, threshold),
        )


def test_imbalance_analysis_rejects_wrong_dataset_type() -> None:
    with pytest.raises(InvalidDatasetError, match="expects a Dataset"):
        analyze_class_imbalance(cast(Any, pd.DataFrame({"target": ["A"]})), target="target")
