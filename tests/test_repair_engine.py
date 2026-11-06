"""Tests for repair plans, previews and application results."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from data_mendrix import (
    ConvertTypes,
    Dataset,
    DropMissing,
    FillMissing,
    InvalidDatasetError,
    InvalidParameterError,
    RemoveDuplicates,
    RepairConversionError,
    RepairReport,
    RepairResult,
    apply_repair,
    preview_repair,
)


def _repair_dataset() -> Dataset:
    return Dataset(
        pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 4],
                "score": [1.0, np.nan, 3.0, 4.0, 4.0],
                "label": ["a", "b", None, "a", "a"],
            }
        ),
        name="repair",
        source="memory",
    )


def _plan() -> list[object]:
    return [
        FillMissing(columns="label", strategy="constant", value="unknown"),
        FillMissing(columns="score", strategy="mean"),
        RemoveDuplicates(subset="id"),
        ConvertTypes(conversions={"score": "Int64"}),
    ]


def test_preview_repair_reports_plan_without_mutating_dataset() -> None:
    dataset = _repair_dataset()
    before = dataset.frame

    report = preview_repair(dataset, cast(Any, _plan()))

    assert isinstance(report, RepairReport)
    assert report.operation_count == 4
    assert report.row_count_before == 5
    assert report.row_count_after == 4
    assert report.column_count_before == 3
    assert report.column_count_after == 3
    assert report.changed
    assert report.change_count == 4
    assert report.affected_rows == (2, 1, 4)
    assert report.affected_columns == ("label", "score", "id")
    assert not report.has_issues
    assert_frame_equal(dataset.frame, before)


def test_apply_repair_returns_new_dataset_and_matching_report() -> None:
    dataset = _repair_dataset()
    before = dataset.frame

    result = apply_repair(dataset, cast(Any, _plan()))

    assert isinstance(result, RepairResult)
    assert result.dataset is not dataset
    assert result.dataset.name == "repair"
    assert result.dataset.source == "memory"
    assert result.dataset.shape == (4, 3)
    assert result.dataset.frame["label"].tolist() == ["a", "b", "unknown", "a"]
    assert result.dataset.frame["score"].tolist() == [1, 3, 3, 4]
    assert str(result.dataset.frame["score"].dtype) == "Int64"
    assert result.report == preview_repair(dataset, cast(Any, _plan()))
    assert_frame_equal(dataset.frame, before)


def test_apply_repair_accepts_single_operation() -> None:
    dataset = Dataset(pd.DataFrame({"a": [1.0, np.nan, 3.0]}))

    result = apply_repair(dataset, FillMissing(columns="a", strategy="mean"))

    assert result.dataset.frame["a"].tolist() == [1.0, 2.0, 3.0]
    assert result.report.operation_count == 1


def test_preview_repair_accepts_operation_iterators() -> None:
    dataset = Dataset(pd.DataFrame({"a": [1.0, np.nan, 3.0]}))

    report = preview_repair(
        dataset,
        (operation for operation in [DropMissing(axis="rows"), FillMissing(columns="a")]),
    )

    assert report.operation_count == 2
    assert report.row_count_after == 2
    assert report.operations[1].row_count_before == 2


def test_apply_repair_with_empty_plan_returns_unchanged_copy() -> None:
    dataset = _repair_dataset()
    before = dataset.frame

    result = apply_repair(dataset, [])

    assert result.dataset is not dataset
    assert result.dataset.name == dataset.name
    assert result.dataset.source == dataset.source
    assert_frame_equal(result.dataset.frame, before)
    assert result.report.operation_count == 0
    assert not result.report.changed
    assert result.report.affected_rows == ()
    assert result.report.affected_columns == ()
    assert result.report.issues == ()


def test_repair_report_aggregates_issues() -> None:
    dataset = Dataset(pd.DataFrame({"bad": ["x", "y"], "missing": [None, None]}))

    result = apply_repair(
        dataset,
        [
            ConvertTypes(conversions={"bad": "Int64"}, errors="skip"),
            FillMissing(columns="missing", strategy="mode"),
        ],
    )
    report = result.report

    assert report.has_issues
    assert len(report.issues) == 2
    assert report.issues[0].column == "bad"
    assert report.issues[1].column == "missing"
    assert not report.changed


def test_repair_report_handles_unhashable_row_labels() -> None:
    index = pd.Index([[1], [2]], dtype=object)
    dataset = Dataset(pd.DataFrame({"a": [np.nan, np.nan]}, index=index))

    result = apply_repair(
        dataset,
        [
            FillMissing(columns="a", strategy="constant", value=0.0),
            RemoveDuplicates(),
        ],
    )

    assert result.report.affected_rows == ([1], [2])


@pytest.mark.parametrize(
    "operations",
    [
        "not-an-operation",
        1,
        [object()],
        [FillMissing(), "not-an-operation"],
    ],
)
def test_repair_rejects_invalid_operation_inputs(operations: object) -> None:
    dataset = Dataset(pd.DataFrame({"a": [1.0, np.nan]}))

    with pytest.raises(InvalidParameterError, match="operations"):
        preview_repair(dataset, cast(Any, operations))


def test_repair_rejects_wrong_dataset_type() -> None:
    with pytest.raises(InvalidDatasetError, match="expects a Dataset"):
        preview_repair(cast(Any, pd.DataFrame({"a": [1]})), [])

    with pytest.raises(InvalidDatasetError, match="expects a Dataset"):
        apply_repair(cast(Any, pd.DataFrame({"a": [1]})), [])


def test_failing_plan_does_not_mutate_original_dataset() -> None:
    dataset = Dataset(pd.DataFrame({"a": ["1", "bad"], "b": [None, None]}))
    before = dataset.frame

    with pytest.raises(RepairConversionError):
        apply_repair(
            dataset,
            [
                FillMissing(columns="b", strategy="constant", value=0),
                ConvertTypes(conversions={"a": "int64"}),
            ],
        )

    assert_frame_equal(dataset.frame, before)
