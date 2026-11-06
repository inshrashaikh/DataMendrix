"""Tests for duplicate-row removal."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from data_mendrix import (
    ColumnNotFoundError,
    Dataset,
    InvalidDatasetError,
    InvalidParameterError,
    RepairAction,
    remove_duplicates,
)


def _duplicate_dataset() -> Dataset:
    return Dataset(
        pd.DataFrame(
            {
                "a": [1, 2, 2, 1, 2],
                "b": ["x", "y", "y", "x", "z"],
            },
            index=["r0", "r1", "r2", "r3", "r4"],
        ),
        name="duplicates",
        source="memory",
    )


def test_remove_full_row_duplicates_keep_first() -> None:
    dataset = _duplicate_dataset()
    before = dataset.frame

    result = remove_duplicates(dataset)
    operation = result.report.operations[0]

    assert result.dataset.frame.index.tolist() == ["r0", "r1", "r4"]
    assert operation.action is RepairAction.REMOVE_DUPLICATES
    assert operation.affected_rows == ("r2", "r3")
    assert operation.affected_columns == ("a", "b")
    assert operation.change_count == 2
    assert operation.row_count_delta == -2
    assert operation.changed
    assert result.dataset.name == "duplicates"
    assert result.dataset.source == "memory"
    assert_frame_equal(dataset.frame, before)


def test_remove_subset_duplicates_keep_last() -> None:
    result = remove_duplicates(_duplicate_dataset(), subset="a", keep="last")
    operation = result.report.operations[0]

    assert result.dataset.frame.index.tolist() == ["r3", "r4"]
    assert operation.affected_rows == ("r0", "r1", "r2")
    assert operation.affected_columns == ("a",)
    assert operation.change_count == 3


def test_remove_duplicates_keep_false_removes_all_occurrences() -> None:
    result = remove_duplicates(_duplicate_dataset(), subset=["a"], keep=False)
    operation = result.report.operations[0]

    assert result.dataset.frame.index.tolist() == []
    assert operation.affected_rows == ("r0", "r1", "r2", "r3", "r4")
    assert operation.change_count == 5


def test_remove_duplicates_no_changes_reports_scope_in_detail() -> None:
    dataset = Dataset(pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}))

    result = remove_duplicates(dataset, subset="a")
    operation = result.report.operations[0]

    assert result.dataset.shape == dataset.shape
    assert not operation.changed
    assert operation.affected_rows == ()
    assert operation.affected_columns == ()
    assert "subset=['a']" in operation.detail


def test_remove_duplicates_accepts_non_string_and_tuple_labels() -> None:
    numeric = Dataset(pd.DataFrame({0: [1, 1, 2], 1: ["x", "y", "z"]}))
    numeric_result = remove_duplicates(numeric, subset=0)
    assert numeric_result.dataset.shape[0] == 2

    tuple_frame = pd.DataFrame(
        {
            ("a", "b"): [1, 1, 2],
            ("c", "d"): ["x", "y", "z"],
        }
    )
    tuple_result = remove_duplicates(Dataset(tuple_frame), subset=("a", "b"))
    assert tuple_result.dataset.shape[0] == 2


def test_remove_duplicates_treats_missing_values_as_equal() -> None:
    dataset = Dataset(
        pd.DataFrame(
            {
                "a": [None, None, 1.0],
                "b": ["x", "x", "y"],
            }
        )
    )

    result = remove_duplicates(dataset)

    assert result.dataset.shape[0] == 2
    assert result.report.operations[0].affected_rows == (1,)


def test_remove_duplicates_deduplicates_subset_columns() -> None:
    result = remove_duplicates(_duplicate_dataset(), subset=["a", "a"])
    operation = result.report.operations[0]

    assert operation.affected_columns == ("a",)
    assert operation.change_count == 3


def test_remove_duplicates_empty_dataset_is_safe() -> None:
    dataset = Dataset(pd.DataFrame({"a": pd.Series([], dtype="float64")}))

    result = remove_duplicates(dataset)

    assert result.dataset.shape == (0, 1)
    assert not result.report.changed
    assert result.report.operations[0].change_count == 0


@pytest.mark.parametrize("keep", ["all", "middle", None, True, 0])
def test_remove_duplicates_rejects_invalid_keep(keep: object) -> None:
    with pytest.raises(InvalidParameterError, match="keep"):
        remove_duplicates(_duplicate_dataset(), keep=cast(Any, keep))


def test_remove_duplicates_rejects_unknown_or_empty_subset() -> None:
    dataset = _duplicate_dataset()

    with pytest.raises(ColumnNotFoundError) as exc_info:
        remove_duplicates(dataset, subset=["a", "missing"])
    assert exc_info.value.columns == ("missing",)

    with pytest.raises(InvalidParameterError, match="at least one"):
        remove_duplicates(dataset, subset=[])


def test_remove_duplicates_rejects_duplicate_column_label() -> None:
    frame = pd.DataFrame([[1, 1], [2, 2]], columns=["a", "a"])
    dataset = Dataset(frame)

    with pytest.raises(InvalidParameterError, match="not unique"):
        remove_duplicates(dataset, subset="a")


def test_remove_duplicates_rejects_wrong_dataset_type() -> None:
    with pytest.raises(InvalidDatasetError, match="expects a Dataset"):
        remove_duplicates(cast(Any, pd.DataFrame({"a": [1, 1]})))


def test_remove_duplicates_does_not_modify_source_frame() -> None:
    frame = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    dataset = Dataset(frame)
    before = frame.copy(deep=True)

    remove_duplicates(dataset)

    assert_frame_equal(frame, before)
    assert np.array_equal(dataset.frame.to_numpy(), frame.to_numpy())
