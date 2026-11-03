"""Tests for duplicate-row detection."""

from __future__ import annotations

from typing import Any, cast

import pandas as pd
import pytest

from data_mendrix import (
    ColumnNotFoundError,
    Dataset,
    InvalidDatasetError,
    InvalidParameterError,
    detect_duplicates,
)


def _duplicate_dataset() -> Dataset:
    return Dataset(
        pd.DataFrame(
            {
                "a": [1, 2, 2, 1, 2],
                "b": ["x", "y", "y", "x", "z"],
            }
        )
    )


def test_full_row_duplicates_are_detected() -> None:
    report = detect_duplicates(_duplicate_dataset())

    assert report.row_count == 5
    assert report.unique_row_count == 3
    assert report.duplicate_row_count == 2
    assert report.duplicate_group_count == 2
    assert report.duplicate_indices == (2, 3)
    assert report.has_duplicates
    assert report.duplicate_fraction == pytest.approx(0.4)


def test_subset_duplicates_are_detected() -> None:
    report = detect_duplicates(_duplicate_dataset(), subset=["a"])

    assert report.duplicate_row_count == 3
    assert report.unique_row_count == 2
    assert report.duplicate_group_count == 2
    assert report.duplicate_indices == (2, 3, 4)


def test_subset_accepts_single_column_label() -> None:
    report = detect_duplicates(_duplicate_dataset(), subset="a")

    assert report.duplicate_row_count == 3


def test_subset_accepts_non_string_column_label() -> None:
    dataset = Dataset(pd.DataFrame({0: [1, 1, 2], 1: ["x", "y", "z"]}))

    report = detect_duplicates(dataset, subset=0)

    assert report.duplicate_row_count == 1
    assert report.duplicate_indices == (1,)


def test_subset_accepts_tuple_column_label() -> None:
    dataset = Dataset(pd.DataFrame({("a", "b"): [1, 1, 2], ("c", "d"): ["x", "y", "z"]}))

    report = detect_duplicates(dataset, subset=("a", "b"))

    assert report.duplicate_row_count == 1
    assert report.duplicate_indices == (1,)


def test_subset_tuple_without_matching_label_is_iterable() -> None:
    report = detect_duplicates(_duplicate_dataset(), subset=("a", "b"))

    assert report.duplicate_row_count == 2


def test_keep_last_marks_earlier_occurrences() -> None:
    report = detect_duplicates(_duplicate_dataset(), keep="last")

    assert report.duplicate_row_count == 2
    assert report.duplicate_indices == (0, 1)
    assert report.unique_row_count == 3


def test_missing_values_are_treated_as_equal() -> None:
    dataset = Dataset(
        pd.DataFrame(
            {
                "a": [None, None, 1.0],
                "b": ["x", "x", "y"],
            }
        )
    )

    report = detect_duplicates(dataset)

    assert report.duplicate_row_count == 1
    assert report.duplicate_indices == (1,)
    assert report.duplicate_group_count == 1


def test_no_duplicates_reports_zero() -> None:
    report = detect_duplicates(Dataset(pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})))

    assert not report.has_duplicates
    assert report.duplicate_row_count == 0
    assert report.duplicate_group_count == 0
    assert report.duplicate_fraction == 0.0
    assert report.duplicate_indices == ()


def test_empty_dataset_has_zero_duplicate_fraction() -> None:
    dataset = Dataset(pd.DataFrame({"a": pd.Series([], dtype="float64")}))

    report = detect_duplicates(dataset)

    assert report.row_count == 0
    assert report.duplicate_row_count == 0
    assert report.duplicate_fraction == 0.0


def test_unknown_subset_column_raises() -> None:
    with pytest.raises(ColumnNotFoundError) as exc_info:
        detect_duplicates(_duplicate_dataset(), subset=["a", "missing"])

    assert exc_info.value.columns == ("missing",)


def test_empty_subset_raises() -> None:
    with pytest.raises(InvalidParameterError, match="at least one"):
        detect_duplicates(_duplicate_dataset(), subset=[])


@pytest.mark.parametrize("keep", ["all", "middle", None])
def test_invalid_keep_policy_raises(keep: Any) -> None:
    with pytest.raises(InvalidParameterError, match="keep"):
        detect_duplicates(_duplicate_dataset(), keep=keep)


def test_duplicate_detection_rejects_wrong_dataset_type() -> None:
    with pytest.raises(InvalidDatasetError, match="expects a Dataset"):
        detect_duplicates(cast(Any, pd.DataFrame({"a": [1]})))
