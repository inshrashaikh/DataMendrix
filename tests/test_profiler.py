"""Tests for the combined dataset profiler."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from data_mendrix import (
    ColumnNotFoundError,
    Dataset,
    InvalidDatasetError,
    detect_duplicates,
    missing_value_report,
    numeric_statistics,
    profile_dataset,
)


def test_profile_combines_all_sections(sample_dataset: Dataset) -> None:
    profile = profile_dataset(sample_dataset)

    assert profile.name == "sample"
    assert profile.source is None
    assert profile.row_count == 4
    assert profile.column_count == 5
    assert profile.memory_usage_bytes > 0
    assert profile.numeric_column_names == ("id", "score")
    assert profile.numeric == numeric_statistics(sample_dataset)
    assert profile.missing == missing_value_report(sample_dataset)
    assert profile.duplicates == detect_duplicates(sample_dataset)


def test_profile_column_details(sample_dataset: Dataset) -> None:
    profile = profile_dataset(sample_dataset)
    frame = sample_dataset.frame

    assert [column.name for column in profile.columns] == [
        "id",
        "name",
        "score",
        "active",
        "category",
    ]
    score = profile.column("score")
    assert score.dtype == str(frame["score"].dtype)
    assert score.missing_count == 1
    assert score.non_missing_count == 3
    assert score.missing_fraction == pytest.approx(0.25)
    assert score.unique_count == 2


def test_profile_reports_missing_cells(sample_dataset: Dataset) -> None:
    profile = profile_dataset(sample_dataset)

    assert profile.missing.missing_cell_count == 2
    assert profile.missing.missing_cell_fraction == pytest.approx(0.1)
    assert profile.missing.rows_with_missing == 2
    assert profile.missing.complete_row_count == 2


def test_profile_does_not_modify_dataset(sample_dataset: Dataset) -> None:
    before = sample_dataset.frame

    profile_dataset(sample_dataset)

    pd.testing.assert_frame_equal(sample_dataset.frame, before)


def test_empty_dataset_can_be_profiled() -> None:
    dataset = Dataset(
        pd.DataFrame({"a": pd.Series([], dtype="float64")}),
        name="empty",
    )

    profile = profile_dataset(dataset)

    assert profile.row_count == 0
    assert profile.column_count == 1
    assert profile.missing.missing_cell_fraction == 0.0
    assert profile.duplicates.duplicate_fraction == 0.0
    assert profile.numeric_column_names == ("a",)
    assert profile.numeric.column("a").count == 0


def test_profile_preserves_source() -> None:
    dataset = Dataset(pd.DataFrame({"a": [1]}), source="data/file.csv")

    profile = profile_dataset(dataset)

    assert profile.source == "data/file.csv"


def test_profile_unknown_column_raises() -> None:
    profile = profile_dataset(Dataset(pd.DataFrame({"a": [1]})))

    with pytest.raises(ColumnNotFoundError) as exc_info:
        profile.column("missing")

    assert exc_info.value.columns == ("missing",)


def test_profile_column_lookup_accepts_non_string_label() -> None:
    profile = profile_dataset(Dataset(pd.DataFrame({0: [1, 2]})))

    assert profile.column(0).name == "0"


def test_profile_handles_unhashable_object_cells() -> None:
    dataset = Dataset(pd.DataFrame({"items": [[1, 2], [1, 2], [3]]}))

    profile = profile_dataset(dataset)

    assert profile.column("items").unique_count == 2
    assert profile.duplicates.duplicate_row_count == 1


def test_profile_handles_unhashable_array_cells() -> None:
    dataset = Dataset(
        pd.DataFrame(
            {
                "items": [
                    np.array([1, 2]),
                    np.array([1, 2]),
                    np.array([3]),
                ]
            }
        )
    )

    profile = profile_dataset(dataset)

    assert profile.column("items").unique_count == 2


def test_profile_rejects_wrong_dataset_type() -> None:
    with pytest.raises(InvalidDatasetError, match="expects a Dataset"):
        profile_dataset(cast(Any, pd.DataFrame({"a": [1]})))


def test_profile_handles_all_missing_dataset() -> None:
    dataset = Dataset(
        pd.DataFrame(
            {
                "x": [np.nan, np.nan],
                "label": [None, None],
            }
        )
    )

    profile = profile_dataset(dataset)

    assert profile.missing.missing_cell_fraction == pytest.approx(1.0)
    assert profile.numeric.column("x").count == 0
    assert profile.column("label").unique_count == 0
