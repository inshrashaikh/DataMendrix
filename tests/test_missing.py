"""Tests for missing-value statistics."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from data_mendrix import (
    ColumnNotFoundError,
    Dataset,
    InvalidDatasetError,
    missing_value_report,
)


def test_missing_statistics_are_correct() -> None:
    frame = pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0, np.nan],
            "b": ["x", "y", "x", "z"],
            "c": [np.nan, 2.0, 3.0, 4.0],
        }
    )

    report = missing_value_report(Dataset(frame))

    assert report.row_count == 4
    assert report.column_count == 3
    assert report.total_cell_count == 12
    assert report.missing_cell_count == 3
    assert report.missing_cell_fraction == pytest.approx(0.25)
    assert report.rows_with_missing == 3
    assert report.complete_row_count == 1
    assert not report.is_complete


def test_per_column_missing_statistics_are_correct() -> None:
    frame = pd.DataFrame(
        {
            "a": [1.0, np.nan, 3.0, np.nan],
            "b": ["x", "y", "x", "z"],
            "c": [np.nan, 2.0, 3.0, 4.0],
        }
    )

    report = missing_value_report(Dataset(frame))

    a_stats, b_stats, c_stats = report.columns
    assert a_stats.column == "a"
    assert a_stats.missing_count == 2
    assert a_stats.non_missing_count == 2
    assert a_stats.missing_fraction == pytest.approx(0.5)
    assert a_stats.total_count == 4
    assert a_stats.has_missing

    assert b_stats.missing_count == 0
    assert not b_stats.has_missing

    assert c_stats.missing_count == 1
    assert c_stats.missing_fraction == pytest.approx(0.25)

    assert report.columns_with_missing == (a_stats, c_stats)
    assert report.column("a") == a_stats


def test_complete_dataset_reports_no_missing_cells() -> None:
    report = missing_value_report(Dataset(pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})))

    assert report.is_complete
    assert report.missing_cell_count == 0
    assert report.missing_cell_fraction == 0.0
    assert report.rows_with_missing == 0
    assert report.complete_row_count == 2
    assert report.columns_with_missing == ()


def test_empty_dataset_has_zero_fractions() -> None:
    report = missing_value_report(Dataset(pd.DataFrame({"a": pd.Series([], dtype="float64")})))

    assert report.row_count == 0
    assert report.total_cell_count == 0
    assert report.missing_cell_fraction == 0.0
    assert report.complete_row_count == 0
    assert report.columns[0].missing_fraction == 0.0


def test_zero_column_dataset_is_supported() -> None:
    report = missing_value_report(Dataset(pd.DataFrame(index=range(3))))

    assert report.row_count == 3
    assert report.column_count == 0
    assert report.total_cell_count == 0
    assert report.missing_cell_fraction == 0.0
    assert report.rows_with_missing == 0
    assert report.complete_row_count == 3
    assert report.columns == ()


def test_missing_report_rejects_wrong_dataset_type() -> None:
    with pytest.raises(InvalidDatasetError, match="expects a Dataset"):
        missing_value_report(cast(Any, pd.DataFrame({"a": [1]})))


def test_missing_report_unknown_column_raises() -> None:
    report = missing_value_report(Dataset(pd.DataFrame({"a": [1]})))

    with pytest.raises(ColumnNotFoundError) as exc_info:
        report.column("b")

    assert exc_info.value.columns == ("b",)


def test_column_lookup_accepts_non_string_label() -> None:
    report = missing_value_report(Dataset(pd.DataFrame({0: [1.0, np.nan]})))

    stats = report.column(0)

    assert stats.column == "0"
    assert stats.missing_count == 1
