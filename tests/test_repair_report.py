"""Regression tests for repair-report row-union semantics."""

from __future__ import annotations

import numpy as np
import pandas as pd

import data_mendrix as dm
from data_mendrix import Dataset, apply_repair, remove_duplicates


def test_affected_rows_deduplicates_hashable_labels_in_plan_order() -> None:
    frame = pd.DataFrame({"a": [np.nan, np.nan, 9.0]}, index=[1, 1, 2])
    dataset = Dataset(frame, name="report")

    result = apply_repair(
        dataset,
        [
            # Fill affects rows 0 and 1 (label ``1`` twice); the follow-up
            # duplicate removal affects label ``1`` again.
            dm.FillMissing(columns="a", strategy="constant", value=0.0),
            dm.RemoveDuplicates(),
        ],
    )

    assert result.report.affected_rows == (1,)


def test_affected_rows_deduplicates_missing_labels() -> None:
    frame = pd.DataFrame({"a": [1, 1, 2]}, index=[np.nan, np.nan, 2])
    dataset = Dataset(frame, name="report")

    result = remove_duplicates(dataset)

    assert len(result.report.affected_rows) == 1
    assert np.isnan(result.report.affected_rows[0])


def test_affected_rows_deduplicates_unhashable_labels_across_operations() -> None:
    frame = pd.DataFrame(
        {"a": [np.nan, np.nan, 9.0]},
        index=pd.Index([[1], [1], [2]], dtype=object),
    )
    dataset = Dataset(frame, name="report")

    result = apply_repair(
        dataset,
        [
            dm.FillMissing(columns="a", strategy="constant", value=0.0),
            dm.RemoveDuplicates(),
        ],
    )

    assert result.report.affected_rows == ([1],)
