"""Tests for missing-value repair operations."""

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
    drop_missing,
    fill_missing,
)


def _missing_dataset() -> Dataset:
    return Dataset(
        pd.DataFrame(
            {
                "a": [1.0, np.nan, 3.0, np.nan],
                "b": [np.nan, "x", "y", np.nan],
            },
            index=["r0", "r1", "r2", "r3"],
        ),
        name="missing",
        source="memory",
    )


def test_drop_missing_rows_with_selected_columns() -> None:
    dataset = _missing_dataset()
    before = dataset.frame

    result = drop_missing(dataset, columns="a")
    operation = result.report.operations[0]

    assert result.dataset.frame.index.tolist() == ["r0", "r2"]
    assert operation.action is RepairAction.DROP_MISSING_ROWS
    assert operation.affected_rows == ("r1", "r3")
    assert operation.affected_columns == ("a",)
    assert operation.change_count == 2
    assert operation.row_count_delta == -2
    assert operation.column_count_delta == 0
    assert operation.affected_row_count == 2
    assert operation.affected_column_count == 1
    assert operation.changed
    assert result.report.changed
    assert result.report.row_count_before == 4
    assert result.report.row_count_after == 2
    assert result.report.row_count_delta == -2
    assert result.report.column_count_delta == 0
    assert result.dataset.name == "missing"
    assert result.dataset.source == "memory"
    assert result.dataset is not dataset
    assert_frame_equal(dataset.frame, before)


def test_drop_missing_rows_how_any_and_all() -> None:
    dataset = Dataset(
        pd.DataFrame(
            {
                "a": [np.nan, np.nan, 1.0],
                "b": [np.nan, "x", np.nan],
            },
            index=["both", "partial", "one"],
        )
    )

    any_result = drop_missing(dataset, columns=["a", "b"], how="any")
    all_result = drop_missing(dataset, columns=["a", "b"], how="all")

    assert any_result.dataset.frame.index.tolist() == []
    assert any_result.report.operations[0].affected_rows == ("both", "partial", "one")
    assert all_result.dataset.frame.index.tolist() == ["partial", "one"]
    assert all_result.report.operations[0].affected_rows == ("both",)


def test_drop_missing_columns_how_any_and_all() -> None:
    dataset = Dataset(
        pd.DataFrame(
            {
                "all_missing": [np.nan, np.nan, np.nan],
                "partly_missing": [np.nan, 1.0, np.nan],
                "complete": [1.0, 2.0, 3.0],
            }
        )
    )

    any_result = drop_missing(dataset, axis="columns", how="any")
    all_result = drop_missing(dataset, axis="columns", how="all")

    assert any_result.dataset.frame.columns.tolist() == ["complete"]
    assert any_result.report.operations[0].affected_columns == ("all_missing", "partly_missing")
    assert any_result.report.operations[0].change_count == 2
    assert all_result.dataset.frame.columns.tolist() == ["partly_missing", "complete"]
    assert all_result.report.operations[0].affected_columns == ("all_missing",)
    assert all_result.report.operations[0].change_count == 1


def test_drop_missing_columns_deduplicates_selected_columns() -> None:
    dataset = Dataset(pd.DataFrame({"a": [np.nan, np.nan], "b": [1, 2]}))

    result = drop_missing(dataset, axis="columns", columns=["a", "a"])

    assert result.dataset.columns.tolist() == ["b"]
    assert result.report.operations[0].change_count == 1
    assert result.report.operations[0].affected_columns == ("a",)


def test_drop_missing_columns_keeps_unselected_missing_columns() -> None:
    dataset = Dataset(
        pd.DataFrame(
            {
                "selected_missing": [np.nan, np.nan],
                "unselected_missing": [np.nan, np.nan],
                "complete": [1, 2],
            }
        )
    )

    result = drop_missing(
        dataset,
        axis="columns",
        columns=["selected_missing", "complete"],
    )

    assert result.dataset.frame.columns.tolist() == ["unselected_missing", "complete"]
    assert result.report.operations[0].affected_columns == ("selected_missing",)


def test_drop_missing_empty_dataset_is_safe() -> None:
    dataset = Dataset(
        pd.DataFrame(
            {
                "a": pd.Series([], dtype="float64"),
                "b": pd.Series([], dtype="float64"),
            }
        )
    )

    result = drop_missing(dataset)

    assert result.dataset.shape == (0, 2)
    assert result.report.operations[0].change_count == 0
    assert not result.report.changed
    assert result.report.affected_rows == ()


def test_fill_missing_deduplicates_selected_columns() -> None:
    dataset = Dataset(pd.DataFrame({"a": [np.nan, np.nan]}))

    result = fill_missing(dataset, columns=["a", "a"], strategy="mode")

    assert len(result.report.issues) == 1


def test_drop_missing_zero_column_dataset_is_safe() -> None:
    dataset = Dataset(pd.DataFrame(index=[0, 1, 2]))

    rows = drop_missing(dataset)
    columns = drop_missing(dataset, axis="columns")

    assert rows.dataset.shape == (3, 0)
    assert columns.dataset.shape == (3, 0)
    assert not rows.report.changed
    assert not columns.report.changed


def test_drop_missing_rows_all_drops_vacuously_matching_rows() -> None:
    dataset = Dataset(pd.DataFrame(index=["r0", "r1"]))

    result = drop_missing(dataset, how="all")

    assert result.dataset.shape == (0, 0)
    assert result.report.operations[0].affected_rows == ("r0", "r1")
    assert result.report.operations[0].change_count == 2


@pytest.mark.parametrize(
    ("strategy", "expected"),
    [
        ("mean", (1.0 + 2.0 + 100.0) / 3.0),
        ("median", 2.0),
    ],
)
def test_fill_mean_and_median_use_finite_values(strategy: str, expected: float) -> None:
    dataset = Dataset(
        pd.DataFrame(
            {
                "score": [1.0, np.nan, 2.0, 100.0, np.inf],
            },
            index=["r0", "r1", "r2", "r3", "r4"],
        )
    )

    result = fill_missing(dataset, columns="score", strategy=cast(Any, strategy))
    operation = result.report.operations[0]
    change = operation.column_changes[0]

    assert result.dataset.frame.loc["r1", "score"] == pytest.approx(expected)
    assert result.dataset.frame.loc["r4", "score"] == np.inf
    assert operation.affected_rows == ("r1",)
    assert operation.affected_columns == ("score",)
    assert operation.change_count == 1
    assert change.column == "score"
    assert change.change_count == 1
    assert change.fill_value == pytest.approx(expected)
    assert change.before_dtype == "float64"
    assert change.after_dtype == "float64"


def test_fill_mode_uses_first_appearance_for_ties() -> None:
    dataset = Dataset(pd.DataFrame({"label": ["b", "a", "b", "a", None]}))

    result = fill_missing(dataset, columns="label", strategy="mode")

    assert result.dataset.frame["label"].tolist() == ["b", "a", "b", "a", "b"]
    assert result.report.operations[0].column_changes[0].fill_value == "b"


def test_fill_constant_only_selected_columns() -> None:
    dataset = Dataset(pd.DataFrame({"a": [1.0, np.nan], "b": [np.nan, 2.0]}))

    result = fill_missing(dataset, columns="a", strategy="constant", value=0)
    frame = result.dataset.frame

    assert frame["a"].tolist() == [1.0, 0.0]
    assert np.isnan(frame.loc[0, "b"])
    assert result.report.operations[0].affected_columns == ("a",)


def test_fill_default_mean_only_selects_numeric_columns() -> None:
    dataset = Dataset(
        pd.DataFrame(
            {
                "number": [1.0, np.nan, 3.0],
                "text": ["a", None, "b"],
            }
        )
    )

    result = fill_missing(dataset)
    frame = result.dataset.frame

    assert frame["number"].tolist() == [1.0, 2.0, 3.0]
    assert frame.loc[1, "text"] is None or pd.isna(frame.loc[1, "text"])
    assert result.report.operations[0].affected_columns == ("number",)


def test_fill_explicit_nonnumeric_mean_raises() -> None:
    dataset = Dataset(pd.DataFrame({"text": ["a", None]}))

    with pytest.raises(InvalidParameterError, match="numeric"):
        fill_missing(dataset, columns="text", strategy="mean")


@pytest.mark.parametrize("strategy", ["mean", "median", "mode"])
def test_fill_all_missing_reports_issue(strategy: str) -> None:
    dataset = Dataset(pd.DataFrame({"a": pd.Series([pd.NA, pd.NA], dtype="Float64")}))

    result = fill_missing(dataset, columns="a", strategy=cast(Any, strategy))
    operation = result.report.operations[0]

    assert not operation.changed
    assert operation.issues
    assert result.report.has_issues
    assert len(result.report.issues) == 1
    assert result.dataset.frame["a"].isna().all()


def test_fill_mean_with_only_infinite_values_reports_issue() -> None:
    dataset = Dataset(pd.DataFrame({"a": [np.inf, -np.inf, np.nan]}))

    result = fill_missing(dataset, columns="a", strategy="mean")

    assert result.report.has_issues
    assert "no finite" in result.report.issues[0].message
    assert not result.report.changed


def test_fill_mean_nonfinite_overflow_reports_issue() -> None:
    dataset = Dataset(pd.DataFrame({"a": [1e308, 1e308, np.nan]}))

    with np.errstate(over="ignore"):
        result = fill_missing(dataset, columns="a", strategy="mean")

    assert result.report.has_issues
    assert "non-finite fill value" in result.report.issues[0].message


def test_fill_incompatible_constant_reports_issue() -> None:
    dataset = Dataset(
        pd.DataFrame({"category": pd.Categorical(["a", None, "b"])}),
    )

    result = fill_missing(dataset, columns="category", strategy="constant", value="c")
    operation = result.report.operations[0]

    assert not operation.changed
    assert operation.issues[0].column == "category"
    assert operation.issues[0].error_type == "TypeError"
    assert result.dataset.frame["category"].isna().sum() == 1


def test_fill_mode_with_unhashable_values_reports_issue() -> None:
    dataset = Dataset(
        pd.DataFrame({"items": pd.Series([[1], [1], None], dtype=object)}),
    )

    result = fill_missing(dataset, columns="items", strategy="mode")

    assert result.report.has_issues
    assert not result.report.changed
    assert result.dataset.frame["items"].isna().sum() == 1


def test_fill_mode_falls_back_when_value_counts_rejects_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_value_counts(*args: object, **kwargs: object) -> None:
        raise TypeError("unhashable values")

    monkeypatch.setattr(pd.Series, "value_counts", reject_value_counts)
    dataset = Dataset(
        pd.DataFrame(
            {"items": pd.Series([np.array([1]), np.array([1]), None], dtype=object)},
        ),
    )

    result = fill_missing(dataset, columns="items", strategy="mode")

    assert result.report.has_issues
    assert not result.report.changed
    assert result.dataset.frame["items"].isna().sum() == 1


@pytest.mark.parametrize("value", [None, np.nan, pd.NA, pd.NaT, [1]])
def test_fill_constant_rejects_missing_or_non_scalar_values(value: object) -> None:
    dataset = Dataset(pd.DataFrame({"a": [1.0, np.nan]}))

    with pytest.raises(InvalidParameterError, match="non-missing scalar"):
        fill_missing(dataset, columns="a", strategy="constant", value=value)


def test_fill_rejects_value_for_non_constant_strategy() -> None:
    dataset = Dataset(pd.DataFrame({"a": [1.0, np.nan]}))

    with pytest.raises(InvalidParameterError, match="only valid"):
        fill_missing(dataset, columns="a", strategy="mean", value=0)


def test_fill_affected_rows_are_union_across_columns() -> None:
    dataset = Dataset(
        pd.DataFrame(
            {
                "a": [np.nan, 1.0, np.nan],
                "b": [1.0, np.nan, np.nan],
            },
            index=["r0", "r1", "r2"],
        )
    )

    result = fill_missing(dataset, columns=["a", "b"], strategy="constant", value=0)
    operation = result.report.operations[0]

    assert operation.affected_rows == ("r0", "r1", "r2")
    assert operation.affected_columns == ("a", "b")
    assert operation.change_count == 4


def test_missing_repair_rejects_unknown_or_empty_columns() -> None:
    dataset = Dataset(pd.DataFrame({"a": [1.0, np.nan]}))

    with pytest.raises(ColumnNotFoundError) as exc_info:
        drop_missing(dataset, columns=["missing"])
    assert exc_info.value.columns == ("missing",)

    with pytest.raises(InvalidParameterError, match="at least one"):
        fill_missing(dataset, columns=[])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"axis": "cells"}, "axis"),
        ({"how": "some"}, "how"),
    ],
)
def test_drop_missing_rejects_invalid_options(kwargs: dict[str, Any], message: str) -> None:
    dataset = Dataset(pd.DataFrame({"a": [1.0, 2.0]}))

    with pytest.raises(InvalidParameterError, match=message):
        drop_missing(dataset, **kwargs)


def test_fill_rejects_invalid_strategy() -> None:
    dataset = Dataset(pd.DataFrame({"a": [1.0, np.nan]}))

    with pytest.raises(InvalidParameterError, match="strategy"):
        fill_missing(dataset, strategy=cast(Any, "quantile"))


def test_missing_repair_rejects_wrong_dataset_type() -> None:
    with pytest.raises(InvalidDatasetError, match="expects a Dataset"):
        drop_missing(cast(Any, pd.DataFrame({"a": [1.0]})))
