"""Tests for the Dataset container."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data_mendrix import Dataset, InvalidDatasetError, InvalidParameterError


def test_dataset_rejects_non_dataframe() -> None:
    with pytest.raises(InvalidDatasetError, match="pandas DataFrame"):
        Dataset({"a": [1, 2]})  # type: ignore[arg-type]


def test_dataset_takes_a_defensive_copy() -> None:
    source = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    dataset = Dataset(source, name="copy")

    source.loc[0, "a"] = 99
    source.loc[0, "b"] = "changed"

    assert dataset.frame["a"].tolist() == [1, 2]
    assert dataset.frame["b"].tolist() == ["x", "y"]


def test_frame_property_returns_a_copy() -> None:
    dataset = Dataset(pd.DataFrame({"a": [1, 2]}))
    exported = dataset.frame
    exported.loc[0, "a"] = 99

    assert dataset.frame["a"].tolist() == [1, 2]


def test_dataset_metadata_and_shape() -> None:
    dataset = Dataset(
        pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]}),
        name="people",
        source="data/people.csv",
    )

    assert dataset.name == "people"
    assert dataset.source == "data/people.csv"
    assert dataset.shape == (3, 2)
    assert dataset.row_count == 3
    assert dataset.column_count == 2
    assert len(dataset) == 3
    assert list(dataset.columns) == ["a", "b"]
    assert list(dataset.dtypes.index) == ["a", "b"]
    assert "people" in repr(dataset)


def test_dataset_accepts_pathlike_source() -> None:
    source = Path("data") / "people.csv"

    dataset = Dataset(pd.DataFrame({"a": [1]}), source=source)

    assert dataset.source == str(source)


def test_dataset_is_immutable() -> None:
    dataset = Dataset(pd.DataFrame({"a": [1]}), name="original")

    with pytest.raises(AttributeError, match="immutable"):
        dataset.name = "changed"  # type: ignore[misc]

    with pytest.raises(AttributeError, match="immutable"):
        dataset._frame = pd.DataFrame({"a": [2]})  # type: ignore[attr-defined]


def test_empty_frame_is_supported() -> None:
    dataset = Dataset(pd.DataFrame({"a": pd.Series([], dtype="float64")}))

    assert dataset.empty
    assert dataset.shape == (0, 1)
    assert len(dataset) == 0


def test_dataset_rejects_blank_name() -> None:
    with pytest.raises(InvalidParameterError, match="name"):
        Dataset(pd.DataFrame({"a": [1]}), name="  ")


def test_dataset_rejects_invalid_source_type() -> None:
    with pytest.raises(InvalidParameterError, match="source"):
        Dataset(pd.DataFrame({"a": [1]}), source=123)  # type: ignore[arg-type]
