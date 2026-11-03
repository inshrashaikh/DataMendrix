"""Tests for the CSV loader."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import pytest

from data_mendrix import (
    Dataset,
    DatasetLoadError,
    InvalidParameterError,
    UnsupportedFileTypeError,
    load_csv,
)


def test_load_csv_reads_values_and_metadata(tmp_path: Path) -> None:
    path = tmp_path / "people.csv"
    path.write_text("id,name,score\n1,alpha,10.5\n2,beta,\n", encoding="utf-8")

    dataset = load_csv(path)

    assert isinstance(dataset, Dataset)
    assert dataset.name == "people"
    assert dataset.source == str(path)
    assert dataset.shape == (2, 3)
    assert list(dataset.columns) == ["id", "name", "score"]
    assert dataset.frame["id"].tolist() == [1, 2]
    assert dataset.frame["name"].tolist() == ["alpha", "beta"]
    assert dataset.frame["score"].iloc[0] == pytest.approx(10.5)
    assert pd.isna(dataset.frame["score"].iloc[1])


def test_load_csv_accepts_path_string(tmp_path: Path) -> None:
    path = tmp_path / "simple.csv"
    path.write_text("a\n1\n", encoding="utf-8")

    dataset = load_csv(str(path))

    assert dataset.shape == (1, 1)
    assert dataset.source == str(path)


def test_load_csv_accepts_text_stream() -> None:
    stream = io.StringIO("a,b\n1,2\n")

    dataset = load_csv(stream)

    assert dataset.name == "dataset"
    assert dataset.source is None
    assert dataset.shape == (1, 2)
    assert not stream.closed


def test_load_csv_supports_separator_and_na_values(tmp_path: Path) -> None:
    path = tmp_path / "scores.csv"
    path.write_text("id;score\n1;10\n2;NA\n3;30\n", encoding="utf-8")

    dataset = load_csv(path, separator=";", na_values=["NA"])

    assert dataset.shape == (3, 2)
    assert dataset.frame["score"].isna().tolist() == [False, True, False]


def test_load_csv_supports_non_utf8_encoding(tmp_path: Path) -> None:
    path = tmp_path / "cafe.csv"
    path.write_text("name\ncafé\n", encoding="latin-1")

    dataset = load_csv(path, encoding="latin-1")

    assert dataset.frame["name"].tolist() == ["café"]


def test_load_csv_named_stream_uses_file_stem(tmp_path: Path) -> None:
    path = tmp_path / "streamed.csv"
    path.write_text("a\n1\n", encoding="utf-8")

    with path.open(encoding="utf-8") as stream:
        dataset = load_csv(stream)
        assert not stream.closed

    assert dataset.name == "streamed"
    assert dataset.source == str(path)


def test_load_csv_na_values_mapping_targets_columns() -> None:
    stream = io.StringIO("a,b\nNA,NA\n")

    dataset = load_csv(
        stream,
        na_values={"a": ["NA"]},
        keep_default_na=False,
    )

    assert dataset.frame["a"].isna().all()
    assert not dataset.frame["b"].isna().any()


def test_load_csv_supports_thousands_and_decimal_options() -> None:
    stream = io.StringIO("value\n1.234,5\n")

    dataset = load_csv(stream, separator=";", thousands=".", decimal=",")

    assert dataset.frame["value"].iloc[0] == pytest.approx(1234.5)


def test_load_csv_allows_header_only_file(tmp_path: Path) -> None:
    path = tmp_path / "empty_rows.csv"
    path.write_text("a,b\n", encoding="utf-8")

    dataset = load_csv(path)

    assert dataset.shape == (0, 2)


def test_load_csv_does_not_modify_source_file(tmp_path: Path) -> None:
    path = tmp_path / "stable.csv"
    path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    before = path.read_bytes()

    load_csv(path)

    assert path.read_bytes() == before


def test_load_csv_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(DatasetLoadError, match="not found"):
        load_csv(tmp_path / "missing.csv")


def test_load_csv_directory_raises(tmp_path: Path) -> None:
    directory = tmp_path / "directory.csv"
    directory.mkdir()

    with pytest.raises(DatasetLoadError, match="not a file"):
        load_csv(directory)


def test_load_csv_rejects_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "data.txt"
    path.write_text("a\n1\n", encoding="utf-8")

    with pytest.raises(UnsupportedFileTypeError, match=r"only '\.csv'"):
        load_csv(path)


def test_load_csv_accepts_uppercase_extension(tmp_path: Path) -> None:
    path = tmp_path / "DATA.CSV"
    path.write_text("a\n1\n", encoding="utf-8")

    assert load_csv(path).shape == (1, 1)


def test_load_csv_empty_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "zero_bytes.csv"
    path.write_bytes(b"")

    with pytest.raises(DatasetLoadError, match="contains no data"):
        load_csv(path)


def test_load_csv_malformed_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "malformed.csv"
    path.write_text('a,b\n"unterminated,2\n', encoding="utf-8")

    with pytest.raises(DatasetLoadError, match="Could not parse CSV"):
        load_csv(path)


def test_load_csv_missing_encoding_fails() -> None:
    stream = io.StringIO("a\n1\n")

    with pytest.raises(InvalidParameterError, match="encoding"):
        load_csv(stream, encoding="")


def test_load_csv_invalid_option_raises_before_reading() -> None:
    stream = io.StringIO("a,b\n1,2\n")

    with pytest.raises(InvalidParameterError, match="separator"):
        load_csv(stream, separator="::")


@pytest.mark.parametrize(
    "options",
    [
        {"decimal": ""},
        {"comment": ".."},
        {"thousands": ".."},
        {"keep_default_na": "yes"},
        {"skipinitialspace": 1},
    ],
)
def test_load_csv_rejects_invalid_option_values(options: dict[str, object]) -> None:
    stream = io.StringIO("a,b\n1,2\n")

    with pytest.raises(InvalidParameterError):
        load_csv(stream, **options)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "na_values",
    [
        [1],
        {1: ["NA"]},
        {"a": [1]},
    ],
)
def test_load_csv_rejects_invalid_na_values(na_values: object) -> None:
    stream = io.StringIO("a,b\n1,2\n")

    with pytest.raises(InvalidParameterError, match="na_values"):
        load_csv(stream, na_values=na_values)  # type: ignore[arg-type]


def test_load_csv_undecodable_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "invalid_utf8.csv"
    path.write_bytes(b"name\n\xff\n")

    with pytest.raises(DatasetLoadError, match="Could not decode"):
        load_csv(path)


def test_load_csv_invalid_source_type_raises() -> None:
    with pytest.raises(InvalidParameterError, match="CSV source"):
        load_csv(123)  # type: ignore[arg-type]


def test_load_csv_source_error_chains_cause(tmp_path: Path) -> None:
    path = tmp_path / "broken.csv"
    path.write_text('a,b\n"unterminated,2\n', encoding="utf-8")

    with pytest.raises(DatasetLoadError) as exc_info:
        load_csv(path)

    assert exc_info.value.__cause__ is not None
