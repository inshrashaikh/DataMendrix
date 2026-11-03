"""Tests for structural dataset validation."""

from __future__ import annotations

from typing import Any, cast

import pandas as pd
import pytest

from data_mendrix import (
    Dataset,
    DatasetValidationError,
    InvalidDatasetError,
    InvalidParameterError,
    ValidationCode,
    ValidationConfig,
    validate_dataset,
)


def _dataset(frame: pd.DataFrame) -> Dataset:
    return Dataset(frame)


def test_valid_dataset_has_empty_report() -> None:
    report = validate_dataset(_dataset(pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})))

    assert report.is_valid
    assert report.issues == ()
    assert report.codes == ()


def test_minimum_row_rule_is_reported() -> None:
    report = validate_dataset(_dataset(pd.DataFrame({"a": pd.Series([], dtype=float)})))

    assert not report.is_valid
    assert report.codes == (ValidationCode.TOO_FEW_ROWS,)
    assert "at least 1" in report.issues[0].message


def test_min_rows_can_be_relaxed() -> None:
    report = validate_dataset(
        _dataset(pd.DataFrame({"a": pd.Series([], dtype=float)})),
        ValidationConfig(min_rows=0),
    )

    assert report.is_valid


def test_minimum_column_rule_is_reported() -> None:
    report = validate_dataset(_dataset(pd.DataFrame(index=[0, 1])))

    assert report.codes == (ValidationCode.TOO_FEW_COLUMNS,)


def test_duplicate_column_names_are_reported() -> None:
    frame = pd.DataFrame([[1, 2]], columns=["a", "a"])

    report = validate_dataset(_dataset(frame))

    assert report.codes == (ValidationCode.DUPLICATE_COLUMNS,)
    assert report.issues[0].column == "a"


def test_blank_column_name_is_reported() -> None:
    frame = pd.DataFrame([[1, 2]], columns=["a", " "])

    report = validate_dataset(_dataset(frame))

    assert ValidationCode.BLANK_COLUMN_NAME in report.codes
    assert report.issues[0].column == " "


def test_unnamed_columns_are_optional() -> None:
    dataset = _dataset(pd.DataFrame([[1]], columns=["Unnamed: 0"]))

    assert validate_dataset(dataset).is_valid
    report = validate_dataset(
        dataset,
        ValidationConfig(forbid_unnamed_columns=True),
    )

    assert report.codes == (ValidationCode.UNNAMED_COLUMN,)
    assert report.issues[0].column == "Unnamed: 0"


def test_required_columns_are_reported() -> None:
    report = validate_dataset(
        _dataset(pd.DataFrame({"a": [1]})),
        ValidationConfig(required_columns=("a", "b", "c")),
    )

    assert report.codes == (
        ValidationCode.MISSING_REQUIRED_COLUMN,
        ValidationCode.MISSING_REQUIRED_COLUMN,
    )
    assert [issue.column for issue in report.issues] == ["b", "c"]


def test_report_raise_for_errors_carries_issues() -> None:
    report = validate_dataset(_dataset(pd.DataFrame({"a": pd.Series([], dtype=float)})))

    with pytest.raises(DatasetValidationError) as exc_info:
        report.raise_for_errors()

    assert exc_info.value.issues == report.issues
    assert "validation failed with 1 issue" in str(exc_info.value).lower()


def test_report_raise_for_errors_does_nothing_when_valid() -> None:
    validate_dataset(_dataset(pd.DataFrame({"a": [1]}))).raise_for_errors()


def test_validator_rejects_wrong_dataset_type() -> None:
    with pytest.raises(InvalidDatasetError, match="expects a Dataset"):
        validate_dataset(cast(Any, pd.DataFrame({"a": [1]})))


def test_validator_rejects_wrong_config_type() -> None:
    with pytest.raises(InvalidParameterError, match="ValidationConfig"):
        validate_dataset(_dataset(pd.DataFrame({"a": [1]})), cast(Any, object()))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_rows": -1},
        {"min_columns": -1},
        {"min_rows": True},
        {"min_columns": 1.5},
        {"forbid_unnamed_columns": "yes"},
        {"required_columns": "abc"},
        {"required_columns": 5},
        {"required_columns": ("a", "a")},
        {"required_columns": ("a", "")},
    ],
)
def test_invalid_validation_config_raises(kwargs: dict[str, Any]) -> None:
    with pytest.raises(InvalidParameterError):
        ValidationConfig(**kwargs)


def test_required_columns_are_normalised_to_tuple() -> None:
    config = ValidationConfig(required_columns=["a", "b"])  # type: ignore[arg-type]

    assert config.required_columns == ("a", "b")
