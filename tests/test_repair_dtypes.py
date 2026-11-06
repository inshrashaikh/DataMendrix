"""Tests for dtype repair and conversion error policies."""

from __future__ import annotations

from decimal import InvalidOperation
from typing import Any, cast

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from data_mendrix import (
    ColumnNotFoundError,
    Dataset,
    InvalidDatasetError,
    InvalidParameterError,
    RepairAction,
    RepairConversionError,
    convert_dtypes,
)
from data_mendrix.repair._errors import builtin_error_type


def test_convert_dtype_success_reports_before_and_after() -> None:
    dataset = Dataset(
        pd.DataFrame({"value": ["1", None, "3"], "other": [1, 2, 3]}),
        name="types",
        source="memory",
    )
    before = dataset.frame

    result = convert_dtypes(dataset, conversions={"value": "Int64"})
    operation = result.report.operations[0]
    change = operation.column_changes[0]

    assert str(result.dataset.frame["value"].dtype) == "Int64"
    assert result.dataset.frame["value"].tolist() == [1, pd.NA, 3]
    assert result.dataset.frame["other"].tolist() == [1, 2, 3]
    assert operation.action is RepairAction.CONVERT_DTYPES
    assert operation.affected_columns == ("value",)
    assert operation.affected_rows == ()
    assert operation.change_count == 1
    assert change.column == "value"
    assert change.before_dtype in ("str", "object")  # pandas 3 vs 2 string storage
    assert change.after_dtype == "Int64"
    assert change.change_count == 3
    assert change.detail == "dtype converted"
    assert result.dataset.name == "types"
    assert result.dataset.source == "memory"
    assert_frame_equal(dataset.frame, before)


def test_convert_multiple_columns() -> None:
    dataset = Dataset(pd.DataFrame({"a": ["1", "2"], "b": ["3.5", "4.5"], "c": [1, 2]}))

    result = convert_dtypes(dataset, conversions={"a": "Int64", "b": "float64"})
    operation = result.report.operations[0]

    assert str(result.dataset.frame["a"].dtype) == "Int64"
    assert str(result.dataset.frame["b"].dtype) == "float64"
    assert str(result.dataset.frame["c"].dtype) == "int64"
    assert operation.affected_columns == ("a", "b")
    assert operation.change_count == 2
    assert len(operation.column_changes) == 2


def test_convert_to_datetime() -> None:
    dataset = Dataset(pd.DataFrame({"when": ["2020-01-01", "2020-02-01"]}))

    result = convert_dtypes(dataset, conversions={"when": "datetime64[ns]"})

    assert str(result.dataset.frame["when"].dtype) == "datetime64[ns]"
    assert result.report.operations[0].column_changes[0].after_dtype == "datetime64[ns]"


def test_convert_coerce_numeric_invalid_values_to_missing() -> None:
    dataset = Dataset(
        pd.DataFrame(
            {
                "value": ["1", "bad", "3"],
            },
            index=["r0", "r1", "r2"],
        )
    )

    result = convert_dtypes(dataset, conversions={"value": "Int64"}, errors="coerce")
    operation = result.report.operations[0]
    change = operation.column_changes[0]

    assert str(result.dataset.frame["value"].dtype) == "Int64"
    assert result.dataset.frame.loc["r1", "value"] is pd.NA
    assert operation.affected_rows == ("r1",)
    assert operation.affected_columns == ("value",)
    assert "1 invalid value" in change.detail


def test_convert_coerce_datetime_invalid_values_to_missing() -> None:
    dataset = Dataset(pd.DataFrame({"when": ["2020-01-01", "bad"]}))

    result = convert_dtypes(
        dataset,
        conversions={"when": "datetime64[ns]"},
        errors="coerce",
    )

    assert str(result.dataset.frame["when"].dtype) == "datetime64[ns]"
    assert result.dataset.frame["when"].isna().tolist() == [False, True]
    assert result.report.operations[0].affected_rows == (1,)


def test_convert_coerce_timezone_aware_datetime() -> None:
    dataset = Dataset(pd.DataFrame({"when": ["2020-01-01", "bad"]}))

    result = convert_dtypes(
        dataset,
        conversions={"when": "datetime64[ns, UTC]"},
        errors="coerce",
    )

    dtype = result.dataset.frame["when"].dtype
    assert str(dtype) == "datetime64[ns, UTC]"
    assert result.dataset.frame["when"].isna().tolist() == [False, True]
    assert result.report.operations[0].affected_rows == (1,)


def test_convert_coerce_timedelta_invalid_values_to_missing() -> None:
    dataset = Dataset(pd.DataFrame({"duration": ["1 day", "bad"]}))

    result = convert_dtypes(
        dataset,
        conversions={"duration": "timedelta64[ns]"},
        errors="coerce",
    )

    assert str(result.dataset.frame["duration"].dtype) == "timedelta64[ns]"
    assert result.dataset.frame["duration"].isna().tolist() == [False, True]


def test_convert_coerce_generic_boolean_values() -> None:
    dataset = Dataset(pd.DataFrame({"flag": [True, False, "bad"]}))

    result = convert_dtypes(dataset, conversions={"flag": "boolean"}, errors="coerce")

    assert str(result.dataset.frame["flag"].dtype) == "boolean"
    assert result.dataset.frame["flag"].tolist() == [True, False, pd.NA]
    assert result.report.operations[0].affected_rows == (2,)


def test_convert_coerce_generic_preserves_existing_missing_values() -> None:
    dataset = Dataset(pd.DataFrame({"flag": [True, None, "bad"]}))

    result = convert_dtypes(dataset, conversions={"flag": "boolean"}, errors="coerce")

    assert str(result.dataset.frame["flag"].dtype) == "boolean"
    assert result.dataset.frame["flag"].tolist() == [True, pd.NA, pd.NA]
    assert result.report.operations[0].affected_rows == (2,)


def test_convert_skip_leaves_failed_column_and_applies_others() -> None:
    dataset = Dataset(pd.DataFrame({"bad": ["1", "x"], "good": ["1", "2"]}))

    result = convert_dtypes(
        dataset,
        conversions={"bad": "Int64", "good": "Int64"},
        errors="skip",
    )
    operation = result.report.operations[0]

    assert str(result.dataset.frame["bad"].dtype) in ("str", "object")
    assert result.dataset.frame["bad"].tolist() == ["1", "x"]
    assert str(result.dataset.frame["good"].dtype) == "Int64"
    assert operation.affected_columns == ("good",)
    assert operation.change_count == 1
    assert len(operation.issues) == 1
    assert operation.issues[0].column == "bad"
    assert operation.issues[0].error_type == "ValueError"


class ExplodingValue:
    """Throw a third-party-style exception during numeric conversion."""

    def __init__(self, text: str) -> None:
        self.text = text

    def __int__(self) -> int:
        if self.text == "bad":
            raise InvalidOperation("boom")
        return int(self.text)

    def __float__(self) -> float:
        if self.text == "bad":
            raise InvalidOperation("boom")
        return float(self.text)

    def __repr__(self) -> str:
        return self.text


@pytest.mark.parametrize("errors", ["skip", "coerce"])
def test_convert_handles_non_builtin_third_party_errors(errors: str) -> None:
    values = [ExplodingValue("1"), ExplodingValue("bad")]
    dataset = Dataset(pd.DataFrame({"value": values}))

    if errors == "skip":
        result = convert_dtypes(dataset, conversions={"value": "int64"}, errors="skip")
        issue = result.report.operations[0].issues[0]
        assert issue.error_type == "ArithmeticError"
        assert result.dataset.frame["value"].tolist() == values
    else:
        with pytest.raises(RepairConversionError):
            convert_dtypes(dataset, conversions={"value": "int64"}, errors="coerce")


def test_convert_raise_reports_typed_conversion_error() -> None:
    dataset = Dataset(pd.DataFrame({"value": ["1", "bad"]}))
    before = dataset.frame

    with pytest.raises(RepairConversionError) as exc_info:
        convert_dtypes(dataset, conversions={"value": "Int64"})

    error = exc_info.value
    assert error.column == "value"
    assert error.target == "Int64"
    assert error.__cause__ is not None
    assert_frame_equal(dataset.frame, before)


def test_convert_coerce_integer_target_cannot_represent_missing() -> None:
    dataset = Dataset(pd.DataFrame({"value": ["1", "bad"]}))

    with pytest.raises(RepairConversionError, match="cannot represent missing"):
        convert_dtypes(dataset, conversions={"value": "int64"}, errors="coerce")


def test_convert_rejects_target_that_hides_existing_missing_values() -> None:
    dataset = Dataset(pd.DataFrame({"flag": [True, None]}))

    with pytest.raises(RepairConversionError, match="existing missing"):
        convert_dtypes(dataset, conversions={"flag": "bool"})


def test_convert_same_dtype_is_a_noop() -> None:
    dataset = Dataset(pd.DataFrame({"value": [1.0, 2.0]}))

    result = convert_dtypes(dataset, conversions={"value": "float64"})
    operation = result.report.operations[0]

    assert not operation.changed
    assert operation.affected_columns == ()
    assert operation.column_changes == ()
    assert not result.report.changed


def test_convert_empty_frame_changes_dtype() -> None:
    dataset = Dataset(pd.DataFrame({"value": pd.Series([], dtype="float64")}))

    result = convert_dtypes(dataset, conversions={"value": "Int64"})
    operation = result.report.operations[0]

    assert result.dataset.shape == (0, 1)
    assert str(result.dataset.frame["value"].dtype) == "Int64"
    assert operation.changed
    assert operation.affected_rows == ()
    assert operation.affected_columns == ("value",)


@pytest.mark.parametrize("target", ["not_a_dtype", None])
def test_convert_rejects_invalid_target_dtype(target: object) -> None:
    dataset = Dataset(pd.DataFrame({"value": [1, 2]}))

    with pytest.raises(InvalidParameterError, match="Target dtype"):
        convert_dtypes(dataset, conversions={"value": target})


def test_builtin_error_type_maps_third_party_errors_to_builtin_category() -> None:
    class BackendSpecificError(ValueError):
        pass

    assert builtin_error_type(BackendSpecificError("boom")) == "ValueError"
    assert builtin_error_type(TypeError("boom")) == "TypeError"
    assert builtin_error_type(OverflowError("boom")) == "OverflowError"


def test_builtin_error_type_keeps_datamendrix_error_names() -> None:
    error = RepairConversionError("boom", column="value", target="int64")

    assert builtin_error_type(error) == "RepairConversionError"


def test_convert_skip_records_builtin_error_type() -> None:
    dataset = Dataset(pd.DataFrame({"bad": ["1", "x"], "good": ["1", "2"]}))

    result = convert_dtypes(
        dataset,
        conversions={"bad": "Int64"},
        errors="skip",
    )
    issue = result.report.operations[0].issues[0]

    assert issue.error_type == "ValueError"


def test_convert_rejects_unknown_or_duplicate_column() -> None:
    dataset = Dataset(pd.DataFrame({"a": [1, 2]}))

    with pytest.raises(ColumnNotFoundError) as exc_info:
        convert_dtypes(dataset, conversions={"missing": "float64"})
    assert exc_info.value.columns == ("missing",)

    duplicate = Dataset(pd.DataFrame([[1, 2]], columns=["a", "a"]))
    with pytest.raises(InvalidParameterError, match="not unique"):
        convert_dtypes(duplicate, conversions={"a": "float64"})


def test_convert_rejects_empty_conversions_or_invalid_errors() -> None:
    dataset = Dataset(pd.DataFrame({"a": [1, 2]}))

    with pytest.raises(InvalidParameterError, match="non-empty"):
        convert_dtypes(dataset, conversions={})

    with pytest.raises(InvalidParameterError, match="errors"):
        convert_dtypes(dataset, conversions={"a": "float64"}, errors=cast(Any, "ignore"))


def test_convert_rejects_wrong_dataset_type() -> None:
    with pytest.raises(InvalidDatasetError, match="expects a Dataset"):
        convert_dtypes(cast(Any, pd.DataFrame({"a": [1]})), conversions={"a": "float64"})
