"""Tests for timezone-aware datetime coercion in dtype repair."""

from __future__ import annotations

import pandas as pd

from data_mendrix import Dataset, convert_dtypes


def test_coerce_to_tz_aware_datetime_localises_naive_values() -> None:
    dataset = Dataset(pd.DataFrame({"ts": ["2024-01-01 00:00:00", "garbage", None]}))

    result = convert_dtypes(dataset, conversions={"ts": "datetime64[ns, UTC]"}, errors="coerce")
    operation = result.report.operations[0]

    assert str(result.dataset.frame["ts"].dtype) == "datetime64[ns, UTC]"
    assert result.dataset.frame["ts"].tolist()[0] == pd.Timestamp("2024-01-01 00:00:00+0000")
    assert result.dataset.frame["ts"].isna().tolist() == [False, True, True]
    assert operation.affected_rows == (1,)


def test_coerce_to_tz_aware_datetime_converts_aware_values() -> None:
    aware = pd.to_datetime(["2024-01-01", "2024-06-01"]).tz_localize("Europe/Berlin")
    dataset = Dataset(pd.DataFrame({"ts": aware}))

    result = convert_dtypes(dataset, conversions={"ts": "datetime64[ns, UTC]"}, errors="coerce")

    converted = result.dataset.frame["ts"]
    assert str(converted.dtype) == "datetime64[ns, UTC]"
    assert converted.tolist() == list(aware.tz_convert("UTC"))


def test_coerce_to_plain_datetime_still_works() -> None:
    dataset = Dataset(pd.DataFrame({"ts": ["2024-01-01", "garbage"]}))

    result = convert_dtypes(dataset, conversions={"ts": "datetime64[ns]"}, errors="coerce")

    assert result.dataset.frame["ts"].isna().tolist() == [False, True]
    assert result.report.operations[0].affected_rows == (1,)


def test_raise_reports_typed_error_for_unparseable_datetimes() -> None:
    dataset = Dataset(pd.DataFrame({"ts": ["not-a-date", "2024-01-01"]}))

    import pytest

    from data_mendrix import RepairConversionError

    with pytest.raises(RepairConversionError):
        convert_dtypes(dataset, conversions={"ts": "datetime64[ns]"}, errors="raise")


def test_skip_leaves_unparseable_datetime_column_unchanged() -> None:
    dataset = Dataset(pd.DataFrame({"ts": ["not-a-date", "2024-01-01"]}))

    result = convert_dtypes(dataset, conversions={"ts": "datetime64[ns]"}, errors="skip")
    operation = result.report.operations[0]

    assert result.dataset.frame["ts"].tolist() == ["not-a-date", "2024-01-01"]
    assert operation.issues[0].column == "ts"
