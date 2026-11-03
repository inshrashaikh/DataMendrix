"""Tests for the top-level public API."""

from __future__ import annotations

import data_mendrix
from data_mendrix import (
    ColumnNotFoundError,
    DataMendrixError,
    DatasetLoadError,
    DatasetValidationError,
    InvalidDatasetError,
    InvalidParameterError,
    UnsupportedFileTypeError,
)


def test_version_is_exposed() -> None:
    assert data_mendrix.__version__ == "0.2.0"


def test_all_exported_names_exist() -> None:
    for name in data_mendrix.__all__:
        assert hasattr(data_mendrix, name), name


def test_detection_engine_names_are_exported() -> None:
    expected = {
        "ClassCount",
        "ClassImbalanceReport",
        "ColumnOutlierReport",
        "ConstantColumnStats",
        "ConstantFeatureReport",
        "OutlierDirection",
        "OutlierEvidence",
        "OutlierMethod",
        "OutlierReport",
        "analyze_class_imbalance",
        "detect_constant_features",
        "detect_outliers",
    }

    assert expected <= set(data_mendrix.__all__)


def test_exception_hierarchy() -> None:
    for exception in (
        DatasetLoadError,
        DatasetValidationError,
        InvalidDatasetError,
        InvalidParameterError,
        ColumnNotFoundError,
    ):
        assert issubclass(exception, DataMendrixError)

    assert issubclass(UnsupportedFileTypeError, DatasetLoadError)
    assert issubclass(InvalidParameterError, ValueError)
    assert issubclass(InvalidDatasetError, ValueError)
    assert issubclass(ColumnNotFoundError, KeyError)
