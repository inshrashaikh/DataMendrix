"""Core dataset container, loading utilities and exception hierarchy."""

from data_mendrix.core.dataset import Dataset
from data_mendrix.core.exceptions import (
    ColumnNotFoundError,
    DataMendrixError,
    DatasetLoadError,
    DatasetValidationError,
    InvalidDatasetError,
    InvalidParameterError,
    RepairConversionError,
    RepairError,
    UnsupportedFileTypeError,
)
from data_mendrix.core.loader import CsvSource, load_csv

__all__ = [
    "ColumnNotFoundError",
    "CsvSource",
    "DataMendrixError",
    "Dataset",
    "DatasetLoadError",
    "DatasetValidationError",
    "InvalidDatasetError",
    "InvalidParameterError",
    "RepairConversionError",
    "RepairError",
    "UnsupportedFileTypeError",
    "load_csv",
]
