"""DataMendrix public API.

The initial foundation covers CSV loading, structural validation, missing-value
statistics, duplicate-row detection, numeric descriptive statistics and combined
profiling. Repair, reporting, the HTTP API and the frontend are intentionally out
of scope for this release.
"""

from data_mendrix.core import (
    ColumnNotFoundError,
    CsvSource,
    DataMendrixError,
    Dataset,
    DatasetLoadError,
    DatasetValidationError,
    InvalidDatasetError,
    InvalidParameterError,
    UnsupportedFileTypeError,
    load_csv,
)
from data_mendrix.detection import (
    DuplicateReport,
    KeepPolicy,
    detect_duplicates,
)
from data_mendrix.profiling import (
    ColumnMissingStats,
    ColumnProfile,
    DatasetProfile,
    MissingValueReport,
    NumericColumnStatistics,
    NumericStatistics,
    missing_value_report,
    numeric_statistics,
    profile_dataset,
)
from data_mendrix.validation import (
    ValidationCode,
    ValidationConfig,
    ValidationIssue,
    ValidationReport,
    validate_dataset,
)

__version__ = "0.1.0"

__all__ = [
    "ColumnMissingStats",
    "ColumnNotFoundError",
    "ColumnProfile",
    "CsvSource",
    "DataMendrixError",
    "Dataset",
    "DatasetLoadError",
    "DatasetProfile",
    "DatasetValidationError",
    "DuplicateReport",
    "InvalidDatasetError",
    "InvalidParameterError",
    "KeepPolicy",
    "MissingValueReport",
    "NumericColumnStatistics",
    "NumericStatistics",
    "UnsupportedFileTypeError",
    "ValidationCode",
    "ValidationConfig",
    "ValidationIssue",
    "ValidationReport",
    "__version__",
    "detect_duplicates",
    "load_csv",
    "missing_value_report",
    "numeric_statistics",
    "profile_dataset",
    "validate_dataset",
]
