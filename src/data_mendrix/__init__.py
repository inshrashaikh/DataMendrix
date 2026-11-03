"""DataMendrix public API.

The foundation covers CSV loading, structural validation, missing-value
statistics, duplicate-row detection, numeric descriptive statistics and combined
profiling. The detection engine adds univariate outlier detection, constant
feature detection and class-imbalance analysis. Repair, reporting, the HTTP API
and the frontend are intentionally out of scope for this release.
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
    ClassCount,
    ClassImbalanceReport,
    ColumnOutlierReport,
    ConstantColumnStats,
    ConstantFeatureReport,
    DuplicateReport,
    KeepPolicy,
    OutlierDirection,
    OutlierEvidence,
    OutlierMethod,
    OutlierReport,
    analyze_class_imbalance,
    detect_constant_features,
    detect_duplicates,
    detect_outliers,
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

__version__ = "0.2.0"

__all__ = [
    "ClassCount",
    "ClassImbalanceReport",
    "ColumnMissingStats",
    "ColumnNotFoundError",
    "ColumnOutlierReport",
    "ColumnProfile",
    "ConstantColumnStats",
    "ConstantFeatureReport",
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
    "OutlierDirection",
    "OutlierEvidence",
    "OutlierMethod",
    "OutlierReport",
    "UnsupportedFileTypeError",
    "ValidationCode",
    "ValidationConfig",
    "ValidationIssue",
    "ValidationReport",
    "__version__",
    "analyze_class_imbalance",
    "detect_constant_features",
    "detect_duplicates",
    "detect_outliers",
    "load_csv",
    "missing_value_report",
    "numeric_statistics",
    "profile_dataset",
    "validate_dataset",
]
