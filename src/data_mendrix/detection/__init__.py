"""Dataset detection: duplicates, outliers, constants and class imbalance."""

from data_mendrix.detection.constants import (
    ConstantColumnStats,
    ConstantFeatureReport,
    detect_constant_features,
)
from data_mendrix.detection.duplicates import (
    DuplicateReport,
    KeepPolicy,
    detect_duplicates,
)
from data_mendrix.detection.imbalance import (
    ClassCount,
    ClassImbalanceReport,
    analyze_class_imbalance,
)
from data_mendrix.detection.outliers import (
    ColumnOutlierReport,
    OutlierDirection,
    OutlierEvidence,
    OutlierMethod,
    OutlierReport,
    detect_outliers,
)

__all__ = [
    "ClassCount",
    "ClassImbalanceReport",
    "ColumnOutlierReport",
    "ConstantColumnStats",
    "ConstantFeatureReport",
    "DuplicateReport",
    "KeepPolicy",
    "OutlierDirection",
    "OutlierEvidence",
    "OutlierMethod",
    "OutlierReport",
    "analyze_class_imbalance",
    "detect_constant_features",
    "detect_duplicates",
    "detect_outliers",
]
