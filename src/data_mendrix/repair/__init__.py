"""Dataset repair: explicit operations, previews and immutable reports."""

from data_mendrix.repair.engine import (
    apply_repair,
    convert_dtypes,
    drop_missing,
    fill_missing,
    preview_repair,
    remove_duplicates,
)
from data_mendrix.repair.models import (
    ColumnChange,
    RepairAction,
    RepairIssue,
    RepairOperationReport,
    RepairReport,
    RepairResult,
)
from data_mendrix.repair.operations import (
    ConversionErrorPolicy,
    ConvertTypes,
    DropAxis,
    DropHow,
    DropMissing,
    FillMissing,
    FillStrategy,
    RemoveDuplicates,
    RepairKeepPolicy,
    RepairOperation,
)

__all__ = [
    "ColumnChange",
    "ConversionErrorPolicy",
    "ConvertTypes",
    "DropAxis",
    "DropHow",
    "DropMissing",
    "FillMissing",
    "FillStrategy",
    "RemoveDuplicates",
    "RepairAction",
    "RepairIssue",
    "RepairKeepPolicy",
    "RepairOperation",
    "RepairOperationReport",
    "RepairReport",
    "RepairResult",
    "apply_repair",
    "convert_dtypes",
    "drop_missing",
    "fill_missing",
    "preview_repair",
    "remove_duplicates",
]
