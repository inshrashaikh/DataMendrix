"""Duplicate detection."""

from data_mendrix.detection.duplicates import (
    DuplicateReport,
    KeepPolicy,
    detect_duplicates,
)

__all__ = ["DuplicateReport", "KeepPolicy", "detect_duplicates"]
