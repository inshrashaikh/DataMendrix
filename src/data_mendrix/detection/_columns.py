"""Compatibility import path for shared column-selection helpers.

The implementation lives in :mod:`data_mendrix.core.columns` so both detection
and repair APIs can reuse it without a dependency between feature packages.
"""

from __future__ import annotations

from data_mendrix.core.columns import resolve_column_selection, unique_column_position

__all__ = ["resolve_column_selection", "unique_column_position"]
