"""Internal helpers for classifying exceptions into stable issue types."""

from __future__ import annotations

from data_mendrix.core.exceptions import DataMendrixError


def builtin_error_type(exc: BaseException) -> str:
    """Return a stable, backend-independent name for ``exc``.

    DataMendrix errors keep their own class name because it is part of the
    library's API. Third-party exceptions (pandas, NumPy, PyArrow, ...) are
    mapped to the nearest built-in exception class in their MRO, so reports do
    not leak backend-specific class names that vary between pandas versions.
    """
    for klass in type(exc).__mro__:
        if issubclass(klass, DataMendrixError):
            return klass.__name__
        if klass.__module__ == "builtins":
            return klass.__name__
    return type(exc).__name__
