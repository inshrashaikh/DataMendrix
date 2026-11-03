"""Class-imbalance analysis for an explicit target column."""

from __future__ import annotations

import math
from collections.abc import Hashable
from dataclasses import dataclass
from numbers import Real

from data_mendrix.core.dataset import Dataset
from data_mendrix.core.exceptions import InvalidDatasetError, InvalidParameterError
from data_mendrix.detection._columns import unique_column_position


@dataclass(frozen=True, slots=True)
class ClassCount:
    """Frequency information for one target class."""

    label: Hashable
    count: int
    percentage: float


@dataclass(frozen=True, slots=True)
class ClassImbalanceReport:
    """Class frequencies and threshold-based imbalance assessment.

    Percentages and the imbalance ratio use non-missing target values. Missing
    target values are reported separately through ``missing_count``.
    """

    target: str
    row_count: int
    observed_count: int
    missing_count: int
    classes: tuple[ClassCount, ...]
    majority_fraction: float
    minority_fraction: float
    imbalance_ratio: float
    imbalance_ratio_threshold: float
    minority_fraction_threshold: float

    @property
    def class_count(self) -> int:
        """Number of observed target classes."""
        return len(self.classes)

    @property
    def is_binary(self) -> bool:
        """Whether exactly two target classes were observed."""
        return self.class_count == 2

    @property
    def is_multiclass(self) -> bool:
        """Whether at least three target classes were observed."""
        return self.class_count >= 3

    @property
    def majority_class(self) -> ClassCount | None:
        """Most frequent class, or ``None`` when no target value is observed."""
        return self.classes[0] if self.classes else None

    @property
    def minority_class(self) -> ClassCount | None:
        """Least frequent class, or ``None`` when no target value is observed."""
        return self.classes[-1] if self.classes else None

    @property
    def is_imbalanced(self) -> bool:
        """Whether a configured imbalance threshold is triggered.

        At least two observed classes are required. The ratio condition uses
        ``>`` rather than ``>=``; the minority-fraction condition uses ``<``.
        A one-class target is not classified as imbalanced because it is better
        described as a constant target.
        """
        if self.class_count < 2:
            return False
        return (
            self.imbalance_ratio > self.imbalance_ratio_threshold
            or self.minority_fraction < self.minority_fraction_threshold
        )


def analyze_class_imbalance(
    dataset: Dataset,
    *,
    target: Hashable,
    imbalance_ratio_threshold: float = 2.0,
    minority_fraction_threshold: float = 0.1,
) -> ClassImbalanceReport:
    """Analyse target-class frequencies without modifying the dataset.

    Parameters
    ----------
    dataset:
        Dataset to inspect.
    target:
        Explicit target column label. The label must identify exactly one
        column.
    imbalance_ratio_threshold:
        Threshold for ``majority_count / minority_count``. A value greater than
        this threshold marks the target as imbalanced. Must be greater than
        ``1.0``. Defaults to ``2.0``.
    minority_fraction_threshold:
        A minority fraction strictly below this value marks the target as
        imbalanced. Set to ``0.0`` to disable this condition. Must be in the
        interval ``[0, 0.5]``. Defaults to ``0.1``.

    Returns
    -------
    ClassImbalanceReport
        Immutable class counts, percentages, majority/minority metrics and the
        configured thresholds. Classes are ordered from most to least frequent;
        ties retain first-appearance order. Missing target values are excluded
        from the percentages and counts.

    Raises
    ------
    InvalidDatasetError
        If ``dataset`` is not a :class:`~data_mendrix.core.dataset.Dataset`.
    InvalidParameterError
        If a threshold is invalid or the target label is not unique.
    ColumnNotFoundError
        If the target column does not exist.

    Examples
    --------
    >>> report = analyze_class_imbalance(dataset, target="outcome")
    >>> report.class_count
    3
    >>> report.is_imbalanced
    True
    """
    if not isinstance(dataset, Dataset):
        raise InvalidDatasetError(
            f"analyze_class_imbalance() expects a Dataset, got {type(dataset).__name__}."
        )
    validated_ratio_threshold = _validate_ratio_threshold(imbalance_ratio_threshold)
    validated_minority_threshold = _validate_minority_threshold(minority_fraction_threshold)

    frame = dataset._view()
    position = unique_column_position(frame, target, parameter="target")
    series = frame.iloc[:, position]
    observed = series.dropna()
    observed_count = int(observed.size)
    missing_count = int(series.isna().sum())
    counts = observed.value_counts(sort=False).sort_values(ascending=False, kind="stable")

    classes: list[ClassCount] = []
    for label, count in counts.items():
        class_count = int(count)
        classes.append(
            ClassCount(
                label=label,
                count=class_count,
                percentage=class_count / observed_count if observed_count else 0.0,
            )
        )

    majority_fraction, minority_fraction, imbalance_ratio = _class_metrics(
        tuple(classes),
        observed_count=observed_count,
    )

    return ClassImbalanceReport(
        target=str(target),
        row_count=int(frame.shape[0]),
        observed_count=observed_count,
        missing_count=missing_count,
        classes=tuple(classes),
        majority_fraction=majority_fraction,
        minority_fraction=minority_fraction,
        imbalance_ratio=imbalance_ratio,
        imbalance_ratio_threshold=validated_ratio_threshold,
        minority_fraction_threshold=validated_minority_threshold,
    )


def _validate_ratio_threshold(value: float) -> float:
    """Validate the majority-to-minority ratio threshold."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise InvalidParameterError(
            "imbalance_ratio_threshold must be a finite number greater than 1.0."
        )
    threshold = float(value)
    if not math.isfinite(threshold) or threshold <= 1.0:
        raise InvalidParameterError(
            "imbalance_ratio_threshold must be a finite number greater than 1.0."
        )
    return threshold


def _validate_minority_threshold(value: float) -> float:
    """Validate the minority-fraction threshold."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise InvalidParameterError("minority_fraction_threshold must be in the interval [0, 0.5].")
    threshold = float(value)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 0.5:
        raise InvalidParameterError("minority_fraction_threshold must be in the interval [0, 0.5].")
    return threshold


def _class_metrics(
    classes: tuple[ClassCount, ...],
    *,
    observed_count: int,
) -> tuple[float, float, float]:
    """Return majority fraction, minority fraction and imbalance ratio."""
    if observed_count == 0 or not classes:
        return math.nan, math.nan, math.nan

    majority = classes[0]
    minority = classes[-1]
    majority_fraction = majority.count / observed_count
    minority_fraction = minority.count / observed_count
    if len(classes) == 1:
        return majority_fraction, minority_fraction, 1.0
    return majority_fraction, minority_fraction, majority.count / minority.count
