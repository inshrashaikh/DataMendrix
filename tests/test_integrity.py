"""Regression tests for data integrity and immutable reports."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import data_mendrix as dm

Analysis = Callable[[dm.Dataset], object]

ANALYSES = [
    pytest.param(dm.validate_dataset, id="validate_dataset"),
    pytest.param(dm.missing_value_report, id="missing_value_report"),
    pytest.param(dm.numeric_statistics, id="numeric_statistics"),
    pytest.param(dm.detect_duplicates, id="detect_duplicates"),
    pytest.param(dm.detect_outliers, id="detect_outliers"),
    pytest.param(dm.detect_constant_features, id="detect_constant_features"),
    pytest.param(
        lambda dataset: dm.analyze_class_imbalance(dataset, target="b"),
        id="analyze_class_imbalance",
    ),
    pytest.param(dm.profile_dataset, id="profile_dataset"),
]


@pytest.mark.parametrize("analysis", ANALYSES)
def test_analysis_does_not_mutate_dataset(analysis: Analysis) -> None:
    dataset = dm.Dataset(
        pd.DataFrame(
            {
                "a": [1.0, np.nan, 1.0],
                "b": ["x", "x", "y"],
            }
        ),
        name="integrity",
    )
    before = dataset.frame

    analysis(dataset)

    assert_frame_equal(dataset.frame, before)


def test_public_reports_are_frozen() -> None:
    dataset = dm.Dataset(
        pd.DataFrame(
            {
                "a": [1.0, np.nan, 1.0],
                "b": ["x", "x", "y"],
            }
        )
    )
    validation = dm.validate_dataset(dataset)
    profile = dm.profile_dataset(dataset)
    constants = dm.detect_constant_features(dataset)
    imbalance = dm.analyze_class_imbalance(dataset, target="b")
    outliers = dm.detect_outliers(dm.Dataset(pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]})))

    targets: list[tuple[object, str]] = [
        (dm.ValidationIssue(dm.ValidationCode.TOO_FEW_ROWS, "message"), "message"),
        (validation, "issues"),
        (profile, "row_count"),
        (profile.missing, "row_count"),
        (profile.missing.columns[0], "missing_count"),
        (profile.duplicates, "row_count"),
        (profile.numeric.column("a"), "mean"),
        (profile.column("a"), "name"),
        (outliers, "row_count"),
        (outliers.column("a"), "center"),
        (outliers.column("a").evidence[0], "score"),
        (constants, "row_count"),
        (constants.column("a"), "unique_count"),
        (imbalance, "row_count"),
        (imbalance.classes[0], "count"),
    ]

    for target, field in targets:
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(target, field, None)


def test_report_collections_are_tuples() -> None:
    dataset = dm.Dataset(pd.DataFrame({"a": [1.0, 1.0], "b": ["x", "y"]}))
    profile = dm.profile_dataset(dataset)
    outliers = dm.detect_outliers(dataset)
    constants = dm.detect_constant_features(dataset)
    imbalance = dm.analyze_class_imbalance(dataset, target="b")

    assert isinstance(profile.columns, tuple)
    assert isinstance(profile.numeric.columns, tuple)
    assert isinstance(profile.missing.columns, tuple)
    assert isinstance(profile.duplicates.duplicate_indices, tuple)
    assert isinstance(dm.validate_dataset(dataset).issues, tuple)
    assert isinstance(outliers.columns, tuple)
    assert isinstance(outliers.column("a").evidence, tuple)
    assert isinstance(outliers.outlier_indices, tuple)
    assert isinstance(constants.columns, tuple)
    assert isinstance(imbalance.classes, tuple)
