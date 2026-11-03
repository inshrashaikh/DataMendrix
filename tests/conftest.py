"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data_mendrix import Dataset


@pytest.fixture
def sample_frame() -> pd.DataFrame:
    """Return a small mixed-type frame with missing values."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "name": ["alpha", "beta", None, "beta"],
            "score": [10.0, np.nan, 30.0, 10.0],
            "active": [True, False, True, False],
            "category": ["x", "y", "x", "x"],
        }
    )


@pytest.fixture
def sample_dataset(sample_frame: pd.DataFrame) -> Dataset:
    """Wrap the shared mixed-type frame in a dataset."""
    return Dataset(sample_frame, name="sample")
