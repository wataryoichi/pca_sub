"""Tests for rolling standardization."""

import numpy as np
import pandas as pd
import pytest

from src.standardize import rolling_standardize


def test_no_future_leakage():
    """Standardization at time t should only use data from t-L to t-1."""
    dates = pd.date_range("2020-01-01", periods=100, freq="B")
    rng = np.random.default_rng(42)
    returns = pd.DataFrame({"A": rng.standard_normal(100)}, index=dates)

    z = rolling_standardize(returns, window=10)

    # First 10 values should be NaN (insufficient history)
    assert z.iloc[:10]["A"].isna().all()
    # 11th value should be valid
    assert not np.isnan(z.iloc[10]["A"])


def test_standardized_stats():
    """After standardization, each column over its window should be roughly mean=0, std=1."""
    dates = pd.date_range("2020-01-01", periods=200, freq="B")
    rng = np.random.default_rng(42)
    returns = pd.DataFrame({"A": rng.standard_normal(200) * 0.01 + 0.001}, index=dates)

    z = rolling_standardize(returns, window=60)
    valid = z.dropna()
    # Not exactly 0/1 since each point uses a different window, but should be reasonable
    assert abs(valid["A"].mean()) < 0.5
    assert 0.3 < valid["A"].std() < 3.0
