"""Tests for portfolio construction."""

import numpy as np
import pandas as pd
import pytest

from src.portfolio import build_long_short_weights


def test_weights_sum_to_zero():
    """Net exposure should be zero."""
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    tickers = [f"T{i}" for i in range(17)]
    rng = np.random.default_rng(42)
    signal = pd.DataFrame(rng.standard_normal((5, 17)), index=dates, columns=tickers)

    weights = build_long_short_weights(signal, quantile=0.3)
    for date in dates:
        assert abs(weights.loc[date].sum()) < 1e-10


def test_weights_gross_exposure():
    """Gross exposure (sum of abs weights) should be 2."""
    dates = pd.date_range("2020-01-01", periods=5, freq="B")
    tickers = [f"T{i}" for i in range(17)]
    rng = np.random.default_rng(42)
    signal = pd.DataFrame(rng.standard_normal((5, 17)), index=dates, columns=tickers)

    weights = build_long_short_weights(signal, quantile=0.3)
    for date in dates:
        assert abs(weights.loc[date].abs().sum() - 2.0) < 1e-10


def test_weights_leg_counts():
    """Check number of long and short positions."""
    dates = pd.date_range("2020-01-01", periods=1, freq="B")
    tickers = [f"T{i}" for i in range(17)]
    rng = np.random.default_rng(42)
    signal = pd.DataFrame(rng.standard_normal((1, 17)), index=dates, columns=tickers)

    weights = build_long_short_weights(signal, quantile=0.3)
    row = weights.iloc[0]
    n_long = (row > 0).sum()
    n_short = (row < 0).sum()
    # floor(17 * 0.3) = 5
    assert n_long == 5
    assert n_short == 5
