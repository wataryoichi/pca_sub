"""Rolling standardization of returns."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_standardize(
    returns: pd.DataFrame,
    window: int,
    min_periods: int | None = None,
    eps: float = 1e-8,
) -> pd.DataFrame:
    """Standardize returns using a trailing window [t-L, ..., t-1].

    For each date t, compute mean and std over the previous `window` observations
    (NOT including t itself), then standardize the return at t.

    z_{i,t} = (r_{i,t} - mu_{i,t}) / sigma_{i,t}
    where mu, sigma are computed over W_t = {t-L, ..., t-1}

    Args:
        returns: Return DataFrame (date x tickers)
        window: L, the rolling window length
        min_periods: Minimum valid observations in window (default = window)
        eps: Floor for standard deviation to avoid division by zero

    Returns:
        Standardized DataFrame (NaN where insufficient history)
    """
    if min_periods is None:
        min_periods = window

    # Shift by 1 so we compute stats over [t-L, ..., t-1] not including t
    # rolling(window).mean() on shifted series gives mean of past L values
    shifted = returns.shift(1)
    roll_mean = shifted.rolling(window=window, min_periods=min_periods).mean()
    roll_std = shifted.rolling(window=window, min_periods=min_periods).std(ddof=0)

    # Floor std to avoid division by zero
    roll_std = roll_std.clip(lower=eps)

    z = (returns - roll_mean) / roll_std
    return z


def rolling_mean_std(
    returns: pd.DataFrame,
    window: int,
    min_periods: int | None = None,
    eps: float = 1e-8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute rolling mean and std over trailing window [t-L, ..., t-1].

    Returns:
        (mean, std) DataFrames
    """
    if min_periods is None:
        min_periods = window

    shifted = returns.shift(1)
    roll_mean = shifted.rolling(window=window, min_periods=min_periods).mean()
    roll_std = shifted.rolling(window=window, min_periods=min_periods).std(ddof=0)
    roll_std = roll_std.clip(lower=eps)
    return roll_mean, roll_std
