"""Liquidity filter for JP ETFs.

Filters out tickers on days where liquidity is insufficient, based on:
- Rolling average turnover (close * volume)
- Rolling average volume
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_turnover_value(
    close: pd.DataFrame,
    volume: pd.DataFrame,
) -> pd.DataFrame:
    """Compute daily turnover value (close * volume) in local currency."""
    common_cols = close.columns.intersection(volume.columns)
    common_idx = close.index.intersection(volume.index)
    return close.loc[common_idx, common_cols] * volume.loc[common_idx, common_cols]


def build_liquidity_mask(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    min_avg_turnover: float = 100_000_000,
    lookback_days: int = 20,
) -> pd.DataFrame:
    """Build a boolean mask: True = illiquid (should be excluded or penalized).

    For each date t, compute rolling average turnover over [t-lookback, ..., t-1].
    Mark as illiquid if average turnover < threshold.

    Args:
        close: Close prices (date x tickers)
        volume: Volume (date x tickers)
        min_avg_turnover: Minimum average daily turnover (JPY)
        lookback_days: Number of days for rolling average

    Returns:
        Boolean DataFrame: True = illiquid
    """
    turnover = compute_turnover_value(close, volume)

    # Rolling mean of past `lookback_days` (not including today)
    avg_turnover = (
        turnover.shift(1)
        .rolling(window=lookback_days, min_periods=lookback_days // 2)
        .mean()
    )

    illiquid = avg_turnover < min_avg_turnover
    return illiquid


def apply_liquidity_filter(
    signal: pd.DataFrame,
    illiquid_mask: pd.DataFrame,
) -> pd.DataFrame:
    """Set signal to NaN for illiquid tickers (excluded from ranking).

    Args:
        signal: (date x tickers) signal DataFrame
        illiquid_mask: (date x tickers) True = illiquid

    Returns:
        Filtered signal with NaN for excluded tickers
    """
    common_idx = signal.index.intersection(illiquid_mask.index)
    common_cols = signal.columns.intersection(illiquid_mask.columns)

    filtered = signal.copy()
    mask = illiquid_mask.loc[common_idx, common_cols]
    filtered.loc[common_idx, common_cols] = filtered.loc[
        common_idx, common_cols
    ].where(~mask)

    return filtered


def liquidity_exclusion_report(
    illiquid_mask: pd.DataFrame,
) -> pd.DataFrame:
    """Generate summary of which tickers were excluded on which dates.

    Returns:
        DataFrame with columns: date, n_excluded, excluded_tickers, n_available
    """
    records = []
    for date in illiquid_mask.index:
        row = illiquid_mask.loc[date]
        excluded = list(row[row].index)
        records.append({
            "date": date,
            "n_excluded": len(excluded),
            "excluded_tickers": ",".join(excluded) if excluded else "",
            "n_available": (~row).sum(),
        })
    return pd.DataFrame(records)
