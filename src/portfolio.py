"""Portfolio construction: long/short based on signal ranking."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_long_short_weights(
    signal: pd.DataFrame,
    quantile: float = 0.3,
) -> pd.DataFrame:
    """Build equal-weight long/short portfolio weights from signal.

    For each date:
    1. Rank JP sectors by signal
    2. Top q fraction -> long (+1/n_long)
    3. Bottom q fraction -> short (-1/n_short)
    4. Rest -> 0

    Net exposure = 0 (long sum = +1, short sum = -1).

    Args:
        signal: (date x JP tickers) signal DataFrame
        quantile: q, fraction for long/short legs

    Returns:
        (date x JP tickers) weight DataFrame
    """
    weights = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)

    for date in signal.index:
        row = signal.loc[date].dropna()
        if len(row) == 0:
            continue

        n_available = len(row)
        n_leg = max(1, int(np.floor(n_available * quantile)))

        # Rank: sort by signal value
        ranked = row.sort_values()
        short_tickers = ranked.index[:n_leg]   # bottom
        long_tickers = ranked.index[-n_leg:]   # top

        weights.loc[date, long_tickers] = 1.0 / n_leg
        weights.loc[date, short_tickers] = -1.0 / n_leg

    return weights


def compute_portfolio_returns(
    weights: pd.DataFrame,
    jp_oc_returns: pd.DataFrame,
) -> pd.Series:
    """Compute daily portfolio returns.

    R_{t+1} = sum_j w_{j,t+1} * r^{oc}_{j,t+1}

    Note: weights and returns must be aligned (same index = jp_date).

    Args:
        weights: (date x JP tickers) weight DataFrame
        jp_oc_returns: (date x JP tickers) open-to-close returns

    Returns:
        Series of daily portfolio returns
    """
    common_dates = weights.index.intersection(jp_oc_returns.index)
    w = weights.loc[common_dates]
    r = jp_oc_returns.loc[common_dates]

    # Fill NaN returns with 0 (if a ticker has no data on a day)
    port_ret = (w * r.fillna(0)).sum(axis=1)
    return port_ret


def compute_turnover(weights: pd.DataFrame) -> pd.Series:
    """Compute daily turnover as sum of absolute weight changes."""
    diff = weights.diff().abs().sum(axis=1)
    return diff
