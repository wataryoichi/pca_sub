"""MOM (momentum) baseline strategy signal."""

from __future__ import annotations

import pandas as pd


def compute_momentum_signal(
    jp_cc_returns: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    """Compute simple momentum signal for JP sectors.

    s_{j,t} = (1/L) * sum_{tau in W_t} r^{cc}_{j,tau}

    This is the rolling mean of JP close-to-close returns over the past L days.

    Args:
        jp_cc_returns: JP close-to-close returns (date x JP tickers)
        window: L, lookback window

    Returns:
        Signal DataFrame (date x JP tickers), NaN where insufficient history
    """
    return jp_cc_returns.shift(1).rolling(window=window, min_periods=window).mean()
