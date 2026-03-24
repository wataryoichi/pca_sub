"""Return computation: close-to-close and open-to-close."""

from __future__ import annotations

import pandas as pd


def close_to_close_return(close: pd.DataFrame) -> pd.DataFrame:
    """Compute close-to-close returns: r_cc = close_t / close_{t-1} - 1.

    Args:
        close: Price DataFrame (date x tickers)

    Returns:
        Returns DataFrame (same shape, first row is NaN)
    """
    return close / close.shift(1) - 1


def open_to_close_return(
    open_prices: pd.DataFrame,
    close_prices: pd.DataFrame,
) -> pd.DataFrame:
    """Compute open-to-close returns: r_oc = close_t / open_t - 1.

    Args:
        open_prices: Open price DataFrame (date x tickers)
        close_prices: Close price DataFrame (date x tickers)

    Returns:
        Returns DataFrame
    """
    return close_prices / open_prices - 1
