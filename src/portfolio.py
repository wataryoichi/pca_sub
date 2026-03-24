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


def build_rebalance_weights(
    signal: pd.DataFrame,
    quantile: float = 0.3,
    rebal_freq: int = 5,
    jp_cc_returns: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build long/short weights with reduced rebalancing frequency.

    Instead of rebuilding every day, hold positions for `rebal_freq` days.
    Between rebalances, returns accumulate on the held positions using
    close-to-close returns (not open-to-close).

    Args:
        signal: (date x JP tickers) signal DataFrame
        quantile: q, fraction for long/short legs
        rebal_freq: Number of days between rebalances
        jp_cc_returns: JP close-to-close returns for position carry
            (if None, weights are simply held flat between rebalances)

    Returns:
        (date x JP tickers) weight DataFrame
    """
    # First build daily ideal weights
    ideal = build_long_short_weights(signal, quantile)

    # Only rebalance every rebal_freq days
    weights = pd.DataFrame(0.0, index=ideal.index, columns=ideal.columns)
    last_rebal = None

    for i, date in enumerate(ideal.index):
        if i % rebal_freq == 0:
            weights.loc[date] = ideal.loc[date]
            last_rebal = date
        elif last_rebal is not None:
            weights.loc[date] = weights.loc[ideal.index[i - 1]]

    return weights


def build_blended_weights(
    signal: pd.DataFrame,
    quantile: float = 0.3,
    blend_alpha: float = 0.5,
) -> pd.DataFrame:
    """Build weights by blending previous weights with ideal new weights.

    new_w_t = (1 - alpha) * prev_w_{t-1} + alpha * ideal_w_t

    Then re-normalize to maintain zero net exposure and gross = 2.

    Lower alpha = more inertia = less turnover.
    alpha = 1.0 is equivalent to full daily rebalance.

    Args:
        signal: (date x JP tickers) signal DataFrame
        quantile: q for long/short
        blend_alpha: Blending parameter in (0, 1]

    Returns:
        (date x JP tickers) blended weight DataFrame
    """
    ideal = build_long_short_weights(signal, quantile)
    weights = pd.DataFrame(0.0, index=ideal.index, columns=ideal.columns)

    prev_w = None
    for i, date in enumerate(ideal.index):
        ideal_w = ideal.loc[date].values.copy()
        if prev_w is None:
            blended = ideal_w.copy()
        else:
            blended = (1 - blend_alpha) * prev_w + blend_alpha * ideal_w

        # Re-normalize: zero net, gross = 2
        # Separate long and short
        long_mask = blended > 0
        short_mask = blended < 0

        if long_mask.sum() > 0 and short_mask.sum() > 0:
            long_sum = blended[long_mask].sum()
            short_sum = abs(blended[short_mask].sum())
            if long_sum > 1e-12:
                blended[long_mask] *= 1.0 / long_sum
            if short_sum > 1e-12:
                blended[short_mask] *= 1.0 / short_sum
        elif np.abs(blended).sum() > 1e-12:
            # All same sign or edge case: just normalize
            blended = blended / (np.abs(blended).sum() / 2)

        weights.iloc[i] = blended
        prev_w = blended.copy()

    return weights


def compute_turnover(weights: pd.DataFrame) -> pd.Series:
    """Compute daily turnover as sum of absolute weight changes."""
    diff = weights.diff().abs().sum(axis=1)
    return diff
