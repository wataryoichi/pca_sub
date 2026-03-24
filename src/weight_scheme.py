"""Portfolio weight schemes beyond equal weight.

Weight A: Equal weight (baseline)
Weight B: Rank-normalized score proportional
Weight C: Signal / estimated volatility proportional
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_rank_proportional_weights(
    signal: pd.DataFrame,
    quantile: float = 0.3,
    max_weight: float = 0.35,
) -> pd.DataFrame:
    """Weight B: Rank-normalized proportional weights.

    Within long/short legs, weight proportional to rank position.
    Normalized so long sum = +1, short sum = -1.
    Capped at max_weight per position.
    """
    weights = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)

    for date in signal.index:
        row = signal.loc[date].dropna()
        if len(row) == 0:
            continue

        n = len(row)
        n_leg = max(1, int(np.floor(n * quantile)))

        ranked = row.sort_values()
        short_t = ranked.index[:n_leg]
        long_t = ranked.index[-n_leg:]

        # Rank scores within each leg (1 = weakest, n_leg = strongest)
        long_ranks = np.arange(1, n_leg + 1, dtype=float)
        short_ranks = np.arange(n_leg, 0, -1, dtype=float)

        # Normalize
        long_w = long_ranks / long_ranks.sum()
        short_w = -short_ranks / short_ranks.sum()

        # Apply max weight cap
        long_w = np.minimum(long_w, max_weight)
        long_w = long_w / long_w.sum()
        short_w = -np.minimum(np.abs(short_w), max_weight)
        short_w = short_w / np.abs(short_w).sum()

        weights.loc[date, long_t] = long_w
        weights.loc[date, short_t] = short_w

    return weights


def build_signal_vol_proportional_weights(
    signal: pd.DataFrame,
    rolling_vol: pd.DataFrame,
    quantile: float = 0.3,
    max_weight: float = 0.35,
) -> pd.DataFrame:
    """Weight C: Signal / volatility proportional weights.

    w_i proportional to |signal_i| / vol_i within each leg.
    Normalized and capped.

    Args:
        signal: Signal DataFrame
        rolling_vol: Rolling volatility DataFrame (same index/columns as signal)
        quantile: Leg fraction
        max_weight: Per-position cap
    """
    weights = pd.DataFrame(0.0, index=signal.index, columns=signal.columns)

    for date in signal.index:
        sig_row = signal.loc[date].dropna()
        if len(sig_row) == 0:
            continue

        vol_row = rolling_vol.loc[date].reindex(sig_row.index).fillna(1.0)
        vol_row = vol_row.clip(lower=1e-8)

        n = len(sig_row)
        n_leg = max(1, int(np.floor(n * quantile)))

        ranked = sig_row.sort_values()
        short_t = ranked.index[:n_leg]
        long_t = ranked.index[-n_leg:]

        # Signal / vol ratio
        long_ratio = sig_row[long_t].abs() / vol_row[long_t]
        short_ratio = sig_row[short_t].abs() / vol_row[short_t]

        # Normalize
        if long_ratio.sum() > 1e-12:
            long_w = long_ratio / long_ratio.sum()
        else:
            long_w = pd.Series(1.0 / n_leg, index=long_t)

        if short_ratio.sum() > 1e-12:
            short_w = -short_ratio / short_ratio.sum()
        else:
            short_w = pd.Series(-1.0 / n_leg, index=short_t)

        # Cap
        long_w = long_w.clip(upper=max_weight)
        long_w = long_w / long_w.sum()
        short_w = -short_w.abs().clip(upper=max_weight)
        short_w = short_w / short_w.abs().sum()

        weights.loc[date, long_t] = long_w.values
        weights.loc[date, short_t] = short_w.values

    return weights
