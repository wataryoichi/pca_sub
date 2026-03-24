"""Transaction cost model for strategy returns.

This is a daily open-to-close strategy: every position is entered at open
and exited at close. Full round-trip cost applies every day.

Cost components:
1. Trading cost: round-trip = 2 * one_way_bps per unit of gross exposure
2. Short borrow cost: daily rate applied to short exposure
3. Illiquidity premium: additional cost for illiquid tickers
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_trading_cost(
    weights: pd.DataFrame,
    one_way_bps: float = 5.0,
    short_extra_bps: float = 3.0,
    illiquid_mask: pd.DataFrame | None = None,
    illiquid_extra_bps: float = 5.0,
) -> pd.Series:
    """Compute daily trading cost.

    Since this is a daily strategy (enter at open, exit at close):
    - Round-trip trade cost = 2 * one_way_bps * gross_exposure_per_side
      where gross_exposure_per_side = sum(|w_long|) = sum(|w_short|) = 1.0
      So trade cost = 2 * one_way_bps * 1.0 per side * 2 sides...

    More precisely: for $1 portfolio capital:
    - Buy long positions worth $1 (cost: one_way * $1)
    - Sell short positions worth $1 (cost: one_way * $1)
    - Sell long positions at close (cost: one_way * $1)
    - Buy back short positions at close (cost: one_way * $1)
    - Total trade cost = 4 * one_way_bps * $1 per $1 notional per side

    Since gross_exposure = 2 and each dollar is traded twice (open + close):
    trade_cost = 2 * one_way_bps * gross_exposure

    Args:
        weights: (date x tickers) portfolio weights (sum(|w|) ≈ 2)
        one_way_bps: One-way cost in basis points (commission + half spread)
        short_extra_bps: Additional daily cost for short positions (borrow fee / 252)
        illiquid_mask: (date x tickers) boolean mask, True = illiquid
        illiquid_extra_bps: Additional one-way cost for illiquid tickers

    Returns:
        Series of daily cost (as decimal fraction of portfolio)
    """
    bps = 1e-4

    # Turnover-based trading cost: one_way per dollar traded
    # Turnover = sum(|w_t - w_{t-1}|), which captures actual trading activity
    # For daily strategy: turnover ≈ 4.0 (full rebuild each day)
    # For multi-day hold: turnover = 0 on non-rebalance days
    turnover = weights.diff().abs().sum(axis=1)
    # First day: treat as full entry
    turnover.iloc[0] = weights.iloc[0].abs().sum()
    trade_cost = one_way_bps * bps * turnover

    # Short borrow cost (daily): applied to short leg every day held
    short_exposure = weights.clip(upper=0).abs().sum(axis=1)  # ≈ 1.0
    short_cost = short_extra_bps * bps * short_exposure

    total = trade_cost + short_cost

    # Illiquid extra cost (on turnover, not gross exposure)
    if illiquid_mask is not None:
        common_idx = weights.index.intersection(illiquid_mask.index)
        common_cols = weights.columns.intersection(illiquid_mask.columns)
        w_diff = weights.diff().abs()
        w_diff.iloc[0] = weights.iloc[0].abs()
        illiq_turnover = w_diff.loc[common_idx, common_cols] * illiquid_mask.loc[
            common_idx, common_cols
        ].astype(float)
        illiq_cost = illiquid_extra_bps * bps * illiq_turnover.sum(axis=1)
        total = total.add(illiq_cost, fill_value=0.0)

    return total


def apply_cost(
    gross_returns: pd.Series,
    costs: pd.Series,
) -> pd.Series:
    """Subtract costs from gross returns.

    r_net = r_gross - cost
    """
    common = gross_returns.index.intersection(costs.index)
    return gross_returns.loc[common] - costs.loc[common]


def cost_breakeven_analysis(
    gross_returns: pd.Series,
    weights: pd.DataFrame,
) -> dict[str, float]:
    """Compute breakeven cost levels.

    Returns the one-way cost (bps) at which the strategy breaks even.
    Uses actual turnover to compute cost, not gross exposure.
    """
    bps = 1e-4
    w = weights.loc[gross_returns.index]
    turnover = w.diff().abs().sum(axis=1)
    turnover.iloc[0] = w.iloc[0].abs().sum()
    avg_turnover = turnover.mean()
    avg_daily_return = gross_returns.mean()

    # breakeven: avg_return = one_way_bps * bps * avg_turnover
    if avg_turnover < 1e-12:
        breakeven_bps = 0.0
    else:
        breakeven_bps = avg_daily_return / (bps * avg_turnover)

    return {
        "breakeven_one_way_bps": breakeven_bps,
        "avg_daily_gross_return_bps": avg_daily_return / bps,
        "avg_daily_turnover": avg_turnover,
    }
