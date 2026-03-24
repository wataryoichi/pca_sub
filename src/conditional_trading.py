"""Conditional trading: trade only when signal strength exceeds a threshold.

Instead of trading every day, filter to days where the signal is strong enough
to justify the transaction costs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .cost_model import apply_cost, compute_trading_cost
from .metrics import compute_all_metrics
from .portfolio import build_long_short_weights, compute_portfolio_returns


def compute_signal_strength(signal: pd.DataFrame) -> pd.Series:
    """Compute daily signal strength as L2 norm of the signal vector.

    Higher values indicate a stronger cross-sectional spread in predictions.
    """
    return np.sqrt((signal ** 2).sum(axis=1))


def build_conditional_weights(
    signal: pd.DataFrame,
    strength: pd.Series,
    threshold: float,
    quantile: float = 0.3,
) -> pd.DataFrame:
    """Build weights only on days where signal strength exceeds threshold.

    On weak-signal days, all weights are zero (no trade).
    """
    # Build ideal weights for all days
    ideal = build_long_short_weights(signal, quantile=quantile)

    # Zero out weak-signal days
    mask = strength >= threshold
    weights = ideal.copy()
    weights.loc[~mask] = 0.0

    return weights


def run_conditional_frontier(
    signal: pd.DataFrame,
    jp_oc_returns: pd.DataFrame,
    quantile: float = 0.3,
    n_points: int = 20,
    cost_one_way_bps: float = 5.0,
    cost_short_bps: float = 3.0,
) -> pd.DataFrame:
    """Sweep signal strength thresholds and compute gross/net performance.

    Returns a DataFrame with one row per threshold, showing the trade-off
    between trading frequency and performance.
    """
    strength = compute_signal_strength(signal)

    # Percentile-based thresholds (0th = trade every day, 90th = trade top 10%)
    percentiles = np.linspace(0, 95, n_points)
    thresholds = [np.nanpercentile(strength, p) for p in percentiles]

    records = []
    for pct, thresh in zip(percentiles, thresholds):
        weights = build_conditional_weights(signal, strength, thresh, quantile)
        port_ret = compute_portfolio_returns(weights, jp_oc_returns).dropna()

        # Count trading days (days with non-zero weights)
        trade_days = (weights.abs().sum(axis=1) > 0.01).sum()
        total_days = len(weights)

        if len(port_ret) < 10 or trade_days < 5:
            continue

        # Gross metrics
        metrics_gross = compute_all_metrics(port_ret)

        # Net metrics
        costs = compute_trading_cost(
            weights.loc[port_ret.index],
            one_way_bps=cost_one_way_bps,
            short_extra_bps=cost_short_bps,
        )
        port_ret_net = apply_cost(port_ret, costs)
        metrics_net = compute_all_metrics(port_ret_net)

        # Per-trade-day alpha
        trade_day_mask = weights.loc[port_ret.index].abs().sum(axis=1) > 0.01
        trade_day_returns = port_ret[trade_day_mask]
        avg_trade_day_gross_bps = trade_day_returns.mean() / 1e-4 if len(trade_day_returns) > 0 else 0

        records.append({
            "strength_percentile": pct,
            "threshold": thresh,
            "trade_days": trade_days,
            "trade_pct": trade_days / total_days * 100,
            "gross_AR": metrics_gross["AR"],
            "gross_RR": metrics_gross["R/R"],
            "gross_MDD": metrics_gross["MDD"],
            "net_AR": metrics_net["AR"],
            "net_RR": metrics_net["R/R"],
            "net_MDD": metrics_net["MDD"],
            "avg_trade_day_gross_bps": avg_trade_day_gross_bps,
        })

    return pd.DataFrame(records)


def run_conditional_oos(
    signal: pd.DataFrame,
    jp_oc_returns: pd.DataFrame,
    periods: dict[str, tuple[str, str]],
    quantile: float = 0.3,
    target_percentiles: list[float] | None = None,
    cost_one_way_bps: float = 5.0,
    cost_short_bps: float = 3.0,
) -> pd.DataFrame:
    """Run conditional trading for multiple sub-periods.

    For each period, compute signal strength thresholds using ONLY in-period data
    (no forward-looking bias).

    Args:
        signal: Full signal DataFrame
        jp_oc_returns: Full JP OC returns
        periods: Dict of period_name -> (start, end)
        quantile: Portfolio quantile
        target_percentiles: List of percentiles to evaluate (default: [0, 50, 70, 80, 90])
        cost_one_way_bps: One-way cost
        cost_short_bps: Short extra cost

    Returns:
        DataFrame with results per period per percentile
    """
    if target_percentiles is None:
        target_percentiles = [0, 50, 70, 80, 90]

    strength = compute_signal_strength(signal)
    records = []

    for period_name, (start, end) in periods.items():
        sig_period = signal.loc[start:end]
        ret_period = jp_oc_returns.loc[start:end]
        str_period = strength.loc[start:end]

        if len(sig_period) < 20:
            continue

        for pct in target_percentiles:
            # Threshold computed within this period only
            thresh = np.nanpercentile(str_period, pct)
            weights = build_conditional_weights(sig_period, str_period, thresh, quantile)
            port_ret = compute_portfolio_returns(weights, ret_period).dropna()

            trade_days = (weights.abs().sum(axis=1) > 0.01).sum()
            total_days = len(sig_period)

            if len(port_ret) < 5 or trade_days < 3:
                continue

            metrics_gross = compute_all_metrics(port_ret)

            costs = compute_trading_cost(
                weights.loc[port_ret.index],
                one_way_bps=cost_one_way_bps,
                short_extra_bps=cost_short_bps,
            )
            port_ret_net = apply_cost(port_ret, costs)
            metrics_net = compute_all_metrics(port_ret_net)

            trade_mask = weights.loc[port_ret.index].abs().sum(axis=1) > 0.01
            trade_rets = port_ret[trade_mask]
            avg_bps = trade_rets.mean() / 1e-4 if len(trade_rets) > 0 else 0

            records.append({
                "period": period_name,
                "percentile": pct,
                "trade_days": trade_days,
                "trade_pct": trade_days / total_days * 100,
                "gross_AR": metrics_gross["AR"],
                "gross_RR": metrics_gross["R/R"],
                "net_AR": metrics_net["AR"],
                "net_RR": metrics_net["R/R"],
                "avg_trade_bps": avg_bps,
            })

    return pd.DataFrame(records)
