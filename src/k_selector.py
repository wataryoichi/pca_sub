"""Walk-forward monthly K selection.

Each month, select K from {3,4,5} using trailing train window.
Evaluates each K's Net R/R on train period, applies chosen K next month.
No future leak: selection uses only past data.
"""

from __future__ import annotations

import copy
import logging

import numpy as np
import pandas as pd

from .config import Config
from .constants import N_JP, N_US
from .covariance import compute_cfull, compute_correlation_matrix, regularize_correlation
from .pca_plain import eigen_decompose, split_us_jp_loadings, compute_pca_signal
from .pca_sub import PCASubModel
from .standardize import rolling_standardize

logger = logging.getLogger(__name__)


def walk_forward_monthly_k(
    all_cc: pd.DataFrame,
    z_all: pd.DataFrame,
    jp_oc: pd.DataFrame,
    cfg: Config,
    k_candidates: list[int] = [3, 4, 5],
    train_months: int = 24,
) -> tuple[pd.DataFrame, pd.Series]:
    """Walk-forward monthly K selection.

    For each month:
    1. Use trailing `train_months` of data to evaluate each K
    2. Select K with best Net R/R (ties: smaller K wins)
    3. Generate signals for the next month using selected K

    Args:
        all_cc: Combined US+JP close-to-close returns
        z_all: Standardized returns
        jp_oc: JP open-to-close returns
        cfg: Configuration
        k_candidates: K values to evaluate
        train_months: Number of months in train window

    Returns:
        signal_df: (date x JP tickers) signal DataFrame using selected K
        k_series: Series of selected K per date
    """
    n_us = len([c for c in all_cc.columns if not c.endswith('.T')])
    n_jp = len([c for c in all_cc.columns if c.endswith('.T')])
    jp_cols = [c for c in all_cc.columns if c.endswith('.T')]
    tickers_all = list(all_cc.columns)
    window = cfg.strategy.window_length

    # Precompute Cfull
    Cfull = compute_cfull(all_cc, cfg.cfull.start_date, cfg.cfull.end_date, window)

    # Get all months in backtest period
    bt_dates = z_all.loc[cfg.backtest.start_date:cfg.backtest.end_date].index
    months = pd.Series(bt_dates).dt.to_period('M').unique()

    signal_records = []
    k_records = []
    current_k = k_candidates[0]  # Default

    for month_idx, month in enumerate(months):
        month_start = month.start_time
        month_end = month.end_time

        # Dates in this month
        month_dates = bt_dates[(bt_dates >= month_start) & (bt_dates <= month_end)]
        if len(month_dates) == 0:
            continue

        # --- K Selection using train window ---
        train_end = month_start - pd.Timedelta(days=1)
        train_start = train_end - pd.DateOffset(months=train_months)

        # Only select K if we have enough train data
        if month_idx >= train_months // 12:
            best_k = k_candidates[0]
            best_metric = -999.0

            for k in k_candidates:
                try:
                    metric = _evaluate_k_on_period(
                        all_cc, z_all, jp_oc, Cfull, tickers_all,
                        n_us, n_jp, jp_cols, window, k,
                        cfg.strategy.lambda_reg, cfg.strategy.quantile,
                        str(train_start.date()), str(train_end.date()),
                    )
                    # Ties: prefer smaller K
                    if metric > best_metric or (abs(metric - best_metric) < 0.01 and k < best_k):
                        best_metric = metric
                        best_k = k
                except Exception:
                    pass

            current_k = best_k

        # --- Generate signals for this month using current_k ---
        model = PCASubModel(
            Cfull=Cfull, tickers=tickers_all,
            n_us=n_us, n_jp=n_jp,
            lambda_reg=cfg.strategy.lambda_reg,
            n_components=current_k,
        )

        for date in month_dates:
            loc = z_all.index.get_loc(date)
            if loc < window:
                signal_records.append({c: np.nan for c in jp_cols})
                k_records.append({"date": date, "k": current_k})
                continue

            row = z_all.loc[date].fillna(0.0).values
            z_us = row[:n_us]
            z_window = z_all.iloc[loc - window:loc].fillna(0.0)

            if len(z_window) < window:
                signal_records.append({c: np.nan for c in jp_cols})
            else:
                Ct = compute_correlation_matrix(z_window.values)
                sig = model.compute_signal(Ct, z_us)
                signal_records.append(dict(zip(jp_cols, sig)))

            k_records.append({"date": date, "k": current_k})

    signal_df = pd.DataFrame(signal_records, index=bt_dates[:len(signal_records)])
    k_df = pd.DataFrame(k_records)
    k_series = pd.Series(k_df["k"].values, index=k_df["date"].values, name="selected_k")

    return signal_df, k_series


def _evaluate_k_on_period(
    all_cc, z_all, jp_oc, Cfull, tickers_all,
    n_us, n_jp, jp_cols, window, k,
    lambda_reg, quantile, start, end,
) -> float:
    """Evaluate a specific K on a train period. Returns approximate R/R."""
    from .portfolio import build_long_short_weights, compute_portfolio_returns
    from .conditional_trading import compute_signal_strength, build_conditional_weights
    from .cost_model import apply_cost, compute_trading_cost
    from .metrics import risk_return_ratio

    model = PCASubModel(
        Cfull=Cfull, tickers=tickers_all,
        n_us=n_us, n_jp=n_jp,
        lambda_reg=lambda_reg, n_components=k,
    )

    dates = z_all.loc[start:end].index
    sigs = []
    for date in dates:
        loc = z_all.index.get_loc(date)
        if loc < window:
            sigs.append({c: np.nan for c in jp_cols})
            continue
        row = z_all.loc[date].fillna(0.0).values
        z_us = row[:n_us]
        z_window = z_all.iloc[loc - window:loc].fillna(0.0)
        if len(z_window) < window:
            sigs.append({c: np.nan for c in jp_cols})
            continue
        Ct = compute_correlation_matrix(z_window.values)
        sig = model.compute_signal(Ct, z_us)
        sigs.append(dict(zip(jp_cols, sig)))

    sig_df = pd.DataFrame(sigs, index=dates)

    # Apply 90th percentile filter
    strength = compute_signal_strength(sig_df)
    thresh = np.nanpercentile(strength, 90)
    weights = build_conditional_weights(sig_df, strength, thresh, quantile)
    port_ret = compute_portfolio_returns(weights, jp_oc).dropna()

    if len(port_ret) < 20:
        return -999.0

    # Compute net R/R with 3bp cost
    costs = compute_trading_cost(weights.loc[port_ret.index], 3.0, 2.0)
    port_net = apply_cost(port_ret, costs)
    return risk_return_ratio(port_net)
