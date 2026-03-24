"""Composite signal filters for trade-day selection.

Filter A: Signal strength top 10% (baseline)
Filter B: A + long-short spread > threshold
Filter C: A + factor score magnitude > threshold
Filter D: A + B + C combined
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .conditional_trading import compute_signal_strength


def compute_long_short_spread(
    signal: pd.DataFrame,
    quantile: float = 0.3,
) -> pd.Series:
    """Compute daily long-short average score spread.

    spread_t = mean(top_q scores) - mean(bottom_q scores)
    """
    spreads = pd.Series(np.nan, index=signal.index)
    for date in signal.index:
        row = signal.loc[date].dropna()
        if len(row) == 0:
            continue
        n = max(1, int(np.floor(len(row) * quantile)))
        ranked = row.sort_values()
        short_mean = ranked.iloc[:n].mean()
        long_mean = ranked.iloc[-n:].mean()
        spreads.loc[date] = long_mean - short_mean
    return spreads


def compute_factor_strength(
    z_all: pd.DataFrame,
    model,
    window: int,
    start: str,
    end: str,
    n_us: int,
    n_jp: int,
) -> pd.Series:
    """Compute daily factor score L2 norm ||f_t||."""
    from .covariance import compute_correlation_matrix, regularize_correlation
    from .pca_plain import eigen_decompose, split_us_jp_loadings

    dates = z_all.loc[start:end].index
    f_norm = pd.Series(np.nan, index=dates)

    for date in dates:
        loc = z_all.index.get_loc(date)
        if loc < window:
            continue
        row = z_all.loc[date].fillna(0.0).values
        z_us = row[:n_us]
        z_window = z_all.iloc[loc - window:loc].fillna(0.0)
        if len(z_window) < window:
            continue
        Ct = compute_correlation_matrix(z_window.values)
        C_reg = regularize_correlation(Ct, model.C0, model.lambda_reg)
        _, V_K = eigen_decompose(C_reg, model.n_components)
        V_U, V_J = split_us_jp_loadings(V_K, n_us, n_jp)
        f = V_U.T @ z_us
        f_norm.loc[date] = np.linalg.norm(f)

    return f_norm


def apply_composite_filter(
    signal: pd.DataFrame,
    strength: pd.Series,
    spread: pd.Series | None = None,
    factor_strength: pd.Series | None = None,
    strength_pct: float = 90,
    spread_pct: float = 70,
    factor_pct: float = 70,
    use_spread: bool = False,
    use_factor: bool = False,
    train_start: str | None = None,
    train_end: str | None = None,
) -> pd.Series:
    """Apply composite filter. Returns boolean mask of trade days.

    Thresholds are computed from train period only (no future leak).
    If train period not specified, uses percentiles on the full series
    (only valid for in-sample analysis).

    Args:
        signal: Signal DataFrame
        strength: Signal strength series
        spread: Long-short spread series
        factor_strength: Factor score norm series
        strength_pct: Percentile for signal strength
        spread_pct: Percentile for spread
        factor_pct: Percentile for factor strength
        use_spread: Whether to apply spread condition
        use_factor: Whether to apply factor strength condition
        train_start/end: Period for threshold computation

    Returns:
        Boolean Series: True = trade day
    """
    if train_start and train_end:
        str_train = strength.loc[train_start:train_end]
    else:
        str_train = strength

    mask = strength >= np.nanpercentile(str_train.dropna(), strength_pct)

    if use_spread and spread is not None:
        if train_start and train_end:
            sp_train = spread.loc[train_start:train_end]
        else:
            sp_train = spread
        sp_thresh = np.nanpercentile(sp_train.dropna(), spread_pct)
        mask = mask & (spread >= sp_thresh)

    if use_factor and factor_strength is not None:
        if train_start and train_end:
            fs_train = factor_strength.loc[train_start:train_end]
        else:
            fs_train = factor_strength
        fs_thresh = np.nanpercentile(fs_train.dropna(), factor_pct)
        mask = mask & (factor_strength >= fs_thresh)

    return mask
