"""Unified signal builder: runs MOM, PCA_PLAIN, PCA_SUB over the backtest period."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .constants import JP_TICKERS, N_JP, N_US, US_TICKERS
from .covariance import compute_correlation_matrix
from .momentum import compute_momentum_signal
from .pca_plain import compute_pca_signal, eigen_decompose, split_us_jp_loadings
from .pca_sub import PCASubModel
from .standardize import rolling_standardize

logger = logging.getLogger(__name__)


def build_all_signals(
    us_cc_returns: pd.DataFrame,
    jp_cc_returns: pd.DataFrame,
    window: int,
    n_components: int,
    lambda_reg: float,
    Cfull: np.ndarray,
    backtest_start: str,
    backtest_end: str,
    tickers_all: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Build signals for all strategies over the backtest period.

    Args:
        us_cc_returns: US close-to-close returns (full history)
        jp_cc_returns: JP close-to-close returns (full history)
        window: L, rolling window
        n_components: K
        lambda_reg: λ
        Cfull: (N, N) long-term correlation matrix
        backtest_start: Start date for signal output
        backtest_end: End date for signal output
        tickers_all: Ordered list of all tickers [US..., JP...]

    Returns:
        Dict with keys 'MOM', 'PCA_PLAIN', 'PCA_SUB', each a DataFrame
        indexed by date with JP ticker columns.
    """
    if tickers_all is None:
        tickers_all = list(us_cc_returns.columns) + list(jp_cc_returns.columns)

    n_us = len(us_cc_returns.columns)
    n_jp = len(jp_cc_returns.columns)
    jp_cols = list(jp_cc_returns.columns)

    # --- Standardize all returns ---
    all_cc = pd.concat([us_cc_returns, jp_cc_returns], axis=1)
    z_all = rolling_standardize(all_cc, window=window)

    # --- MOM ---
    mom_signal = compute_momentum_signal(jp_cc_returns, window=window)

    # --- PCA_PLAIN and PCA_SUB ---
    pca_sub_model = PCASubModel(
        Cfull=Cfull,
        tickers=tickers_all,
        n_us=n_us,
        n_jp=n_jp,
        lambda_reg=lambda_reg,
        n_components=n_components,
    )

    dates = z_all.loc[backtest_start:backtest_end].index
    pca_plain_signals: list[dict] = []
    pca_sub_signals: list[dict] = []

    for t_idx, date in enumerate(dates):
        row = z_all.loc[date]

        # Check that at least all US and JP tickers have data
        # For tickers with NaN (e.g. XLC before 2018), fill with 0 in z
        z_all_t = row.fillna(0.0).values  # (N,)
        z_us_t = z_all_t[:n_us]

        # Need valid US data for signal generation
        us_valid = row.iloc[:n_us].notna().sum()
        if us_valid < n_us // 2:
            pca_plain_signals.append({c: np.nan for c in jp_cols})
            pca_sub_signals.append({c: np.nan for c in jp_cols})
            continue

        # Build rolling sample correlation from the window ending at t
        loc = z_all.index.get_loc(date)
        if loc < window:
            pca_plain_signals.append({c: np.nan for c in jp_cols})
            pca_sub_signals.append({c: np.nan for c in jp_cols})
            continue

        z_window = z_all.iloc[loc - window : loc].fillna(0.0)
        if len(z_window) < window:
            pca_plain_signals.append({c: np.nan for c in jp_cols})
            pca_sub_signals.append({c: np.nan for c in jp_cols})
            continue

        Ct = compute_correlation_matrix(z_window.values)

        # PCA_PLAIN: no regularization
        _, V_K_plain = eigen_decompose(Ct, n_components)
        V_U_plain, V_J_plain = split_us_jp_loadings(V_K_plain, n_us, n_jp)
        sig_plain = compute_pca_signal(z_us_t, V_U_plain, V_J_plain)
        pca_plain_signals.append(dict(zip(jp_cols, sig_plain)))

        # PCA_SUB: regularized
        sig_sub = pca_sub_model.compute_signal(Ct, z_us_t)
        pca_sub_signals.append(dict(zip(jp_cols, sig_sub)))

    pca_plain_df = pd.DataFrame(pca_plain_signals, index=dates)
    pca_sub_df = pd.DataFrame(pca_sub_signals, index=dates)
    mom_df = mom_signal.loc[backtest_start:backtest_end]

    return {
        "MOM": mom_df,
        "PCA_PLAIN": pca_plain_df,
        "PCA_SUB": pca_sub_df,
    }
