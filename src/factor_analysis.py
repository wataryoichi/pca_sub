"""Factor score decomposition and alpha attribution.

Decomposes the PCA_SUB signal into individual factor contributions
and measures each factor's predictive power for JP returns.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .constants import N_JP, N_US
from .covariance import compute_correlation_matrix, regularize_correlation
from .pca_plain import eigen_decompose, split_us_jp_loadings
from .pca_sub import PCASubModel

logger = logging.getLogger(__name__)


def compute_factor_scores_timeseries(
    z_all: pd.DataFrame,
    pca_sub_model: PCASubModel,
    window: int,
    backtest_start: str,
    backtest_end: str,
    n_us: int = N_US,
    n_jp: int = N_JP,
) -> tuple[pd.DataFrame, pd.DataFrame, list[np.ndarray]]:
    """Compute daily factor scores and per-factor signals.

    Returns:
        factor_scores: (date x K) DataFrame of factor scores f_t
        factor_signals: dict mapping k -> (date x JP tickers) signal from factor k alone
        V_J_list: list of V_J matrices for each date (for B_t computation)
    """
    n_components = pca_sub_model.n_components
    dates = z_all.loc[backtest_start:backtest_end].index
    jp_cols = list(z_all.columns[n_us:n_us + n_jp])

    f_records: list[dict] = []
    # Per-factor signals: signal_k[j] = V_J[:,k] * f_k
    per_factor_signals: dict[int, list[dict]] = {k: [] for k in range(n_components)}
    V_J_list: list[np.ndarray] = []

    for date in dates:
        row = z_all.loc[date]
        z_all_t = row.fillna(0.0).values
        z_us_t = z_all_t[:n_us]

        loc = z_all.index.get_loc(date)
        if loc < window:
            f_records.append({f"f{k}": np.nan for k in range(n_components)})
            for k in range(n_components):
                per_factor_signals[k].append({c: np.nan for c in jp_cols})
            V_J_list.append(np.full((n_jp, n_components), np.nan))
            continue

        z_window = z_all.iloc[loc - window:loc].fillna(0.0)
        if len(z_window) < window:
            f_records.append({f"f{k}": np.nan for k in range(n_components)})
            for k in range(n_components):
                per_factor_signals[k].append({c: np.nan for c in jp_cols})
            V_J_list.append(np.full((n_jp, n_components), np.nan))
            continue

        Ct = compute_correlation_matrix(z_window.values)
        C_reg = regularize_correlation(Ct, pca_sub_model.C0, pca_sub_model.lambda_reg)
        _, V_K = eigen_decompose(C_reg, n_components)
        V_U, V_J = split_us_jp_loadings(V_K, n_us, n_jp)

        # Factor scores
        f_t = V_U.T @ z_us_t  # (K,)
        f_records.append({f"f{k}": f_t[k] for k in range(n_components)})

        # Per-factor signal: V_J[:, k] * f_k
        for k in range(n_components):
            sig_k = V_J[:, k] * f_t[k]
            per_factor_signals[k].append(dict(zip(jp_cols, sig_k)))

        V_J_list.append(V_J)

    factor_scores = pd.DataFrame(f_records, index=dates)

    factor_signal_dfs = {}
    for k in range(n_components):
        factor_signal_dfs[k] = pd.DataFrame(per_factor_signals[k], index=dates)

    return factor_scores, factor_signal_dfs, V_J_list


def compute_propagation_matrix_stats(
    z_all: pd.DataFrame,
    pca_sub_model: PCASubModel,
    window: int,
    backtest_start: str,
    backtest_end: str,
    n_us: int = N_US,
    n_jp: int = N_JP,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute time-average and std of propagation matrix B_t = V_J V_U^T.

    Returns:
        B_mean: (N_JP, N_US) time-average propagation matrix
        B_std: (N_JP, N_US) time-std of propagation matrix
    """
    n_components = pca_sub_model.n_components
    dates = z_all.loc[backtest_start:backtest_end].index

    B_list: list[np.ndarray] = []

    for date in dates:
        loc = z_all.index.get_loc(date)
        if loc < window:
            continue

        z_window = z_all.iloc[loc - window:loc].fillna(0.0)
        if len(z_window) < window:
            continue

        Ct = compute_correlation_matrix(z_window.values)
        C_reg = regularize_correlation(Ct, pca_sub_model.C0, pca_sub_model.lambda_reg)
        _, V_K = eigen_decompose(C_reg, n_components)
        V_U, V_J = split_us_jp_loadings(V_K, n_us, n_jp)

        B_t = V_J @ V_U.T  # (N_JP, N_US)
        B_list.append(B_t)

    B_stack = np.stack(B_list)
    B_mean = B_stack.mean(axis=0)
    B_std = B_stack.std(axis=0)

    return B_mean, B_std


def factor_alpha_attribution(
    factor_signal_dfs: dict[int, pd.DataFrame],
    jp_oc_returns: pd.DataFrame,
    quantile: float = 0.3,
) -> pd.DataFrame:
    """Compute alpha attribution: what fraction of total alpha comes from each factor.

    Builds a long/short portfolio for each individual factor's signal and
    computes its standalone performance.

    Returns:
        DataFrame with per-factor metrics
    """
    from .portfolio import build_long_short_weights, compute_portfolio_returns
    from .metrics import compute_all_metrics
    from .cost_model import cost_breakeven_analysis

    records = []
    for k, sig_df in factor_signal_dfs.items():
        weights = build_long_short_weights(sig_df, quantile=quantile)
        port_ret = compute_portfolio_returns(weights, jp_oc_returns)
        port_ret = port_ret.dropna()
        if len(port_ret) < 10:
            continue
        metrics = compute_all_metrics(port_ret)
        be = cost_breakeven_analysis(port_ret, weights)
        records.append({
            "factor": f"f{k} ({'Global' if k == 0 else 'Country' if k == 1 else 'CycDef' if k == 2 else f'PC{k+1}'})",
            "factor_idx": k,
            **metrics,
            "breakeven_bps": be["breakeven_one_way_bps"],
        })

    return pd.DataFrame(records)
