"""Final judgment experiment runner: 3 experiments + integration.

Runs walk-forward K, composite filters, weight schemes, and final combos.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .basket_backtest import _distribute_etf_weights_to_stocks, _load_stock_prices
from .calendar_align import build_aligned_dataset
from .conditional_trading import build_conditional_weights, compute_signal_strength
from .config import Config, load_config
from .cost_model import apply_cost, compute_trading_cost, cost_breakeven_analysis
from .covariance import compute_cfull
from .data_loader import build_price_panels, load_raw_data
from .k_selector import walk_forward_monthly_k
from .metrics import compute_all_metrics, compute_subperiod_metrics
from .portfolio import build_long_short_weights
from .signal_builder import build_all_signals
from .signal_filter import apply_composite_filter, compute_factor_strength, compute_long_short_spread
from .standardize import rolling_standardize, rolling_mean_std
from .weight_scheme import build_rank_proportional_weights, build_signal_vol_proportional_weights
from .pca_sub import PCASubModel

logger = logging.getLogger(__name__)

PERIODS = {
    "Full": ("2015-01-01", "2025-12-31"),
    "Train": ("2015-01-01", "2019-12-31"),
    "Valid": ("2020-01-01", "2022-12-31"),
    "Test": ("2023-01-01", "2025-12-31"),
}


def _prepare(cfg: Config):
    """Common data prep."""
    raw_data = load_raw_data(cfg.data.raw_dir)
    us_close, jp_close, jp_open, _ = build_price_panels(raw_data)
    date_map, us_close_a, jp_close_a, jp_open_a = build_aligned_dataset(us_close, jp_close, jp_open)

    us_cc = us_close_a / us_close_a.shift(1) - 1
    us_cc.index = date_map["us_date"].values
    jp_cc = jp_close_a / jp_close_a.shift(1) - 1
    jp_cc.index = date_map["us_date"].values
    jp_oc = pd.DataFrame(
        jp_close_a.values / jp_open_a.values - 1,
        index=pd.to_datetime(date_map["us_date"].values), columns=jp_close.columns,
    )
    all_cc = pd.concat([us_cc, jp_cc], axis=1)
    all_cc.index = pd.to_datetime(all_cc.index)
    z_all = rolling_standardize(all_cc, window=cfg.strategy.window_length)

    Cfull = compute_cfull(all_cc, cfg.cfull.start_date, cfg.cfull.end_date, cfg.strategy.window_length)
    tickers_all = list(all_cc.columns)
    n_us = len(us_cc.columns)
    n_jp = len(jp_cc.columns)
    jp_cols = list(jp_cc.columns)

    # Signals (K=3 baseline)
    signals = build_all_signals(
        us_cc, jp_cc, cfg.strategy.window_length, cfg.strategy.n_components,
        cfg.strategy.lambda_reg, Cfull, cfg.backtest.start_date, cfg.backtest.end_date, tickers_all,
    )

    # Stock data
    stock_close, stock_open = _load_stock_prices("data/raw/stocks")
    jp_dates = date_map["jp_date"].values
    us_dates = pd.to_datetime(date_map["us_date"].values)
    stock_oc = pd.DataFrame(
        stock_close.reindex(jp_dates).values / stock_open.reindex(jp_dates).values - 1,
        index=us_dates, columns=stock_close.columns,
    )

    # Model for factor analysis
    model = PCASubModel(
        Cfull=Cfull, tickers=tickers_all, n_us=n_us, n_jp=n_jp,
        lambda_reg=cfg.strategy.lambda_reg, n_components=cfg.strategy.n_components,
    )

    return {
        "all_cc": all_cc, "z_all": z_all, "jp_oc": jp_oc, "jp_cc": jp_cc,
        "stock_oc": stock_oc, "signals": signals, "Cfull": Cfull,
        "tickers_all": tickers_all, "n_us": n_us, "n_jp": n_jp,
        "jp_cols": jp_cols, "model": model, "cfg": cfg,
    }


def _eval(etf_weights, stock_oc, start, end, cost_bps=3.0, short_bps=2.0):
    """Evaluate basket performance."""
    sw = _distribute_etf_weights_to_stocks(etf_weights)
    cd = sw.index.intersection(stock_oc.index)
    cd = cd[(cd >= start) & (cd <= end)]
    cc = sw.columns.intersection(stock_oc.columns)
    w = sw.loc[cd, cc]
    r = stock_oc.loc[cd, cc].fillna(0)
    pr = (w * r).sum(axis=1).dropna()
    if len(pr) < 10:
        return {"AR": np.nan, "RR": np.nan, "MDD": np.nan, "net_AR": np.nan, "net_RR": np.nan, "net_MDD": np.nan, "BE": np.nan, "days": 0}
    mg = compute_all_metrics(pr)
    costs = compute_trading_cost(w.loc[pr.index], cost_bps, short_bps)
    pn = apply_cost(pr, costs)
    mn = compute_all_metrics(pn)
    be = cost_breakeven_analysis(pr, w)
    sub_n = compute_subperiod_metrics(pn, PERIODS)
    sub_g = compute_subperiod_metrics(pr, PERIODS)
    trade_days = (w.abs().sum(axis=1) > 0.01).sum()
    return {
        "AR": mg["AR"], "RR": mg["R/R"], "MDD": mg["MDD"],
        "net_AR": mn["AR"], "net_RR": mn["R/R"], "net_MDD": mn["MDD"],
        "BE": be["breakeven_one_way_bps"], "days": trade_days,
        "sub_g": sub_g, "sub_n": sub_n,
    }


# ================================================================
# Experiment 1: Walk-forward K
# ================================================================
def run_exp1(data: dict) -> dict:
    """Walk-forward K selection."""
    cfg = data["cfg"]
    logger.info("Exp1: Walk-forward K selection...")

    signal_wf, k_series = walk_forward_monthly_k(
        data["all_cc"], data["z_all"], data["jp_oc"], cfg,
        k_candidates=[3, 4, 5], train_months=24,
    )

    # Baseline: K=3 fixed
    signal_k3 = data["signals"]["PCA_SUB"]
    strength_k3 = compute_signal_strength(signal_k3)
    thresh_k3 = np.nanpercentile(strength_k3, 90)
    w_k3 = build_conditional_weights(signal_k3, strength_k3, thresh_k3, cfg.strategy.quantile)

    # WF-K
    strength_wf = compute_signal_strength(signal_wf)
    thresh_wf = np.nanpercentile(strength_wf, 90)
    w_wf = build_conditional_weights(signal_wf, strength_wf, thresh_wf, cfg.strategy.quantile)

    r_k3 = _eval(w_k3, data["stock_oc"], cfg.backtest.start_date, cfg.backtest.end_date)
    r_wf = _eval(w_wf, data["stock_oc"], cfg.backtest.start_date, cfg.backtest.end_date)

    # K distribution
    k_dist = k_series.value_counts().sort_index().to_dict()

    return {"baseline_k3": r_k3, "wf_k": r_wf, "k_distribution": k_dist, "k_series": k_series, "signal_wf": signal_wf}


# ================================================================
# Experiment 2: Composite filters
# ================================================================
def run_exp2(data: dict) -> dict:
    """Composite signal filters."""
    cfg = data["cfg"]
    logger.info("Exp2: Composite filters...")

    signal = data["signals"]["PCA_SUB"]
    strength = compute_signal_strength(signal)
    spread = compute_long_short_spread(signal, cfg.strategy.quantile)
    factor_str = compute_factor_strength(
        data["z_all"], data["model"], cfg.strategy.window_length,
        cfg.backtest.start_date, cfg.backtest.end_date, data["n_us"], data["n_jp"],
    )

    train_s, train_e = "2015-01-01", "2019-12-31"
    results = {}

    filters = {
        "A (strength only)": {"use_spread": False, "use_factor": False},
        "B (strength+spread)": {"use_spread": True, "use_factor": False},
        "C (strength+factor)": {"use_spread": False, "use_factor": True},
        "D (all combined)": {"use_spread": True, "use_factor": True},
    }

    for name, params in filters.items():
        mask = apply_composite_filter(
            signal, strength, spread, factor_str,
            strength_pct=90, spread_pct=70, factor_pct=70,
            train_start=train_s, train_end=train_e, **params,
        )
        ideal = build_long_short_weights(signal, cfg.strategy.quantile)
        w = ideal.copy()
        w.loc[~mask] = 0.0
        r = _eval(w, data["stock_oc"], cfg.backtest.start_date, cfg.backtest.end_date)
        results[name] = r

    return results


# ================================================================
# Experiment 3: Weight schemes
# ================================================================
def run_exp3(data: dict) -> dict:
    """Score-proportional weight schemes."""
    cfg = data["cfg"]
    logger.info("Exp3: Weight schemes...")

    signal = data["signals"]["PCA_SUB"]
    strength = compute_signal_strength(signal)
    thresh = np.nanpercentile(strength, 90)

    # Rolling vol for Weight C
    _, rolling_vol = rolling_mean_std(data["jp_cc"], cfg.strategy.window_length)

    results = {}

    # Weight A: equal weight
    w_a = build_long_short_weights(signal, cfg.strategy.quantile)
    w_a.loc[strength < thresh] = 0.0
    results["A (equal)"] = _eval(w_a, data["stock_oc"], cfg.backtest.start_date, cfg.backtest.end_date)

    # Weight B: rank proportional
    w_b = build_rank_proportional_weights(signal, cfg.strategy.quantile, max_weight=0.35)
    w_b.loc[strength < thresh] = 0.0
    results["B (rank)"] = _eval(w_b, data["stock_oc"], cfg.backtest.start_date, cfg.backtest.end_date)

    # Weight C: signal/vol proportional
    jp_cols = [c for c in signal.columns if c in rolling_vol.columns]
    vol_aligned = rolling_vol[jp_cols].reindex(signal.index)
    w_c = build_signal_vol_proportional_weights(signal[jp_cols], vol_aligned, cfg.strategy.quantile, max_weight=0.35)
    w_c.loc[strength < thresh] = 0.0
    results["C (sig/vol)"] = _eval(w_c, data["stock_oc"], cfg.backtest.start_date, cfg.backtest.end_date)

    # Also try q=0.2 variants
    w_a2 = build_long_short_weights(signal, 0.2)
    w_a2.loc[strength < thresh] = 0.0
    results["A (equal q0.2)"] = _eval(w_a2, data["stock_oc"], cfg.backtest.start_date, cfg.backtest.end_date)

    w_b2 = build_rank_proportional_weights(signal, 0.2, max_weight=0.35)
    w_b2.loc[strength < thresh] = 0.0
    results["B (rank q0.2)"] = _eval(w_b2, data["stock_oc"], cfg.backtest.start_date, cfg.backtest.end_date)

    return results


# ================================================================
# Final integration
# ================================================================
def run_final(data: dict, exp1_result: dict, exp2_result: dict, exp3_result: dict) -> dict:
    """Run best 1-2 combinations."""
    cfg = data["cfg"]
    logger.info("Final integration...")

    results = {}

    # Combo 1: WF-K + Filter A + rank weight q=0.2
    signal_wf = exp1_result["signal_wf"]
    strength_wf = compute_signal_strength(signal_wf)
    thresh_wf = np.nanpercentile(strength_wf, 90)
    w_combo1 = build_rank_proportional_weights(signal_wf, 0.2, max_weight=0.35)
    w_combo1.loc[strength_wf < thresh_wf] = 0.0
    results["WF-K + rank q0.2"] = _eval(w_combo1, data["stock_oc"], cfg.backtest.start_date, cfg.backtest.end_date)

    # Combo 2: WF-K + Filter A + equal q=0.2
    w_combo2 = build_long_short_weights(signal_wf, 0.2)
    w_combo2.loc[strength_wf < thresh_wf] = 0.0
    results["WF-K + equal q0.2"] = _eval(w_combo2, data["stock_oc"], cfg.backtest.start_date, cfg.backtest.end_date)

    # Combo 3: K=3 fixed + best filter + best weight (as sanity check)
    signal_k3 = data["signals"]["PCA_SUB"]
    strength_k3 = compute_signal_strength(signal_k3)
    thresh_k3 = np.nanpercentile(strength_k3, 90)
    w_combo3 = build_rank_proportional_weights(signal_k3, 0.2, max_weight=0.35)
    w_combo3.loc[strength_k3 < thresh_k3] = 0.0
    results["K3 + rank q0.2"] = _eval(w_combo3, data["stock_oc"], cfg.backtest.start_date, cfg.backtest.end_date)

    return results


def run_all_final_judgment(config_path: str) -> dict:
    """Run all Phase 4 experiments and return results."""
    cfg = load_config(config_path)
    data = _prepare(cfg)

    r1 = run_exp1(data)
    r2 = run_exp2(data)
    r3 = run_exp3(data)
    r_final = run_final(data, r1, r2, r3)

    return {"exp1": r1, "exp2": r2, "exp3": r3, "final": r_final}
