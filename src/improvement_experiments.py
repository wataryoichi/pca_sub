"""Focused improvement experiments to push Net AR from 4% toward 6-8%.

Three experiments:
1. Signal filter improvement (beyond simple L2 norm)
2. Portfolio construction improvement (score-proportional weights, concentration)
3. Execution target optimization (fewer stocks, liquid sectors only)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .basket_backtest import _distribute_etf_weights_to_stocks, _load_stock_prices
from .calendar_align import build_aligned_dataset
from .conditional_trading import compute_signal_strength
from .config import Config, load_config
from .constants import JP_TICKERS, N_JP, N_US
from .cost_model import apply_cost, compute_trading_cost, cost_breakeven_analysis
from .covariance import compute_cfull, compute_correlation_matrix, regularize_correlation
from .data_loader import build_price_panels, load_raw_data
from .metrics import compute_all_metrics, compute_subperiod_metrics
from .pca_plain import eigen_decompose, split_us_jp_loadings
from .pca_sub import PCASubModel
from .portfolio import build_long_short_weights
from .signal_builder import build_all_signals
from .standardize import rolling_standardize
from .stock_basket import SECTOR_STOCK_MAPPING

logger = logging.getLogger(__name__)

# Periods for OOS evaluation
PERIODS = {
    "Full": ("2015-01-01", "2025-12-31"),
    "Train": ("2015-01-01", "2019-12-31"),
    "Valid": ("2020-01-01", "2022-12-31"),
    "Test": ("2023-01-01", "2025-12-31"),
}


def _prepare_data(cfg: Config):
    """Common data preparation pipeline. Returns all needed objects."""
    raw_data = load_raw_data(cfg.data.raw_dir)
    us_close, jp_close, jp_open, _ = build_price_panels(raw_data)
    date_map, us_close_a, jp_close_a, jp_open_a = build_aligned_dataset(
        us_close, jp_close, jp_open
    )
    us_cc = us_close_a / us_close_a.shift(1) - 1
    us_cc.index = date_map["us_date"].values
    jp_cc = jp_close_a / jp_close_a.shift(1) - 1
    jp_cc.index = date_map["us_date"].values
    jp_oc = pd.DataFrame(
        jp_close_a.values / jp_open_a.values - 1,
        index=pd.to_datetime(date_map["us_date"].values),
        columns=jp_close.columns,
    )
    all_cc = pd.concat([us_cc, jp_cc], axis=1)
    all_cc.index = pd.to_datetime(all_cc.index)

    Cfull = compute_cfull(all_cc, cfg.cfull.start_date, cfg.cfull.end_date, cfg.strategy.window_length)
    tickers_all = list(us_cc.columns) + list(jp_cc.columns)

    signals = build_all_signals(
        us_cc, jp_cc, cfg.strategy.window_length, cfg.strategy.n_components,
        cfg.strategy.lambda_reg, Cfull, cfg.backtest.start_date, cfg.backtest.end_date,
        tickers_all,
    )

    # Stock data
    stock_close, stock_open = _load_stock_prices("data/raw/stocks")
    jp_dates = date_map["jp_date"].values
    us_dates = pd.to_datetime(date_map["us_date"].values)
    stock_close_a = stock_close.reindex(jp_dates)
    stock_open_a = stock_open.reindex(jp_dates)
    stock_oc = pd.DataFrame(
        stock_close_a.values / stock_open_a.values - 1,
        index=us_dates, columns=stock_close.columns,
    )

    # Factor scores for f2 filter
    z_all = rolling_standardize(all_cc, window=cfg.strategy.window_length)
    n_us = len(us_cc.columns)
    n_jp = len(jp_cc.columns)
    model = PCASubModel(
        Cfull=Cfull, tickers=tickers_all, n_us=n_us, n_jp=n_jp,
        lambda_reg=cfg.strategy.lambda_reg, n_components=cfg.strategy.n_components,
    )

    return {
        "signals": signals,
        "jp_oc": jp_oc,
        "stock_oc": stock_oc,
        "z_all": z_all,
        "model": model,
        "n_us": n_us,
        "n_jp": n_jp,
        "cfg": cfg,
        "date_map": date_map,
    }


def _eval_basket(
    etf_weights: pd.DataFrame,
    stock_oc: pd.DataFrame,
    start: str,
    end: str,
    cost_bps: float = 3.0,
    short_bps: float = 2.0,
) -> dict:
    """Evaluate a basket strategy with stock execution."""
    stock_weights = _distribute_etf_weights_to_stocks(etf_weights)
    common_dates = stock_weights.index.intersection(stock_oc.index)
    common_dates = common_dates[(common_dates >= start) & (common_dates <= end)]
    common_cols = stock_weights.columns.intersection(stock_oc.columns)

    w = stock_weights.loc[common_dates, common_cols]
    r = stock_oc.loc[common_dates, common_cols].fillna(0)
    port_ret = (w * r).sum(axis=1).dropna()

    if len(port_ret) < 10:
        return {"gross_AR": np.nan, "gross_RR": np.nan, "net_AR": np.nan, "net_RR": np.nan,
                "MDD": np.nan, "breakeven": np.nan, "trade_days": 0}

    mg = compute_all_metrics(port_ret)
    costs = compute_trading_cost(w.loc[port_ret.index], cost_bps, short_bps)
    pn = apply_cost(port_ret, costs)
    mn = compute_all_metrics(pn)
    be = cost_breakeven_analysis(port_ret, w)

    trade_days = (w.abs().sum(axis=1) > 0.01).sum()

    # Sub-period
    sub_g = compute_subperiod_metrics(port_ret, PERIODS)
    sub_n = compute_subperiod_metrics(pn, PERIODS)

    return {
        "gross_AR": mg["AR"], "gross_RR": mg["R/R"],
        "net_AR": mn["AR"], "net_RR": mn["R/R"], "MDD": mn["MDD"],
        "breakeven": be["breakeven_one_way_bps"],
        "trade_days": trade_days,
        "sub_gross": sub_g, "sub_net": sub_n,
    }


# ============================================================
# Experiment 1: Filter improvements
# ============================================================

def _compute_f2_score(z_all, model, window, start, end, n_us, n_jp):
    """Compute daily f2 (cyclical/defensive) factor score magnitude."""
    dates = z_all.loc[start:end].index
    f2_abs = pd.Series(np.nan, index=dates)

    for date in dates:
        loc = z_all.index.get_loc(date)
        if loc < window:
            continue
        z_window = z_all.iloc[loc - window:loc].fillna(0.0)
        if len(z_window) < window:
            continue
        row = z_all.loc[date].fillna(0.0).values
        z_us = row[:n_us]

        Ct = compute_correlation_matrix(z_window.values)
        C_reg = regularize_correlation(Ct, model.C0, model.lambda_reg)
        _, V_K = eigen_decompose(C_reg, model.n_components)
        V_U, V_J = split_us_jp_loadings(V_K, n_us, n_jp)
        f = V_U.T @ z_us
        if len(f) >= 3:
            f2_abs.loc[date] = abs(f[2])

    return f2_abs


def _compute_spread(signal):
    """Top-bottom spread as filter: max(signal) - min(signal) per day."""
    return signal.max(axis=1) - signal.min(axis=1)


def experiment_filters(data: dict) -> pd.DataFrame:
    """Compare different signal filter methods."""
    signal = data["signals"]["PCA_SUB"]
    jp_oc = data["jp_oc"]
    stock_oc = data["stock_oc"]
    cfg = data["cfg"]
    q = cfg.strategy.quantile

    # Filter candidates
    filters = {}

    # 1. Baseline: L2 norm
    filters["L2_norm"] = compute_signal_strength(signal)

    # 2. f2 score (cyclical/defensive factor only)
    filters["f2_score"] = _compute_f2_score(
        data["z_all"], data["model"], cfg.strategy.window_length,
        cfg.backtest.start_date, cfg.backtest.end_date, data["n_us"], data["n_jp"],
    )

    # 3. Top-bottom spread
    filters["spread"] = _compute_spread(signal)

    # 4. PCA_SUB × PCA_PLAIN agreement: trade only when both agree on top/bottom
    plain_sig = data["signals"]["PCA_PLAIN"]
    sub_rank = signal.rank(axis=1, pct=True)
    plain_rank = plain_sig.rank(axis=1, pct=True)
    # Agreement score: correlation of ranks per day
    agreement = sub_rank.corrwith(plain_rank, axis=1)
    filters["agreement"] = agreement

    records = []
    for filter_name, strength in filters.items():
        for pct in [0, 70, 80, 90]:
            valid_strength = strength.dropna()
            if len(valid_strength) < 20:
                continue
            thresh = np.nanpercentile(valid_strength, pct)

            # Build conditional weights
            ideal = build_long_short_weights(signal, q)
            mask = strength >= thresh
            weights = ideal.copy()
            weights.loc[~mask.reindex(weights.index, fill_value=False)] = 0.0

            result = _eval_basket(weights, stock_oc, cfg.backtest.start_date, cfg.backtest.end_date)

            # Test period only
            test_net = result["sub_net"].get("Test", {})
            test_gross = result["sub_gross"].get("Test", {})

            records.append({
                "filter": filter_name,
                "percentile": pct,
                "full_net_AR": result["net_AR"],
                "full_net_RR": result["net_RR"],
                "test_net_AR": test_net.get("AR", np.nan),
                "test_net_RR": test_net.get("R/R", np.nan),
                "test_gross_AR": test_gross.get("AR", np.nan),
                "breakeven": result["breakeven"],
                "trade_days": result["trade_days"],
            })

    return pd.DataFrame(records)


# ============================================================
# Experiment 2: Portfolio construction
# ============================================================

def _build_score_proportional_weights(signal, quantile=0.3):
    """Score-proportional weights instead of equal weight."""
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

        # Score-proportional: weight proportional to |score|
        long_scores = row[long_t].abs()
        short_scores = row[short_t].abs()

        if long_scores.sum() > 1e-12:
            weights.loc[date, long_t] = long_scores / long_scores.sum()
        if short_scores.sum() > 1e-12:
            weights.loc[date, short_t] = -short_scores / short_scores.sum()

    return weights


def experiment_portfolio(data: dict) -> pd.DataFrame:
    """Compare portfolio construction methods."""
    signal = data["signals"]["PCA_SUB"]
    stock_oc = data["stock_oc"]
    cfg = data["cfg"]
    strength = compute_signal_strength(signal)

    records = []
    constructions = {}

    # Apply 90th percentile filter to all
    thresh_90 = np.nanpercentile(strength, 90)
    mask_90 = strength >= thresh_90

    for q_val in [0.1, 0.2, 0.3]:
        # Equal weight
        ideal_eq = build_long_short_weights(signal, q_val)
        w_eq = ideal_eq.copy()
        w_eq.loc[~mask_90] = 0.0
        constructions[f"equal_q{q_val}"] = w_eq

    # Score proportional, q=0.3
    ideal_sp = _build_score_proportional_weights(signal, 0.3)
    w_sp = ideal_sp.copy()
    w_sp.loc[~mask_90] = 0.0
    constructions["score_prop_q0.3"] = w_sp

    # Score proportional, q=0.2
    ideal_sp2 = _build_score_proportional_weights(signal, 0.2)
    w_sp2 = ideal_sp2.copy()
    w_sp2.loc[~mask_90] = 0.0
    constructions["score_prop_q0.2"] = w_sp2

    for name, weights in constructions.items():
        result = _eval_basket(weights, stock_oc, cfg.backtest.start_date, cfg.backtest.end_date)
        test_net = result["sub_net"].get("Test", {})
        test_gross = result["sub_gross"].get("Test", {})
        records.append({
            "construction": name,
            "full_net_AR": result["net_AR"],
            "full_net_RR": result["net_RR"],
            "test_net_AR": test_net.get("AR", np.nan),
            "test_net_RR": test_net.get("R/R", np.nan),
            "test_gross_AR": test_gross.get("AR", np.nan),
            "breakeven": result["breakeven"],
        })

    return pd.DataFrame(records)


# ============================================================
# Experiment 3: Execution target optimization
# ============================================================

def _distribute_etf_weights_top_n(etf_weights, n_per_sector=2):
    """Distribute to only top N stocks per sector (by hardcoded liquidity rank)."""
    stock_cols = []
    for sector in SECTOR_STOCK_MAPPING.values():
        for stock in sector["stocks"][:n_per_sector]:
            stock_cols.append(stock["ticker"])

    stock_weights = pd.DataFrame(0.0, index=etf_weights.index, columns=stock_cols)
    for etf_ticker, sector in SECTOR_STOCK_MAPPING.items():
        if etf_ticker not in etf_weights.columns:
            continue
        stocks = [s["ticker"] for s in sector["stocks"][:n_per_sector]]
        n = len(stocks)
        for s in stocks:
            if s in stock_weights.columns:
                stock_weights[s] = etf_weights[etf_ticker] / n
    return stock_weights


# Liquid sectors (avg daily turnover > 50M JPY for ETF, or top stocks very liquid)
LIQUID_SECTORS = ["1622.T", "1625.T", "1629.T", "1631.T", "1632.T", "1621.T", "1627.T"]


def experiment_execution(data: dict) -> pd.DataFrame:
    """Compare execution target configurations."""
    signal = data["signals"]["PCA_SUB"]
    stock_oc = data["stock_oc"]
    cfg = data["cfg"]
    strength = compute_signal_strength(signal)
    thresh_90 = np.nanpercentile(strength, 90)
    mask_90 = strength >= thresh_90

    ideal = build_long_short_weights(signal, cfg.strategy.quantile)
    w_filtered = ideal.copy()
    w_filtered.loc[~mask_90] = 0.0

    records = []

    # Config 1: All stocks, all sectors (baseline)
    sw_all = _distribute_etf_weights_to_stocks(w_filtered)
    r1 = _eval_basket_direct(sw_all, stock_oc, cfg.backtest.start_date, cfg.backtest.end_date)
    records.append({"config": "all_stocks (baseline)", **r1})

    # Config 2: Top 2 per sector
    sw_top2 = _distribute_etf_weights_top_n(w_filtered, n_per_sector=2)
    r2 = _eval_basket_direct(sw_top2, stock_oc, cfg.backtest.start_date, cfg.backtest.end_date)
    records.append({"config": "top2_per_sector", **r2})

    # Config 3: Top 1 per sector
    sw_top1 = _distribute_etf_weights_top_n(w_filtered, n_per_sector=1)
    r3 = _eval_basket_direct(sw_top1, stock_oc, cfg.backtest.start_date, cfg.backtest.end_date)
    records.append({"config": "top1_per_sector", **r3})

    # Config 4: Liquid sectors only (all stocks)
    w_liquid = w_filtered.copy()
    non_liquid = [t for t in w_liquid.columns if t not in LIQUID_SECTORS]
    w_liquid[non_liquid] = 0.0
    # Renormalize
    for date in w_liquid.index:
        row = w_liquid.loc[date]
        pos = row[row > 0]
        neg = row[row < 0]
        if pos.sum() > 1e-12:
            w_liquid.loc[date, pos.index] = pos / pos.sum()
        if neg.sum() < -1e-12:
            w_liquid.loc[date, neg.index] = neg / neg.abs().sum()
    sw_liquid = _distribute_etf_weights_to_stocks(w_liquid)
    r4 = _eval_basket_direct(sw_liquid, stock_oc, cfg.backtest.start_date, cfg.backtest.end_date)
    records.append({"config": "liquid_sectors_only", **r4})

    # Config 5: Lower cost assumption (2bp for largest stocks)
    r5 = _eval_basket_direct(sw_top1, stock_oc, cfg.backtest.start_date, cfg.backtest.end_date, cost_bps=2.0, short_bps=1.0)
    records.append({"config": "top1_2bp_cost", **r5})

    return pd.DataFrame(records)


def _eval_basket_direct(stock_weights, stock_oc, start, end, cost_bps=3.0, short_bps=2.0):
    """Evaluate with already-distributed stock weights."""
    common_dates = stock_weights.index.intersection(stock_oc.index)
    common_dates = common_dates[(common_dates >= start) & (common_dates <= end)]
    common_cols = stock_weights.columns.intersection(stock_oc.columns)

    w = stock_weights.loc[common_dates, common_cols]
    r = stock_oc.loc[common_dates, common_cols].fillna(0)
    port_ret = (w * r).sum(axis=1).dropna()

    if len(port_ret) < 10:
        return {"full_net_AR": np.nan, "full_net_RR": np.nan, "test_net_AR": np.nan,
                "test_net_RR": np.nan, "breakeven": np.nan}

    mg = compute_all_metrics(port_ret)
    costs = compute_trading_cost(w.loc[port_ret.index], cost_bps, short_bps)
    pn = apply_cost(port_ret, costs)
    mn = compute_all_metrics(pn)
    be = cost_breakeven_analysis(port_ret, w)

    sub_n = compute_subperiod_metrics(pn, PERIODS)
    sub_g = compute_subperiod_metrics(port_ret, PERIODS)
    test_n = sub_n.get("Test", {})
    test_g = sub_g.get("Test", {})

    return {
        "full_net_AR": mn["AR"], "full_net_RR": mn["R/R"],
        "test_net_AR": test_n.get("AR", np.nan),
        "test_net_RR": test_n.get("R/R", np.nan),
        "test_gross_AR": test_g.get("AR", np.nan),
        "breakeven": be["breakeven_one_way_bps"],
    }
