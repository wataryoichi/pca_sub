"""Backtest engine for individual stock basket execution.

Uses ETF-level PCA_SUB signals, then distributes weights to
representative individual stocks within each sector.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .calendar_align import build_aligned_dataset
from .conditional_trading import build_conditional_weights, compute_signal_strength
from .config import Config
from .constants import JP_TICKERS
from .cost_model import apply_cost, compute_trading_cost, cost_breakeven_analysis
from .covariance import compute_cfull
from .data_loader import build_price_panels, load_raw_data
from .metrics import compute_all_metrics, compute_subperiod_metrics
from .portfolio import build_long_short_weights
from .signal_builder import build_all_signals
from .stock_basket import SECTOR_STOCK_MAPPING

logger = logging.getLogger(__name__)


def _load_stock_prices(
    stock_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load individual stock open/close prices.

    Returns:
        stock_close: (date x stock_tickers)
        stock_open: (date x stock_tickers)
    """
    stock_dir = Path(stock_dir)
    close_dict = {}
    open_dict = {}
    for sector in SECTOR_STOCK_MAPPING.values():
        for stock in sector["stocks"]:
            t = stock["ticker"]
            path = stock_dir / f"{t}.csv"
            if not path.exists():
                logger.warning(f"Stock data not found: {path}")
                continue
            df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
            close_dict[t] = df["Close"]
            open_dict[t] = df["Open"]
    return pd.DataFrame(close_dict), pd.DataFrame(open_dict)


def _distribute_etf_weights_to_stocks(
    etf_weights: pd.DataFrame,
) -> pd.DataFrame:
    """Convert ETF-level weights to individual stock weights.

    Each ETF's weight is split equally among its representative stocks.
    """
    stock_cols = []
    for sector in SECTOR_STOCK_MAPPING.values():
        for stock in sector["stocks"]:
            stock_cols.append(stock["ticker"])

    stock_weights = pd.DataFrame(0.0, index=etf_weights.index, columns=stock_cols)

    for etf_ticker, sector in SECTOR_STOCK_MAPPING.items():
        if etf_ticker not in etf_weights.columns:
            continue
        stocks = [s["ticker"] for s in sector["stocks"]]
        n = len(stocks)
        for s in stocks:
            if s in stock_weights.columns:
                stock_weights[s] = etf_weights[etf_ticker] / n

    return stock_weights


def run_basket_backtest(
    cfg: Config,
    stock_dir: str = "data/raw/stocks",
    signal_filter_pct: float | None = None,
    cost_one_way_bps: float = 3.0,
    cost_short_bps: float = 2.0,
) -> dict:
    """Run backtest using individual stock basket execution.

    Args:
        cfg: Configuration
        stock_dir: Directory with individual stock CSVs
        signal_filter_pct: If set, only trade on days where signal strength
            is above this percentile (0-100). None = trade every day.
        cost_one_way_bps: One-way cost (lower than ETF due to better liquidity)
        cost_short_bps: Short extra cost

    Returns:
        Dict with results for PCA_SUB strategy
    """
    # --- ETF-level signal generation (same as standard backtest) ---
    raw_data = load_raw_data(cfg.data.raw_dir)
    us_close, jp_close, jp_open, _ = build_price_panels(raw_data)
    date_map, us_close_a, jp_close_a, jp_open_a = build_aligned_dataset(
        us_close, jp_close, jp_open
    )

    us_cc = us_close_a / us_close_a.shift(1) - 1
    us_cc.index = date_map["us_date"].values
    jp_cc = jp_close_a / jp_close_a.shift(1) - 1
    jp_cc.index = date_map["us_date"].values

    all_cc = pd.concat([us_cc, jp_cc], axis=1)
    all_cc.index = pd.to_datetime(all_cc.index)

    Cfull = compute_cfull(
        all_cc, cfg.cfull.start_date, cfg.cfull.end_date, cfg.strategy.window_length
    )
    tickers_all = list(us_cc.columns) + list(jp_cc.columns)
    signals = build_all_signals(
        us_cc, jp_cc, cfg.strategy.window_length, cfg.strategy.n_components,
        cfg.strategy.lambda_reg, Cfull,
        cfg.backtest.start_date, cfg.backtest.end_date, tickers_all,
    )
    pca_sub_signal = signals["PCA_SUB"]

    # --- ETF-level weights ---
    if signal_filter_pct is not None and signal_filter_pct > 0:
        strength = compute_signal_strength(pca_sub_signal)
        thresh = np.nanpercentile(strength, signal_filter_pct)
        etf_weights = build_conditional_weights(
            pca_sub_signal, strength, thresh, cfg.strategy.quantile
        )
    else:
        etf_weights = build_long_short_weights(pca_sub_signal, cfg.strategy.quantile)

    # --- Distribute to individual stocks ---
    stock_weights = _distribute_etf_weights_to_stocks(etf_weights)

    # --- Load stock prices and compute OC returns ---
    stock_close, stock_open = _load_stock_prices(stock_dir)

    # Align stock prices to the date_map JP dates
    # Stock prices are indexed by JP calendar dates
    # We need OC returns on jp_date, indexed by us_date
    jp_dates = date_map["jp_date"].values
    us_dates = pd.to_datetime(date_map["us_date"].values)

    stock_close_aligned = stock_close.reindex(jp_dates)
    stock_open_aligned = stock_open.reindex(jp_dates)

    stock_oc = stock_close_aligned.values / stock_open_aligned.values - 1
    stock_oc = pd.DataFrame(stock_oc, index=us_dates, columns=stock_close.columns)

    # --- Compute portfolio returns ---
    common_dates = stock_weights.index.intersection(stock_oc.index)
    common_dates = common_dates[
        (common_dates >= cfg.backtest.start_date) & (common_dates <= cfg.backtest.end_date)
    ]

    w = stock_weights.loc[common_dates]
    r = stock_oc.loc[common_dates]

    # Align columns
    common_cols = w.columns.intersection(r.columns)
    w = w[common_cols]
    r = r[common_cols].fillna(0)

    port_ret = (w * r).sum(axis=1).dropna()

    # --- Metrics ---
    metrics_gross = compute_all_metrics(port_ret)
    be = cost_breakeven_analysis(port_ret, w)

    costs = compute_trading_cost(
        w.loc[port_ret.index],
        one_way_bps=cost_one_way_bps,
        short_extra_bps=cost_short_bps,
    )
    port_ret_net = apply_cost(port_ret, costs)
    metrics_net = compute_all_metrics(port_ret_net)

    # Sub-period metrics
    periods = {
        "Full": (cfg.backtest.start_date, cfg.backtest.end_date),
        "Train (2015-2019)": ("2015-01-01", "2019-12-31"),
        "Valid (2020-2022)": ("2020-01-01", "2022-12-31"),
        "Test (2023-2025)": ("2023-01-01", "2025-12-31"),
    }
    sub_gross = compute_subperiod_metrics(port_ret, periods)
    sub_net = compute_subperiod_metrics(port_ret_net, periods)

    return {
        "port_ret": port_ret,
        "port_ret_net": port_ret_net,
        "metrics_gross": metrics_gross,
        "metrics_net": metrics_net,
        "breakeven": be,
        "sub_gross": sub_gross,
        "sub_net": sub_net,
        "stock_weights": w,
    }
