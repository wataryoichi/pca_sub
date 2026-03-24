"""Backtest engine: orchestrates signal -> portfolio -> returns -> metrics."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config
from .constants import ALL_TICKERS, JP_TICKERS, US_TICKERS
from .covariance import compute_cfull
from .data_loader import build_price_panels, load_raw_data
from .calendar_align import align_us_jp_dates, build_aligned_dataset
from .metrics import compute_all_metrics
from .portfolio import build_long_short_weights, compute_portfolio_returns, compute_turnover
from .returns import close_to_close_return, open_to_close_return
from .signal_builder import build_all_signals

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Container for backtest outputs."""
    strategy_name: str
    daily_returns: pd.Series
    weights: pd.DataFrame
    signal: pd.DataFrame
    metrics: dict
    turnover: pd.Series


def run_backtest(cfg: Config) -> dict[str, BacktestResult]:
    """Run full backtest pipeline.

    Returns:
        Dict mapping strategy name -> BacktestResult
    """
    logger.info("Loading raw data...")
    raw_data = load_raw_data(cfg.data.raw_dir)
    if not raw_data:
        raise RuntimeError(f"No data found in {cfg.data.raw_dir}")

    logger.info(f"Loaded {len(raw_data)} tickers")
    us_close, jp_close, jp_open, volume = build_price_panels(raw_data)

    # Compute returns
    logger.info("Computing returns...")
    us_cc = close_to_close_return(us_close)
    jp_cc = close_to_close_return(jp_close)

    # Align US-JP calendar
    logger.info("Aligning US-JP calendars...")
    date_map, us_close_a, jp_close_a, jp_open_a = build_aligned_dataset(
        us_close, jp_close, jp_open
    )

    # Build aligned returns
    # US cc returns: use the aligned US close prices
    us_cc_aligned = us_close_a / us_close_a.shift(1) - 1
    us_cc_aligned.index = date_map["us_date"].values

    # JP cc returns aligned (for MOM and standardization)
    jp_cc_aligned = jp_close_a / jp_close_a.shift(1) - 1
    jp_cc_aligned.index = date_map["us_date"].values

    # JP open-to-close returns (what we're predicting / trading)
    jp_oc_aligned = jp_close_a.values / jp_open_a.values - 1
    jp_oc_aligned = pd.DataFrame(
        jp_oc_aligned,
        index=date_map["us_date"].values,
        columns=jp_close.columns,
    )

    # Combine for joint standardization
    all_cc = pd.concat([us_cc_aligned, jp_cc_aligned], axis=1)
    all_cc.index = pd.to_datetime(all_cc.index)
    jp_oc_aligned.index = pd.to_datetime(jp_oc_aligned.index)

    # Compute Cfull
    logger.info(f"Computing Cfull ({cfg.cfull.start_date} to {cfg.cfull.end_date})...")
    Cfull = compute_cfull(
        all_cc,
        start_date=cfg.cfull.start_date,
        end_date=cfg.cfull.end_date,
        window=cfg.strategy.window_length,
    )

    tickers_all = list(us_cc_aligned.columns) + list(jp_cc_aligned.columns)

    # Build signals
    logger.info("Building signals...")
    signals = build_all_signals(
        us_cc_returns=us_cc_aligned,
        jp_cc_returns=jp_cc_aligned,
        window=cfg.strategy.window_length,
        n_components=cfg.strategy.n_components,
        lambda_reg=cfg.strategy.lambda_reg,
        Cfull=Cfull,
        backtest_start=cfg.backtest.start_date,
        backtest_end=cfg.backtest.end_date,
        tickers_all=tickers_all,
    )

    # Build portfolios and compute returns
    results: dict[str, BacktestResult] = {}
    for name, sig in signals.items():
        logger.info(f"Building portfolio for {name}...")
        weights = build_long_short_weights(sig, quantile=cfg.strategy.quantile)
        port_ret = compute_portfolio_returns(weights, jp_oc_aligned)
        turnover = compute_turnover(weights)

        # Drop NaN returns
        port_ret = port_ret.dropna()

        # Filter to backtest period
        mask = (port_ret.index >= cfg.backtest.start_date) & (
            port_ret.index <= cfg.backtest.end_date
        )
        port_ret = port_ret[mask]

        metrics = compute_all_metrics(port_ret)
        results[name] = BacktestResult(
            strategy_name=name,
            daily_returns=port_ret,
            weights=weights,
            signal=sig,
            metrics=metrics,
            turnover=turnover,
        )
        logger.info(f"  {name}: AR={metrics['AR']:.2f}%, R/R={metrics['R/R']:.2f}")

    return results
