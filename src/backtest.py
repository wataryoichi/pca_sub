"""Backtest engine: orchestrates signal -> portfolio -> returns -> metrics."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config
from .constants import ALL_TICKERS, JP_TICKERS, US_TICKERS
from .cost_model import apply_cost, compute_trading_cost
from .covariance import compute_cfull
from .data_loader import build_price_panels, load_raw_data
from .calendar_align import align_us_jp_dates, build_aligned_dataset
from .liquidity import apply_liquidity_filter, build_liquidity_mask, liquidity_exclusion_report
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
    daily_returns_net: pd.Series | None
    weights: pd.DataFrame
    signal: pd.DataFrame
    metrics: dict
    metrics_net: dict | None
    turnover: pd.Series
    liquidity_report: pd.DataFrame | None = None


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

    # --- Liquidity filter ---
    illiquid_mask = None
    liq_report = None
    if cfg.liquidity.enabled:
        logger.info("Building liquidity filter...")
        # Build JP close and volume panels aligned to us_date index
        jp_close_for_liq = jp_close_a.copy()
        jp_close_for_liq.index = pd.to_datetime(date_map["us_date"].values)
        jp_vol = pd.DataFrame(
            {t: raw_data[t]["Volume"] for t in JP_TICKERS if t in raw_data}
        )
        # Align volume to the same dates
        jp_vol_aligned = jp_vol.reindex(jp_close.index)
        # Map to us_date via date_map
        jp_vol_for_liq = jp_vol_aligned.loc[date_map["jp_date"].values].copy()
        jp_vol_for_liq.index = pd.to_datetime(date_map["us_date"].values)

        illiquid_mask = build_liquidity_mask(
            close=jp_close_for_liq,
            volume=jp_vol_for_liq,
            min_avg_turnover=cfg.liquidity.min_avg_turnover_jpy,
            lookback_days=cfg.liquidity.lookback_days,
        )
        liq_report = liquidity_exclusion_report(
            illiquid_mask.loc[cfg.backtest.start_date : cfg.backtest.end_date]
        )
        avg_excluded = liq_report["n_excluded"].mean()
        logger.info(f"  Average tickers excluded per day: {avg_excluded:.1f}")

    # Build portfolios and compute returns
    results: dict[str, BacktestResult] = {}
    for name, sig in signals.items():
        logger.info(f"Building portfolio for {name}...")

        # Apply liquidity filter to signals
        sig_filtered = sig
        if cfg.liquidity.enabled and illiquid_mask is not None:
            sig_filtered = apply_liquidity_filter(sig, illiquid_mask)

        weights = build_long_short_weights(sig_filtered, quantile=cfg.strategy.quantile)
        port_ret = compute_portfolio_returns(weights, jp_oc_aligned)
        turnover = compute_turnover(weights)

        # Drop NaN returns
        port_ret = port_ret.dropna()

        # Filter to backtest period
        mask = (port_ret.index >= cfg.backtest.start_date) & (
            port_ret.index <= cfg.backtest.end_date
        )
        port_ret = port_ret[mask]

        metrics_gross = compute_all_metrics(port_ret)

        # Apply costs
        daily_returns_net = None
        metrics_net = None
        if cfg.cost.enabled:
            costs = compute_trading_cost(
                weights=weights.loc[port_ret.index],
                one_way_bps=cfg.cost.one_way_bps,
                short_extra_bps=cfg.cost.short_extra_bps,
                illiquid_mask=illiquid_mask,
                illiquid_extra_bps=cfg.cost.illiquid_extra_bps,
            )
            daily_returns_net = apply_cost(port_ret, costs)
            metrics_net = compute_all_metrics(daily_returns_net)
            logger.info(
                f"  {name} GROSS: AR={metrics_gross['AR']:.2f}%, R/R={metrics_gross['R/R']:.2f}"
            )
            logger.info(
                f"  {name} NET:   AR={metrics_net['AR']:.2f}%, R/R={metrics_net['R/R']:.2f}"
            )
        else:
            logger.info(f"  {name}: AR={metrics_gross['AR']:.2f}%, R/R={metrics_gross['R/R']:.2f}")

        results[name] = BacktestResult(
            strategy_name=name,
            daily_returns=port_ret,
            daily_returns_net=daily_returns_net,
            weights=weights,
            signal=sig_filtered,
            metrics=metrics_gross,
            metrics_net=metrics_net,
            turnover=turnover,
            liquidity_report=liq_report,
        )

    return results
