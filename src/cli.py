"""CLI interface for PCA_SUB project."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer

app = typer.Typer(help="PCA_SUB: Lead-lag strategy reproduction and validation")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


@app.command()
def fetch_data(
    config: str = typer.Option("configs/base.yaml", "--config", "-c"),
) -> None:
    """Fetch price data for all tickers."""
    from .config import load_config
    from .constants import ALL_TICKERS
    from .data_loader import fetch_and_save, get_provider

    cfg = load_config(config)
    provider = get_provider(cfg.data.provider)
    fetch_and_save(
        provider=provider,
        tickers=ALL_TICKERS,
        start=cfg.data.start_date,
        end=cfg.data.end_date,
        output_dir=cfg.data.raw_dir,
    )
    typer.echo(f"Data saved to {cfg.data.raw_dir}/")


@app.command()
def run_backtest(
    config: str = typer.Option("configs/paper_reproduction.yaml", "--config", "-c"),
) -> None:
    """Run backtest for all strategies."""
    from .backtest import run_backtest as _run
    from .config import load_config

    cfg = load_config(config)
    results = _run(cfg)

    # Print summary table
    has_cost = cfg.cost.enabled
    has_liq = cfg.liquidity.enabled

    typer.echo("\n" + "=" * 70)
    typer.echo(f"BACKTEST RESULTS (cost={'ON' if has_cost else 'OFF'}, liquidity={'ON' if has_liq else 'OFF'})")
    typer.echo("=" * 70)

    # Gross results
    typer.echo(f"\n{'[GROSS]':<15}")
    typer.echo(f"{'Strategy':<15} {'AR%':>8} {'RISK%':>8} {'R/R':>8} {'MDD%':>8} {'Hit%':>8}")
    typer.echo("-" * 70)
    for name, res in results.items():
        m = res.metrics
        typer.echo(
            f"{name:<15} {m['AR']:>8.2f} {m['RISK']:>8.2f} "
            f"{m['R/R']:>8.2f} {m['MDD']:>8.2f} {m['Hit Ratio']*100:>8.1f}"
        )

    # Net results (if cost enabled)
    if has_cost:
        typer.echo(f"\n{'[NET]':<15} (cost: {cfg.cost.one_way_bps}bp one-way + {cfg.cost.short_extra_bps}bp short)")
        typer.echo(f"{'Strategy':<15} {'AR%':>8} {'RISK%':>8} {'R/R':>8} {'MDD%':>8} {'Hit%':>8}")
        typer.echo("-" * 70)
        for name, res in results.items():
            if res.metrics_net:
                m = res.metrics_net
                typer.echo(
                    f"{name:<15} {m['AR']:>8.2f} {m['RISK']:>8.2f} "
                    f"{m['R/R']:>8.2f} {m['MDD']:>8.2f} {m['Hit Ratio']*100:>8.1f}"
                )

    # Breakeven analysis
    typer.echo("\n[BREAKEVEN ANALYSIS]")
    from .cost_model import cost_breakeven_analysis
    for name, res in results.items():
        be = cost_breakeven_analysis(res.daily_returns, res.weights)
        typer.echo(
            f"  {name}: breakeven one-way = {be['breakeven_one_way_bps']:.1f} bp, "
            f"avg daily gross = {be['avg_daily_gross_return_bps']:.1f} bp"
        )
    typer.echo("=" * 70)

    # Save results
    _save_results(cfg, results)


@app.command()
def generate_report(
    config: str = typer.Option("configs/paper_reproduction.yaml", "--config", "-c"),
) -> None:
    """Generate report from saved results."""
    from .config import load_config
    from .report import generate_report as _gen

    cfg = load_config(config)
    _gen(cfg)
    typer.echo(f"Report saved to {cfg.output.reports_dir}/")


@app.command()
def run_sensitivity(
    config: str = typer.Option("configs/paper_reproduction.yaml", "--config", "-c"),
) -> None:
    """Run parameter sensitivity analysis."""
    from .config import load_config
    from .experiments import (
        generate_sensitivity_plots,
        run_multi_sensitivity,
        save_sensitivity_results,
    )

    cfg = load_config(config)

    experiments = {
        "strategy.window_length": [20, 40, 60, 120],
        "strategy.lambda_reg": [0.0, 0.3, 0.6, 0.9, 0.95],
        "strategy.n_components": [1, 2, 3, 4, 5],
        "strategy.quantile": [0.1, 0.2, 0.3],
    }

    results = run_multi_sensitivity(config, experiments)

    out_dir = cfg.output.results_dir
    save_sensitivity_results(results, f"{out_dir}/sensitivity")
    generate_sensitivity_plots(results, f"{out_dir}/plots", cfg.output.plots_format)

    # Print summary
    for param_name, df in results.items():
        typer.echo(f"\n{'='*70}")
        typer.echo(f"Sensitivity: {param_name}")
        typer.echo(f"{'='*70}")
        pca_sub = df[df["strategy"] == "PCA_SUB"]
        if not pca_sub.empty:
            typer.echo(f"{'Value':>10} {'AR%':>8} {'R/R':>8} {'MDD%':>8} {'BE bp':>8}")
            typer.echo("-" * 50)
            for _, row in pca_sub.iterrows():
                typer.echo(
                    f"{str(row['param_value']):>10} {row['AR']:>8.2f} "
                    f"{row['R/R']:>8.2f} {row['MDD']:>8.2f} {row['breakeven_bps']:>8.1f}"
                )

    typer.echo(f"\nResults saved to {out_dir}/sensitivity/")


@app.command()
def run_oos_validation(
    config: str = typer.Option("configs/paper_reproduction.yaml", "--config", "-c"),
) -> None:
    """Run out-of-sample validation with sub-period analysis."""
    from .config import load_config
    from .experiments import run_subperiod_analysis

    cfg = load_config(config)

    periods = {
        "Full (2015-2025)": ("2015-01-01", "2025-12-31"),
        "Train (2015-2019)": ("2015-01-01", "2019-12-31"),
        "Validation (2020-2022)": ("2020-01-01", "2022-12-31"),
        "Test (2023-2025)": ("2023-01-01", "2025-12-31"),
    }

    df = run_subperiod_analysis(cfg, periods)

    # Print results
    for period_name in periods:
        sub = df[df["period"] == period_name]
        typer.echo(f"\n{'='*70}")
        typer.echo(f"{period_name}")
        typer.echo(f"{'Strategy':<15} {'AR%':>8} {'RISK%':>8} {'R/R':>8} {'MDD%':>8} {'BE bp':>8}")
        typer.echo("-" * 70)
        for _, row in sub.iterrows():
            typer.echo(
                f"{row['strategy']:<15} {row['AR']:>8.2f} {row['RISK']:>8.2f} "
                f"{row['R/R']:>8.2f} {row['MDD']:>8.2f} {row['breakeven_bps']:>8.1f}"
            )

    # Save
    out_dir = cfg.output.results_dir
    from pathlib import Path
    Path(f"{out_dir}/oos").mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{out_dir}/oos/subperiod_metrics.csv", index=False)
    typer.echo(f"\nSaved to {out_dir}/oos/")


@app.command()
def run_rebal_sensitivity(
    config: str = typer.Option("configs/paper_reproduction.yaml", "--config", "-c"),
) -> None:
    """Run rebalancing frequency sensitivity analysis."""
    from .config import load_config
    from .experiments import run_sensitivity

    cfg = load_config(config)

    # Test rebalancing frequencies: 1 (daily), 2, 3, 5, 10, 20
    rebal_values = [1, 2, 3, 5, 10, 20]
    df = run_sensitivity(cfg, "strategy.rebal_freq", rebal_values)

    typer.echo(f"\n{'='*70}")
    typer.echo("Rebalancing Frequency Sensitivity (PCA_SUB)")
    typer.echo(f"{'Freq':>6} {'AR%':>8} {'R/R':>8} {'MDD%':>8} {'BE bp':>8} {'Turnover':>10}")
    typer.echo("-" * 70)
    sub = df[df["strategy"] == "PCA_SUB"]
    for _, row in sub.iterrows():
        typer.echo(
            f"{int(row['param_value']):>6} {row['AR']:>8.2f} "
            f"{row['R/R']:>8.2f} {row['MDD']:>8.2f} {row['breakeven_bps']:>8.1f}"
        )

    # Save
    out_dir = cfg.output.results_dir
    from pathlib import Path
    Path(f"{out_dir}/sensitivity").mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{out_dir}/sensitivity/rebal_freq.csv", index=False)
    typer.echo(f"\nSaved to {out_dir}/sensitivity/")


@app.command()
def run_factor_analysis(
    config: str = typer.Option("configs/paper_reproduction.yaml", "--config", "-c"),
) -> None:
    """Run factor decomposition and alpha attribution analysis."""
    from pathlib import Path

    import numpy as np

    from .backtest import run_backtest
    from .calendar_align import build_aligned_dataset
    from .config import load_config
    from .constants import JP_TICKERS, US_TICKERS
    from .covariance import compute_cfull
    from .data_loader import build_price_panels, load_raw_data
    from .factor_analysis import (
        compute_factor_scores_timeseries,
        compute_propagation_matrix_stats,
        factor_alpha_attribution,
    )
    from .pca_sub import PCASubModel
    from .standardize import rolling_standardize

    cfg = load_config(config)

    # --- Data pipeline (same as backtest) ---
    raw_data = load_raw_data(cfg.data.raw_dir)
    us_close, jp_close, jp_open, volume = build_price_panels(raw_data)
    date_map, us_close_a, jp_close_a, jp_open_a = build_aligned_dataset(
        us_close, jp_close, jp_open
    )
    import pandas as pd

    us_cc_aligned = us_close_a / us_close_a.shift(1) - 1
    us_cc_aligned.index = date_map["us_date"].values
    jp_cc_aligned = jp_close_a / jp_close_a.shift(1) - 1
    jp_cc_aligned.index = date_map["us_date"].values
    jp_oc_aligned = pd.DataFrame(
        jp_close_a.values / jp_open_a.values - 1,
        index=pd.to_datetime(date_map["us_date"].values),
        columns=jp_close.columns,
    )
    all_cc = pd.concat([us_cc_aligned, jp_cc_aligned], axis=1)
    all_cc.index = pd.to_datetime(all_cc.index)
    z_all = rolling_standardize(all_cc, window=cfg.strategy.window_length)

    tickers_all = list(us_cc_aligned.columns) + list(jp_cc_aligned.columns)
    n_us = len(us_cc_aligned.columns)
    n_jp = len(jp_cc_aligned.columns)

    Cfull = compute_cfull(
        all_cc, cfg.cfull.start_date, cfg.cfull.end_date, cfg.strategy.window_length
    )
    model = PCASubModel(
        Cfull=Cfull, tickers=tickers_all, n_us=n_us, n_jp=n_jp,
        lambda_reg=cfg.strategy.lambda_reg, n_components=cfg.strategy.n_components,
    )

    # --- Factor scores ---
    typer.echo("Computing factor scores...")
    factor_scores, factor_signal_dfs, _ = compute_factor_scores_timeseries(
        z_all, model, cfg.strategy.window_length,
        cfg.backtest.start_date, cfg.backtest.end_date, n_us, n_jp,
    )

    # --- Alpha attribution ---
    typer.echo("Computing alpha attribution...")
    attr_df = factor_alpha_attribution(factor_signal_dfs, jp_oc_aligned, cfg.strategy.quantile)

    typer.echo(f"\n{'='*70}")
    typer.echo("FACTOR ALPHA ATTRIBUTION")
    typer.echo(f"{'='*70}")
    typer.echo(f"{'Factor':<20} {'AR%':>8} {'R/R':>8} {'MDD%':>8} {'BE bp':>8}")
    typer.echo("-" * 70)
    for _, row in attr_df.iterrows():
        typer.echo(
            f"{row['factor']:<20} {row['AR']:>8.2f} {row['R/R']:>8.2f} "
            f"{row['MDD']:>8.2f} {row['breakeven_bps']:>8.1f}"
        )
    typer.echo("=" * 70)

    # --- Propagation matrix ---
    typer.echo("\nComputing propagation matrix B_t statistics...")
    B_mean, B_std = compute_propagation_matrix_stats(
        z_all, model, cfg.strategy.window_length,
        cfg.backtest.start_date, cfg.backtest.end_date, n_us, n_jp,
    )

    # Show top propagation paths
    us_cols = list(us_cc_aligned.columns)
    jp_cols = list(jp_cc_aligned.columns)
    stability = np.abs(B_mean) / (B_std + 1e-12)

    typer.echo(f"\nTop 10 stable propagation paths (|B_mean| / B_std):")
    typer.echo(f"{'US -> JP':<30} {'B_mean':>8} {'B_std':>8} {'Stability':>10}")
    typer.echo("-" * 60)
    flat_idx = np.argsort(stability.flatten())[::-1]
    for rank, idx in enumerate(flat_idx[:10]):
        j, i = divmod(idx, len(us_cols))
        typer.echo(
            f"{us_cols[i]:>6} -> {jp_cols[j]:<10}      "
            f"{B_mean[j, i]:>8.4f} {B_std[j, i]:>8.4f} {stability[j, i]:>10.2f}"
        )

    # Save
    out = Path(cfg.output.results_dir) / "factor_analysis"
    out.mkdir(parents=True, exist_ok=True)
    factor_scores.to_csv(out / "factor_scores.csv")
    attr_df.to_csv(out / "alpha_attribution.csv", index=False)
    pd.DataFrame(B_mean, index=jp_cols, columns=us_cols).to_csv(out / "B_mean.csv")
    pd.DataFrame(B_std, index=jp_cols, columns=us_cols).to_csv(out / "B_std.csv")
    pd.DataFrame(stability, index=jp_cols, columns=us_cols).to_csv(out / "B_stability.csv")
    typer.echo(f"\nSaved to {out}/")


@app.command()
def run_conditional_analysis(
    config: str = typer.Option("configs/paper_reproduction.yaml", "--config", "-c"),
) -> None:
    """Run conditional trading frontier analysis."""
    from pathlib import Path

    import pandas as pd

    from .backtest import run_backtest
    from .calendar_align import build_aligned_dataset
    from .conditional_trading import run_conditional_frontier
    from .config import load_config
    from .covariance import compute_cfull
    from .data_loader import build_price_panels, load_raw_data
    from .signal_builder import build_all_signals
    from .standardize import rolling_standardize

    cfg = load_config(config)

    # Quick pipeline to get PCA_SUB signal and JP OC returns
    raw_data = load_raw_data(cfg.data.raw_dir)
    us_close, jp_close, jp_open, volume = build_price_panels(raw_data)
    date_map, us_close_a, jp_close_a, jp_open_a = build_aligned_dataset(
        us_close, jp_close, jp_open
    )
    us_cc_aligned = us_close_a / us_close_a.shift(1) - 1
    us_cc_aligned.index = date_map["us_date"].values
    jp_cc_aligned = jp_close_a / jp_close_a.shift(1) - 1
    jp_cc_aligned.index = date_map["us_date"].values
    jp_oc_aligned = pd.DataFrame(
        jp_close_a.values / jp_open_a.values - 1,
        index=pd.to_datetime(date_map["us_date"].values),
        columns=jp_close.columns,
    )
    all_cc = pd.concat([us_cc_aligned, jp_cc_aligned], axis=1)
    all_cc.index = pd.to_datetime(all_cc.index)

    Cfull = compute_cfull(
        all_cc, cfg.cfull.start_date, cfg.cfull.end_date, cfg.strategy.window_length
    )
    tickers_all = list(us_cc_aligned.columns) + list(jp_cc_aligned.columns)
    signals = build_all_signals(
        us_cc_aligned, jp_cc_aligned, cfg.strategy.window_length,
        cfg.strategy.n_components, cfg.strategy.lambda_reg, Cfull,
        cfg.backtest.start_date, cfg.backtest.end_date, tickers_all,
    )
    pca_sub_signal = signals["PCA_SUB"]

    typer.echo("Running conditional trading frontier...")
    frontier = run_conditional_frontier(
        pca_sub_signal, jp_oc_aligned,
        quantile=cfg.strategy.quantile,
        cost_one_way_bps=5.0,
        cost_short_bps=3.0,
    )

    typer.echo(f"\n{'='*80}")
    typer.echo("CONDITIONAL TRADING FRONTIER (PCA_SUB, cost=5bp+3bp)")
    typer.echo(f"{'='*80}")
    typer.echo(
        f"{'Pctl':>6} {'TradeDays':>10} {'Trade%':>8} "
        f"{'GrossAR%':>10} {'GrossR/R':>10} "
        f"{'NetAR%':>10} {'NetR/R':>10} {'AvgBps':>8}"
    )
    typer.echo("-" * 80)
    for _, row in frontier.iterrows():
        typer.echo(
            f"{row['strength_percentile']:>6.0f} {int(row['trade_days']):>10} "
            f"{row['trade_pct']:>7.1f}% "
            f"{row['gross_AR']:>10.2f} {row['gross_RR']:>10.2f} "
            f"{row['net_AR']:>10.2f} {row['net_RR']:>10.2f} "
            f"{row['avg_trade_day_gross_bps']:>8.1f}"
        )
    typer.echo("=" * 80)

    # Find best net R/R
    if not frontier.empty:
        best = frontier.loc[frontier["net_RR"].idxmax()]
        typer.echo(
            f"\nBest net R/R: {best['net_RR']:.2f} at percentile {best['strength_percentile']:.0f} "
            f"(trade {best['trade_pct']:.1f}% of days, net AR={best['net_AR']:.2f}%)"
        )

    out = Path(cfg.output.results_dir) / "conditional"
    out.mkdir(parents=True, exist_ok=True)
    frontier.to_csv(out / "frontier.csv", index=False)
    typer.echo(f"Saved to {out}/")


@app.command()
def run_blend_analysis(
    config: str = typer.Option("configs/paper_reproduction.yaml", "--config", "-c"),
) -> None:
    """Run partial rebalance (blending) analysis."""
    from pathlib import Path

    import pandas as pd

    from .calendar_align import build_aligned_dataset
    from .config import load_config
    from .cost_model import apply_cost, compute_trading_cost, cost_breakeven_analysis
    from .covariance import compute_cfull
    from .data_loader import build_price_panels, load_raw_data
    from .metrics import compute_all_metrics
    from .portfolio import build_blended_weights, compute_portfolio_returns, compute_turnover
    from .signal_builder import build_all_signals
    from .standardize import rolling_standardize

    cfg = load_config(config)

    raw_data = load_raw_data(cfg.data.raw_dir)
    us_close, jp_close, jp_open, volume = build_price_panels(raw_data)
    date_map, us_close_a, jp_close_a, jp_open_a = build_aligned_dataset(
        us_close, jp_close, jp_open
    )
    us_cc_aligned = us_close_a / us_close_a.shift(1) - 1
    us_cc_aligned.index = date_map["us_date"].values
    jp_cc_aligned = jp_close_a / jp_close_a.shift(1) - 1
    jp_cc_aligned.index = date_map["us_date"].values
    jp_oc_aligned = pd.DataFrame(
        jp_close_a.values / jp_open_a.values - 1,
        index=pd.to_datetime(date_map["us_date"].values),
        columns=jp_close.columns,
    )
    all_cc = pd.concat([us_cc_aligned, jp_cc_aligned], axis=1)
    all_cc.index = pd.to_datetime(all_cc.index)

    Cfull = compute_cfull(
        all_cc, cfg.cfull.start_date, cfg.cfull.end_date, cfg.strategy.window_length
    )
    tickers_all = list(us_cc_aligned.columns) + list(jp_cc_aligned.columns)
    signals = build_all_signals(
        us_cc_aligned, jp_cc_aligned, cfg.strategy.window_length,
        cfg.strategy.n_components, cfg.strategy.lambda_reg, Cfull,
        cfg.backtest.start_date, cfg.backtest.end_date, tickers_all,
    )
    pca_sub_signal = signals["PCA_SUB"]

    alphas = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    records = []

    for alpha in alphas:
        weights = build_blended_weights(pca_sub_signal, cfg.strategy.quantile, alpha)
        port_ret = compute_portfolio_returns(weights, jp_oc_aligned).dropna()
        mask = (port_ret.index >= cfg.backtest.start_date) & (
            port_ret.index <= cfg.backtest.end_date
        )
        port_ret = port_ret[mask]
        if len(port_ret) < 10:
            continue

        turnover = compute_turnover(weights).loc[port_ret.index]
        avg_turnover = turnover.mean()

        metrics_gross = compute_all_metrics(port_ret)
        be = cost_breakeven_analysis(port_ret, weights)

        costs = compute_trading_cost(
            weights.loc[port_ret.index], one_way_bps=5.0, short_extra_bps=3.0
        )
        port_ret_net = apply_cost(port_ret, costs)
        metrics_net = compute_all_metrics(port_ret_net)

        records.append({
            "blend_alpha": alpha,
            "gross_AR": metrics_gross["AR"],
            "gross_RR": metrics_gross["R/R"],
            "gross_MDD": metrics_gross["MDD"],
            "net_AR": metrics_net["AR"],
            "net_RR": metrics_net["R/R"],
            "net_MDD": metrics_net["MDD"],
            "avg_turnover": avg_turnover,
            "breakeven_bps": be["breakeven_one_way_bps"],
        })

    df = pd.DataFrame(records)

    typer.echo(f"\n{'='*80}")
    typer.echo("PARTIAL REBALANCE (BLEND) ANALYSIS — PCA_SUB (cost=5bp+3bp)")
    typer.echo(f"{'='*80}")
    typer.echo(
        f"{'Alpha':>6} {'GrossAR%':>10} {'GrossR/R':>10} "
        f"{'NetAR%':>10} {'NetR/R':>10} {'Turnover':>10} {'BE bp':>8}"
    )
    typer.echo("-" * 80)
    for _, row in df.iterrows():
        typer.echo(
            f"{row['blend_alpha']:>6.1f} {row['gross_AR']:>10.2f} {row['gross_RR']:>10.2f} "
            f"{row['net_AR']:>10.2f} {row['net_RR']:>10.2f} "
            f"{row['avg_turnover']:>10.3f} {row['breakeven_bps']:>8.1f}"
        )
    typer.echo("=" * 80)

    if not df.empty:
        best = df.loc[df["net_RR"].idxmax()]
        typer.echo(
            f"\nBest net R/R: {best['net_RR']:.2f} at alpha={best['blend_alpha']:.1f} "
            f"(net AR={best['net_AR']:.2f}%, turnover={best['avg_turnover']:.3f})"
        )

    out = Path(cfg.output.results_dir) / "blend"
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "blend_analysis.csv", index=False)
    typer.echo(f"Saved to {out}/")


def _save_results(cfg, results) -> None:
    """Save signals, portfolios, metrics, and plots."""
    import json
    from .config import Config
    from .metrics import compute_cumulative_wealth, compute_drawdown_series

    out = Path(cfg.output.results_dir)

    # Save signals
    sig_dir = out / "signals"
    sig_dir.mkdir(parents=True, exist_ok=True)
    for name, res in results.items():
        res.signal.to_csv(sig_dir / f"{name}_signal.csv")

    # Save daily returns
    port_dir = out / "portfolios"
    port_dir.mkdir(parents=True, exist_ok=True)
    for name, res in results.items():
        res.daily_returns.to_csv(port_dir / f"{name}_returns.csv", header=True)

    # Save metrics
    met_dir = out / "metrics"
    met_dir.mkdir(parents=True, exist_ok=True)
    summary = {name: res.metrics for name, res in results.items()}
    with open(met_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Save plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_dir = out / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)

        # Cumulative returns
        fig, ax = plt.subplots(figsize=(12, 6))
        for name, res in results.items():
            wealth = compute_cumulative_wealth(res.daily_returns)
            ax.plot(wealth.index, wealth.values, label=name)
        ax.set_title("Cumulative Returns")
        ax.set_ylabel("Wealth")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / f"cumulative_returns.{cfg.output.plots_format}", dpi=150)
        plt.close(fig)

        # Drawdown
        fig, ax = plt.subplots(figsize=(12, 4))
        for name, res in results.items():
            dd = compute_drawdown_series(res.daily_returns)
            ax.plot(dd.index, dd.values * 100, label=name)
        ax.set_title("Drawdown")
        ax.set_ylabel("Drawdown (%)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plot_dir / f"drawdown.{cfg.output.plots_format}", dpi=150)
        plt.close(fig)

    except Exception as e:
        logging.getLogger(__name__).warning(f"Plot generation failed: {e}")


if __name__ == "__main__":
    app()
