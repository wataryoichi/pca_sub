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
