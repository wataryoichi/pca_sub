"""Report generation for backtest results."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import Config


def generate_report(cfg: Config) -> None:
    """Generate a Markdown report from saved results."""
    results_dir = Path(cfg.output.results_dir)
    reports_dir = Path(cfg.output.reports_dir) / "paper_reproduction"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Load metrics
    metrics_path = results_dir / "metrics" / "summary.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"No metrics found at {metrics_path}. Run backtest first.")

    with open(metrics_path) as f:
        metrics = json.load(f)

    # Build report
    lines: list[str] = []
    lines.append("# Paper Reproduction Report")
    lines.append("")
    lines.append("## Parameters")
    lines.append("")
    lines.append(f"- Window (L): {cfg.strategy.window_length}")
    lines.append(f"- Lambda (λ): {cfg.strategy.lambda_reg}")
    lines.append(f"- Components (K): {cfg.strategy.n_components}")
    lines.append(f"- Quantile (q): {cfg.strategy.quantile}")
    lines.append(f"- Cfull period: {cfg.cfull.start_date} to {cfg.cfull.end_date}")
    lines.append(f"- Backtest period: {cfg.backtest.start_date} to {cfg.backtest.end_date}")
    lines.append("")

    # Summary table
    lines.append("## Strategy Comparison")
    lines.append("")
    lines.append("| Strategy | AR (%) | RISK (%) | R/R | MDD (%) | Hit Ratio |")
    lines.append("|----------|-------:|--------:|----:|--------:|----------:|")
    for name, m in metrics.items():
        lines.append(
            f"| {name} | {m['AR']:.2f} | {m['RISK']:.2f} | "
            f"{m['R/R']:.2f} | {m['MDD']:.2f} | {m['Hit Ratio']*100:.1f}% |"
        )
    lines.append("")

    # Plots reference
    plots_dir = results_dir / "plots"
    if plots_dir.exists():
        lines.append("## Charts")
        lines.append("")
        for img in sorted(plots_dir.glob(f"*.{cfg.output.plots_format}")):
            rel = img.relative_to(reports_dir)
            lines.append(f"![{img.stem}]({rel})")
            lines.append("")

    report_path = reports_dir / "report.md"
    report_path.write_text("\n".join(lines))
