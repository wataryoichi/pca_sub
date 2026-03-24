"""Experiment runner for sensitivity analysis.

Runs multiple backtest configurations and collects results for comparison.
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .backtest import BacktestResult, run_backtest
from .config import Config, load_config
from .cost_model import cost_breakeven_analysis

logger = logging.getLogger(__name__)


def run_sensitivity(
    base_cfg: Config,
    param_name: str,
    param_values: list,
    strategies: list[str] | None = None,
) -> pd.DataFrame:
    """Run sensitivity analysis over a single parameter.

    Args:
        base_cfg: Base configuration to vary
        param_name: Dot-separated parameter path (e.g., 'strategy.window_length')
        param_values: List of values to test
        strategies: Strategy names to include (default: all)

    Returns:
        DataFrame with columns: param_value, strategy, AR, RISK, R/R, MDD, ...
    """
    if strategies is None:
        strategies = ["MOM", "PCA_PLAIN", "PCA_SUB"]

    records: list[dict] = []

    for val in param_values:
        logger.info(f"Running {param_name}={val} ...")
        cfg = _set_param(base_cfg, param_name, val)

        try:
            results = run_backtest(cfg)
        except Exception as e:
            logger.error(f"  Failed for {param_name}={val}: {e}")
            continue

        for name in strategies:
            if name not in results:
                continue
            res = results[name]
            m = res.metrics
            be = cost_breakeven_analysis(res.daily_returns, res.weights)
            records.append({
                "param_name": param_name,
                "param_value": val,
                "strategy": name,
                "AR": m["AR"],
                "RISK": m["RISK"],
                "R/R": m["R/R"],
                "MDD": m["MDD"],
                "Hit Ratio": m["Hit Ratio"],
                "N_days": m["N_days"],
                "breakeven_bps": be["breakeven_one_way_bps"],
            })

    return pd.DataFrame(records)


def run_multi_sensitivity(
    base_config_path: str,
    experiments: dict[str, list],
) -> dict[str, pd.DataFrame]:
    """Run multiple sensitivity analyses.

    Args:
        base_config_path: Path to base YAML config
        experiments: Dict mapping param_name -> list of values

    Returns:
        Dict mapping param_name -> results DataFrame
    """
    base_cfg = load_config(base_config_path)
    all_results: dict[str, pd.DataFrame] = {}

    for param_name, values in experiments.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Sensitivity: {param_name}")
        logger.info(f"Values: {values}")
        logger.info(f"{'='*60}")

        df = run_sensitivity(base_cfg, param_name, values)
        all_results[param_name] = df

    return all_results


def _set_param(cfg: Config, param_path: str, value) -> Config:
    """Set a nested parameter on a Config copy.

    param_path examples: 'strategy.window_length', 'strategy.lambda_reg'
    """
    cfg_copy = copy.deepcopy(cfg)
    parts = param_path.split(".")
    obj = cfg_copy
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)
    return cfg_copy


def save_sensitivity_results(
    results: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> None:
    """Save sensitivity results to CSV files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for param_name, df in results.items():
        safe_name = param_name.replace(".", "_")
        df.to_csv(output_dir / f"sensitivity_{safe_name}.csv", index=False)


def generate_sensitivity_plots(
    results: dict[str, pd.DataFrame],
    output_dir: str | Path,
    fmt: str = "png",
) -> None:
    """Generate sensitivity analysis plots."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for param_name, df in results.items():
        if df.empty:
            continue

        safe_name = param_name.replace(".", "_")

        # R/R plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for metric, ax, title in [
            ("R/R", axes[0], "Risk-Return Ratio"),
            ("AR", axes[1], "Annualized Return (%)"),
            ("MDD", axes[2], "Max Drawdown (%)"),
        ]:
            for strategy in df["strategy"].unique():
                sub = df[df["strategy"] == strategy]
                ax.plot(sub["param_value"], sub[metric], "o-", label=strategy)
            ax.set_xlabel(param_name.split(".")[-1])
            ax.set_ylabel(metric)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"Sensitivity: {param_name}", fontsize=14)
        fig.tight_layout()
        fig.savefig(output_dir / f"sensitivity_{safe_name}.{fmt}", dpi=150)
        plt.close(fig)

        # Breakeven plot
        fig, ax = plt.subplots(figsize=(8, 5))
        for strategy in df["strategy"].unique():
            sub = df[df["strategy"] == strategy]
            ax.plot(sub["param_value"], sub["breakeven_bps"], "o-", label=strategy)
        ax.set_xlabel(param_name.split(".")[-1])
        ax.set_ylabel("Breakeven one-way cost (bp)")
        ax.set_title(f"Breakeven Analysis: {param_name}")
        ax.axhline(y=3.0, color="red", linestyle="--", alpha=0.5, label="3bp reference")
        ax.axhline(y=5.0, color="darkred", linestyle="--", alpha=0.5, label="5bp reference")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"breakeven_{safe_name}.{fmt}", dpi=150)
        plt.close(fig)
