"""Walk-forward validation for parameter selection.

Selects K (and optionally other params) on a rolling train window,
then evaluates on a fixed-length test window.
"""

from __future__ import annotations

import copy
import logging

import numpy as np
import pandas as pd

from .backtest import run_backtest
from .config import Config
from .cost_model import cost_breakeven_analysis

logger = logging.getLogger(__name__)


def walk_forward_k(
    base_cfg: Config,
    k_candidates: list[int],
    train_years: int = 3,
    test_years: int = 1,
    start_year: int = 2015,
    end_year: int = 2025,
) -> pd.DataFrame:
    """Walk-forward selection of K.

    For each test window:
    1. Select K with highest R/R on the train window
    2. Evaluate that K on the test window

    Args:
        base_cfg: Base configuration
        k_candidates: List of K values to evaluate
        train_years: Length of training window in years
        test_years: Length of test window in years
        start_year: First year of first test window
        end_year: Last year of last test window

    Returns:
        DataFrame with walk-forward results
    """
    records = []
    year = start_year

    while year + test_years - 1 <= end_year:
        train_start = f"{year - train_years}-01-01"
        train_end = f"{year - 1}-12-31"
        test_start = f"{year}-01-01"
        test_end = f"{year + test_years - 1}-12-31"

        logger.info(f"Walk-forward: train={train_start}~{train_end}, test={test_start}~{test_end}")

        # --- Train: select best K ---
        best_k = k_candidates[0]
        best_rr = -999.0

        for k in k_candidates:
            cfg = copy.deepcopy(base_cfg)
            cfg.strategy.n_components = k
            cfg.backtest.start_date = train_start
            cfg.backtest.end_date = train_end

            try:
                results = run_backtest(cfg)
                if "PCA_SUB" in results:
                    rr = results["PCA_SUB"].metrics["R/R"]
                    if rr > best_rr:
                        best_rr = rr
                        best_k = k
            except Exception as e:
                logger.warning(f"  K={k} failed on train: {e}")

        logger.info(f"  Selected K={best_k} (train R/R={best_rr:.2f})")

        # --- Test: evaluate selected K ---
        cfg = copy.deepcopy(base_cfg)
        cfg.strategy.n_components = best_k
        cfg.backtest.start_date = test_start
        cfg.backtest.end_date = test_end

        try:
            results = run_backtest(cfg)
            if "PCA_SUB" in results:
                res = results["PCA_SUB"]
                m = res.metrics
                be = cost_breakeven_analysis(res.daily_returns, res.weights)
                records.append({
                    "test_start": test_start,
                    "test_end": test_end,
                    "selected_k": best_k,
                    "train_RR": best_rr,
                    "test_AR": m["AR"],
                    "test_RR": m["R/R"],
                    "test_MDD": m["MDD"],
                    "test_BE": be["breakeven_one_way_bps"],
                    "N_days": m["N_days"],
                })
        except Exception as e:
            logger.warning(f"  Test failed: {e}")

        year += test_years

    return pd.DataFrame(records)
