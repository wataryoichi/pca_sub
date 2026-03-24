"""Performance metrics for strategy evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def annualized_return(returns: pd.Series, periods_per_year: float = 252) -> float:
    """Annualized return (AR) in percent.

    AR = (mean daily return) * periods_per_year * 100
    """
    return returns.mean() * periods_per_year * 100


def annualized_volatility(returns: pd.Series, periods_per_year: float = 252) -> float:
    """Annualized volatility (RISK) in percent.

    RISK = (std of daily returns) * sqrt(periods_per_year) * 100
    """
    return returns.std(ddof=1) * np.sqrt(periods_per_year) * 100


def risk_return_ratio(returns: pd.Series, periods_per_year: float = 252) -> float:
    """Risk-return ratio (R/R = AR / RISK)."""
    ar = annualized_return(returns, periods_per_year)
    risk = annualized_volatility(returns, periods_per_year)
    if risk < 1e-12:
        return 0.0
    return ar / risk


def maximum_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown (MDD) in percent.

    MDD = max_t (1 - W_t / max_{s<=t} W_s) * 100
    where W_t = cumulative wealth
    """
    wealth = (1 + returns).cumprod()
    running_max = wealth.cummax()
    drawdown = (wealth / running_max) - 1
    return abs(drawdown.min()) * 100


def hit_ratio(returns: pd.Series) -> float:
    """Fraction of days with positive returns."""
    valid = returns.dropna()
    if len(valid) == 0:
        return 0.0
    return (valid > 0).mean()


def compute_all_metrics(
    returns: pd.Series,
    periods_per_year: float = 252,
) -> dict:
    """Compute all standard metrics."""
    return {
        "AR": annualized_return(returns, periods_per_year),
        "RISK": annualized_volatility(returns, periods_per_year),
        "R/R": risk_return_ratio(returns, periods_per_year),
        "MDD": maximum_drawdown(returns),
        "Hit Ratio": hit_ratio(returns),
        "N_days": len(returns),
    }


def compute_cumulative_wealth(returns: pd.Series) -> pd.Series:
    """Compute cumulative wealth series (starting at 1)."""
    return (1 + returns).cumprod()


def compute_drawdown_series(returns: pd.Series) -> pd.Series:
    """Compute drawdown time series."""
    wealth = compute_cumulative_wealth(returns)
    running_max = wealth.cummax()
    return wealth / running_max - 1


def compute_rolling_sharpe(
    returns: pd.Series,
    window: int = 60,
    periods_per_year: float = 252,
) -> pd.Series:
    """Compute rolling Sharpe ratio (annualized)."""
    roll_mean = returns.rolling(window=window).mean()
    roll_std = returns.rolling(window=window).std(ddof=1)
    return (roll_mean / roll_std.clip(lower=1e-12)) * np.sqrt(periods_per_year)
