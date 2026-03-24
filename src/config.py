"""YAML configuration loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class StrategyConfig:
    window_length: int = 60
    lambda_reg: float = 0.9
    n_components: int = 3
    quantile: float = 0.3
    prior_dim: int = 3
    rebal_freq: int = 1  # 1 = daily (default), 5 = weekly, etc.


@dataclass
class CfullConfig:
    start_date: str = "2010-01-01"
    end_date: str = "2014-12-31"
    mode: str = "fixed"  # fixed / rolling / expanding


@dataclass
class BacktestConfig:
    start_date: str = "2015-01-01"
    end_date: str = "2025-12-31"


@dataclass
class CostConfig:
    enabled: bool = False
    one_way_bps: float = 10.0
    short_extra_bps: float = 5.0
    illiquid_extra_bps: float = 5.0


@dataclass
class LiquidityConfig:
    enabled: bool = False
    min_avg_turnover_jpy: float = 100_000_000
    lookback_days: int = 20


@dataclass
class DataConfig:
    provider: str = "yfinance"
    start_date: str = "2010-01-01"
    end_date: str = "2025-12-31"
    raw_dir: str = "data/raw"
    interim_dir: str = "data/interim"
    processed_dir: str = "data/processed"


@dataclass
class OutputConfig:
    results_dir: str = "results"
    reports_dir: str = "reports"
    plots_format: str = "png"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    cfull: CfullConfig = field(default_factory=CfullConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    cost: CostConfig = field(default_factory=CostConfig)
    liquidity: LiquidityConfig = field(default_factory=LiquidityConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base."""
    merged = base.copy()
    for k, v in override.items():
        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = _merge_dict(merged[k], v)
        else:
            merged[k] = v
    return merged


def _dict_to_dataclass(section: dict[str, Any], cls: type) -> Any:
    known = {f.name for f in cls.__dataclass_fields__.values()}
    return cls(**{k: v for k, v in section.items() if k in known})


def load_config(path: str | Path) -> Config:
    """Load a YAML config file, resolving inheritance."""
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    # Handle inheritance
    if "inherit" in raw:
        base_path = path.parent / raw.pop("inherit")
        with open(base_path) as f:
            base_raw = yaml.safe_load(f) or {}
        raw = _merge_dict(base_raw, raw)

    return Config(
        data=_dict_to_dataclass(raw.get("data", {}), DataConfig),
        strategy=_dict_to_dataclass(raw.get("strategy", {}), StrategyConfig),
        cfull=_dict_to_dataclass(raw.get("cfull", {}), CfullConfig),
        backtest=_dict_to_dataclass(raw.get("backtest", {}), BacktestConfig),
        cost=_dict_to_dataclass(raw.get("cost", {}), CostConfig),
        liquidity=_dict_to_dataclass(raw.get("liquidity", {}), LiquidityConfig),
        output=_dict_to_dataclass(raw.get("output", {}), OutputConfig),
    )
