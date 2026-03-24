"""Data fetching with provider abstraction."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from .constants import ALL_TICKERS, JP_TICKERS, US_TICKERS

logger = logging.getLogger(__name__)


class DataProvider(ABC):
    """Abstract data provider interface."""

    @abstractmethod
    def fetch(
        self,
        tickers: list[str],
        start: str,
        end: str,
    ) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV data for given tickers.

        Returns dict mapping ticker -> DataFrame with columns:
        [Open, High, Low, Close, Volume] indexed by date.
        """
        ...


class YFinanceProvider(DataProvider):
    """Fetch data via yfinance."""

    def fetch(
        self,
        tickers: list[str],
        start: str,
        end: str,
    ) -> dict[str, pd.DataFrame]:
        import yfinance as yf

        result: dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            logger.info(f"Fetching {ticker} ...")
            try:
                df = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                )
                if df.empty:
                    logger.warning(f"No data for {ticker}")
                    continue
                # yfinance may return MultiIndex columns for single ticker
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.index = pd.to_datetime(df.index).tz_localize(None)
                df.index.name = "Date"
                result[ticker] = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            except Exception as e:
                logger.error(f"Failed to fetch {ticker}: {e}")
        return result


class CsvProvider(DataProvider):
    """Load data from CSV files in a directory.

    Expects files named like XLB.csv, 1617.T.csv with columns:
    Date, Open, High, Low, Close, Volume
    """

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)

    def fetch(
        self,
        tickers: list[str],
        start: str,
        end: str,
    ) -> dict[str, pd.DataFrame]:
        result: dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            path = self.data_dir / f"{ticker}.csv"
            if not path.exists():
                logger.warning(f"CSV not found: {path}")
                continue
            df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
            df = df.loc[start:end]
            result[ticker] = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        return result


def get_provider(name: str, **kwargs) -> DataProvider:
    """Factory for data providers."""
    providers = {
        "yfinance": YFinanceProvider,
        "csv": CsvProvider,
    }
    cls = providers.get(name)
    if cls is None:
        raise ValueError(f"Unknown provider: {name}. Available: {list(providers)}")
    return cls(**kwargs)


def fetch_and_save(
    provider: DataProvider,
    tickers: list[str],
    start: str,
    end: str,
    output_dir: str | Path,
) -> dict[str, pd.DataFrame]:
    """Fetch data and save individual CSVs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = provider.fetch(tickers, start, end)
    for ticker, df in data.items():
        path = output_dir / f"{ticker}.csv"
        df.to_csv(path)
        logger.info(f"Saved {ticker} -> {path} ({len(df)} rows)")
    return data


def load_raw_data(raw_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load all CSVs from raw directory."""
    raw_dir = Path(raw_dir)
    result: dict[str, pd.DataFrame] = {}
    for ticker in ALL_TICKERS:
        path = raw_dir / f"{ticker}.csv"
        if path.exists():
            df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
            result[ticker] = df
    return result


def build_price_panels(
    raw_data: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build aligned price panels from raw data.

    Returns:
        us_close: US close prices (date x US tickers)
        jp_close: JP close prices (date x JP tickers)
        jp_open:  JP open prices (date x JP tickers)
        volume:   All volumes (date x all tickers)
    """
    us_close = pd.DataFrame(
        {t: raw_data[t]["Close"] for t in US_TICKERS if t in raw_data}
    )
    jp_close = pd.DataFrame(
        {t: raw_data[t]["Close"] for t in JP_TICKERS if t in raw_data}
    )
    jp_open = pd.DataFrame(
        {t: raw_data[t]["Open"] for t in JP_TICKERS if t in raw_data}
    )
    volume = pd.DataFrame(
        {t: raw_data[t]["Volume"] for t in ALL_TICKERS if t in raw_data}
    )
    return us_close, jp_close, jp_open, volume
