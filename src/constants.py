"""Ticker definitions and sector classifications."""

from __future__ import annotations

# --- US Select Sector SPDR ETFs (11 sectors) ---
US_TICKERS: list[str] = [
    "XLB",   # Materials
    "XLC",   # Communication Services
    "XLE",   # Energy
    "XLF",   # Financials
    "XLI",   # Industrials
    "XLK",   # Information Technology
    "XLP",   # Consumer Staples
    "XLRE",  # Real Estate
    "XLU",   # Utilities
    "XLV",   # Health Care
    "XLY",   # Consumer Discretionary
]

# --- JP TOPIX-17 ETFs (17 sectors) ---
JP_TICKERS: list[str] = [
    "1617.T",  # 食品
    "1618.T",  # エネルギー資源
    "1619.T",  # 建設・資材
    "1620.T",  # 素材・化学
    "1621.T",  # 医薬品
    "1622.T",  # 自動車・輸送機
    "1623.T",  # 鉄鋼・非鉄
    "1624.T",  # 機械
    "1625.T",  # 電機・精密
    "1626.T",  # 情報通信・サービスその他
    "1627.T",  # 電力・ガス
    "1628.T",  # 運輸・物流
    "1629.T",  # 商社・卸売
    "1630.T",  # 小売
    "1631.T",  # 銀行
    "1632.T",  # 金融（除く銀行）
    "1633.T",  # 不動産
]

ALL_TICKERS: list[str] = US_TICKERS + JP_TICKERS

N_US: int = len(US_TICKERS)   # 11
N_JP: int = len(JP_TICKERS)   # 17
N_TOTAL: int = N_US + N_JP    # 28

# --- Cyclical / Defensive classification (paper Table in Section 4.1) ---
# +1 = cyclical, -1 = defensive, 0 = neutral
US_CYCLICAL_DEFENSIVE: dict[str, int] = {
    "XLB":  +1,   # Materials — cyclical
    "XLC":   0,   # Communication Services — neutral
    "XLE":  +1,   # Energy — cyclical
    "XLF":  +1,   # Financials — cyclical
    "XLI":   0,   # Industrials — neutral
    "XLK":  -1,   # Information Technology — defensive
    "XLP":  -1,   # Consumer Staples — defensive
    "XLRE": +1,   # Real Estate — cyclical
    "XLU":  -1,   # Utilities — defensive
    "XLV":  -1,   # Health Care — defensive
    "XLY":   0,   # Consumer Discretionary — neutral
}

JP_CYCLICAL_DEFENSIVE: dict[str, int] = {
    "1617.T": -1,  # 食品 — defensive
    "1618.T": +1,  # エネルギー資源 — cyclical
    "1619.T":  0,  # 建設・資材 — neutral
    "1620.T":  0,  # 素材・化学 — neutral
    "1621.T": -1,  # 医薬品 — defensive
    "1622.T":  0,  # 自動車・輸送機 — neutral
    "1623.T":  0,  # 鉄鋼・非鉄 — neutral
    "1624.T":  0,  # 機械 — neutral
    "1625.T": +1,  # 電機・精密 — cyclical
    "1626.T":  0,  # 情報通信・サービスその他 — neutral
    "1627.T": -1,  # 電力・ガス — defensive
    "1628.T":  0,  # 運輸・物流 — neutral
    "1629.T": +1,  # 商社・卸売 — cyclical
    "1630.T": -1,  # 小売 — defensive
    "1631.T": +1,  # 銀行 — cyclical
    "1632.T":  0,  # 金融（除く銀行）— neutral
    "1633.T":  0,  # 不動産 — neutral
}

CYCLICAL_DEFENSIVE: dict[str, int] = {**US_CYCLICAL_DEFENSIVE, **JP_CYCLICAL_DEFENSIVE}

# --- JP ticker name mapping ---
JP_TICKER_NAMES: dict[str, str] = {
    "1617.T": "食品",
    "1618.T": "エネルギー資源",
    "1619.T": "建設・資材",
    "1620.T": "素材・化学",
    "1621.T": "医薬品",
    "1622.T": "自動車・輸送機",
    "1623.T": "鉄鋼・非鉄",
    "1624.T": "機械",
    "1625.T": "電機・精密",
    "1626.T": "情報通信・サービスその他",
    "1627.T": "電力・ガス",
    "1628.T": "運輸・物流",
    "1629.T": "商社・卸売",
    "1630.T": "小売",
    "1631.T": "銀行",
    "1632.T": "金融（除く銀行）",
    "1633.T": "不動産",
}
