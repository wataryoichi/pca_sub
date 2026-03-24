"""Individual stock basket mapping for TOPIX-17 sectors.

Maps each TOPIX-17 ETF to representative large-cap individual stocks
for improved liquidity execution.
"""

from __future__ import annotations

# TOPIX-17 sector -> representative large-cap stocks
# Selected by market cap and liquidity (top 3-5 per sector)
SECTOR_STOCK_MAPPING: dict[str, dict] = {
    "1617.T": {  # 食品
        "name": "食品",
        "stocks": [
            {"ticker": "2914.T", "name": "日本たばこ産業 (JT)"},
            {"ticker": "2802.T", "name": "味の素"},
            {"ticker": "2503.T", "name": "キリンHD"},
            {"ticker": "2502.T", "name": "アサヒグループHD"},
        ],
    },
    "1618.T": {  # エネルギー資源
        "name": "エネルギー資源",
        "stocks": [
            {"ticker": "5020.T", "name": "ENEOS HD"},
            {"ticker": "1605.T", "name": "INPEX"},
            {"ticker": "5021.T", "name": "コスモエネルギーHD"},
        ],
    },
    "1619.T": {  # 建設・資材
        "name": "建設・資材",
        "stocks": [
            {"ticker": "1925.T", "name": "大和ハウス工業"},
            {"ticker": "5332.T", "name": "TOTO"},
            {"ticker": "1928.T", "name": "積水ハウス"},
            {"ticker": "1802.T", "name": "大林組"},
        ],
    },
    "1620.T": {  # 素材・化学
        "name": "素材・化学",
        "stocks": [
            {"ticker": "4063.T", "name": "信越化学工業"},
            {"ticker": "4188.T", "name": "三菱ケミカルグループ"},
            {"ticker": "4005.T", "name": "住友化学"},
            {"ticker": "4452.T", "name": "花王"},
        ],
    },
    "1621.T": {  # 医薬品
        "name": "医薬品",
        "stocks": [
            {"ticker": "4502.T", "name": "武田薬品工業"},
            {"ticker": "4568.T", "name": "第一三共"},
            {"ticker": "4519.T", "name": "中外製薬"},
            {"ticker": "4503.T", "name": "アステラス製薬"},
        ],
    },
    "1622.T": {  # 自動車・輸送機
        "name": "自動車・輸送機",
        "stocks": [
            {"ticker": "7203.T", "name": "トヨタ自動車"},
            {"ticker": "7267.T", "name": "本田技研工業"},
            {"ticker": "7269.T", "name": "スズキ"},
            {"ticker": "7201.T", "name": "日産自動車"},
        ],
    },
    "1623.T": {  # 鉄鋼・非鉄
        "name": "鉄鋼・非鉄",
        "stocks": [
            {"ticker": "5401.T", "name": "日本製鉄"},
            {"ticker": "5411.T", "name": "JFEホールディングス"},
            {"ticker": "5713.T", "name": "住友金属鉱山"},
            {"ticker": "5802.T", "name": "住友電気工業"},
        ],
    },
    "1624.T": {  # 機械
        "name": "機械",
        "stocks": [
            {"ticker": "6301.T", "name": "小松製作所"},
            {"ticker": "6326.T", "name": "クボタ"},
            {"ticker": "6367.T", "name": "ダイキン工業"},
            {"ticker": "7011.T", "name": "三菱重工業"},
        ],
    },
    "1625.T": {  # 電機・精密
        "name": "電機・精密",
        "stocks": [
            {"ticker": "6758.T", "name": "ソニーグループ"},
            {"ticker": "6861.T", "name": "キーエンス"},
            {"ticker": "6501.T", "name": "日立製作所"},
            {"ticker": "6902.T", "name": "デンソー"},
            {"ticker": "8035.T", "name": "東京エレクトロン"},
        ],
    },
    "1626.T": {  # 情報通信・サービスその他
        "name": "情報通信・サービスその他",
        "stocks": [
            {"ticker": "9432.T", "name": "日本電信電話 (NTT)"},
            {"ticker": "9433.T", "name": "KDDI"},
            {"ticker": "9434.T", "name": "ソフトバンク"},
            {"ticker": "4755.T", "name": "楽天グループ"},
        ],
    },
    "1627.T": {  # 電力・ガス
        "name": "電力・ガス",
        "stocks": [
            {"ticker": "9501.T", "name": "東京電力HD"},
            {"ticker": "9503.T", "name": "関西電力"},
            {"ticker": "9531.T", "name": "東京ガス"},
        ],
    },
    "1628.T": {  # 運輸・物流
        "name": "運輸・物流",
        "stocks": [
            {"ticker": "9020.T", "name": "東日本旅客鉄道 (JR東日本)"},
            {"ticker": "9022.T", "name": "東海旅客鉄道 (JR東海)"},
            {"ticker": "9021.T", "name": "西日本旅客鉄道 (JR西日本)"},
            {"ticker": "9064.T", "name": "ヤマトHD"},
        ],
    },
    "1629.T": {  # 商社・卸売
        "name": "商社・卸売",
        "stocks": [
            {"ticker": "8058.T", "name": "三菱商事"},
            {"ticker": "8001.T", "name": "伊藤忠商事"},
            {"ticker": "8031.T", "name": "三井物産"},
            {"ticker": "8002.T", "name": "丸紅"},
        ],
    },
    "1630.T": {  # 小売
        "name": "小売",
        "stocks": [
            {"ticker": "9983.T", "name": "ファーストリテイリング"},
            {"ticker": "3382.T", "name": "セブン&アイ・HD"},
            {"ticker": "8267.T", "name": "イオン"},
        ],
    },
    "1631.T": {  # 銀行
        "name": "銀行",
        "stocks": [
            {"ticker": "8306.T", "name": "三菱UFJフィナンシャル・グループ"},
            {"ticker": "8316.T", "name": "三井住友フィナンシャルグループ"},
            {"ticker": "8411.T", "name": "みずほフィナンシャルグループ"},
        ],
    },
    "1632.T": {  # 金融（除く銀行）
        "name": "金融（除く銀行）",
        "stocks": [
            {"ticker": "8766.T", "name": "東京海上HD"},
            {"ticker": "8750.T", "name": "第一生命HD"},
            {"ticker": "8591.T", "name": "オリックス"},
            {"ticker": "8601.T", "name": "大和証券グループ"},
        ],
    },
    "1633.T": {  # 不動産
        "name": "不動産",
        "stocks": [
            {"ticker": "8801.T", "name": "三井不動産"},
            {"ticker": "8802.T", "name": "三菱地所"},
            {"ticker": "8830.T", "name": "住友不動産"},
        ],
    },
}


def get_all_stock_tickers() -> list[str]:
    """Get flat list of all representative stock tickers."""
    tickers = []
    for sector in SECTOR_STOCK_MAPPING.values():
        for stock in sector["stocks"]:
            tickers.append(stock["ticker"])
    return tickers


def get_sector_for_stock(ticker: str) -> str | None:
    """Get the TOPIX-17 ETF ticker for a given stock ticker."""
    for etf_ticker, sector in SECTOR_STOCK_MAPPING.items():
        for stock in sector["stocks"]:
            if stock["ticker"] == ticker:
                return etf_ticker
    return None


def build_stock_weight_mapping(
    etf_weights: dict[str, float],
) -> dict[str, float]:
    """Map ETF-level weights to equal-weighted individual stock weights.

    Each ETF's weight is split equally among its representative stocks.

    Args:
        etf_weights: Dict mapping ETF ticker -> weight

    Returns:
        Dict mapping stock ticker -> weight
    """
    stock_weights: dict[str, float] = {}
    for etf_ticker, w in etf_weights.items():
        if etf_ticker not in SECTOR_STOCK_MAPPING:
            continue
        stocks = SECTOR_STOCK_MAPPING[etf_ticker]["stocks"]
        n = len(stocks)
        for stock in stocks:
            stock_weights[stock["ticker"]] = w / n
    return stock_weights


def print_liquidity_comparison() -> str:
    """Generate a liquidity comparison summary between ETFs and representative stocks."""
    lines = []
    lines.append("| セクター | ETF | 代表銘柄 | 備考 |")
    lines.append("|---------|-----|---------|------|")
    for etf_ticker, sector in SECTOR_STOCK_MAPPING.items():
        stock_names = ", ".join(s["name"] for s in sector["stocks"][:3])
        lines.append(
            f"| {sector['name']} | {etf_ticker} | {stock_names} | "
            f"{len(sector['stocks'])}銘柄 |"
        )
    return "\n".join(lines)
