# PCA_SUB: 部分空間正則化付きPCAを用いた日米業種リードラグ投資戦略

論文 "Lead-lag strategies for Japanese and U.S. sectors using subspace regularization PCA" (Nakagawa et al., 2026, SIG-FIN-036) の再現・実務検証プロジェクト。

## 概要

米国の取引時間帯終了後に確定した業種別リターン情報を、翌営業日の日本市場の業種別日中リターン予測に活用するリードラグ戦略。部分空間正則化付きPCAにより、日米結合相関行列の共通ファクター構造を安定的に推定する。

### 戦略の流れ

1. 米国11業種ETF（Select Sector SPDR）の当日Close-to-Closeリターンを観測
2. 部分空間正則化PCAで共通ファクタースコアを抽出
3. 日本17業種ETF（TOPIX-17）の翌営業日Open-to-Closeリターンを予測
4. 予測スコアに基づくロング・ショートポートフォリオを構築

## セットアップ

```bash
pip install -e ".[dev]"
```

## 使い方

### 1. データ取得

```bash
python -m src.cli fetch-data --config configs/base.yaml
```

### 2. バックテスト実行

```bash
python -m src.cli run-backtest --config configs/paper_reproduction.yaml
```

### 3. レポート生成

```bash
python -m src.cli generate-report --config configs/paper_reproduction.yaml
```

## テスト

```bash
pytest tests/ -v
```

## 実装戦略

| 戦略 | 説明 |
|------|------|
| **MOM** | 日本側の単純モメンタム（ベースライン） |
| **PCA_PLAIN** | 正則化なしのPCA（λ=0） |
| **PCA_SUB** | 部分空間正則化付きPCA（λ=0.9）— 提案手法 |

## デフォルトパラメータ（論文再現用）

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| L | 60 | ローリングウィンドウ長 |
| λ | 0.9 | 正則化強度 |
| K | 3 | 主成分数 |
| q | 0.3 | ロング/ショート分位点 |
| Cfull期間 | 2010-2014 | 長期相関行列の推定期間 |

## プロジェクト構成

```
├── configs/          # YAML実験設定
├── data/raw/         # 生データ（CSV）
├── src/              # コアライブラリ
│   ├── config.py         # 設定読み込み
│   ├── constants.py      # 銘柄定義・分類
│   ├── data_loader.py    # データ取得
│   ├── calendar_align.py # 日米営業日整列
│   ├── returns.py        # リターン計算
│   ├── standardize.py    # ローリング標準化
│   ├── prior_factors.py  # 事前部分空間V0
│   ├── covariance.py     # 相関行列構築
│   ├── pca_plain.py      # PCA_PLAIN
│   ├── pca_sub.py        # PCA_SUB（本命）
│   ├── momentum.py       # MOMベースライン
│   ├── portfolio.py      # ポートフォリオ構築
│   ├── backtest.py       # バックテストエンジン
│   ├── metrics.py        # 評価指標
│   └── cli.py            # CLIインターフェース
├── tests/            # テスト
├── results/          # 出力（シグナル、損益、グラフ）
└── reports/          # レポート
```

## 開発ロードマップ

- [x] Phase 1: 論文再現の最小版
- [ ] Phase 2: コスト・流動性制約の検証
- [ ] Phase 3: パラメータ感度分析・改良実験
