# PCA_SUB: 部分空間正則化付きPCAを用いた日米業種リードラグ投資戦略

論文 "Lead-lag strategies for Japanese and U.S. sectors using subspace regularization PCA" (Nakagawa et al., 2026, SIG-FIN-036) の再現・実務検証プロジェクト。

## TL;DR

**Gross alpha は確実に存在する。問題は執行コストとの戦い。**

- 米国業種ETF → 日本業種の翌日リードラグ効果を統計的に確認（OOS Test R/R=1.6）
- αの95%は**シクリカル/ディフェンシブファクター**から発生
- 毎日取引ではコスト負け → **シグナル強度上位10%の日だけ取引**でNet黒字
- ETFの流動性不足 → **個別株バスケット**（62大型株）で解決
- **最終形: Net AR=3.9%, Net R/R=0.99（OOS Test、3bp片道コスト）**

詳細: [`reports/paper_reproduction/findings.md`](reports/paper_reproduction/findings.md) | 実務PoC: [`docs/practical_poc.md`](docs/practical_poc.md)

## セットアップ

```bash
pip install -e ".[dev]"
python -m src.cli fetch-data --config configs/base.yaml
```

## 主要コマンド

```bash
# 論文再現バックテスト
python -m src.cli run-backtest --config configs/paper_reproduction.yaml

# 感度分析
python -m src.cli run-sensitivity --config configs/paper_reproduction.yaml

# ファクター分解
python -m src.cli run-factor-analysis

# 条件付き取引OOS検証
python -m src.cli run-conditional-oos

# Walk-forward K選択
python -m src.cli run-walk-forward-k

# 個別株バスケット（最終形）
python -m src.cli run-basket-backtest --filter-pct 90 --cost-bps 3

# テスト
pytest tests/ -v
```

## 最終形の戦略仕様

| 項目 | 値 |
|------|------|
| シグナル | PCA_SUB (K=3, λ=0.9, L=60) |
| フィルタ | シグナル強度 上位10%の日のみ |
| 執行 | 個別株バスケット（62大型株） |
| ポジション | 等ウェイト long/short, q=0.3 |
| 取引頻度 | 月2-3回 |
| Breakeven | 8.6bp（コスト前提3bpに対して余裕） |
| **Test Net AR** | **3.91%** |
| **Test Net R/R** | **0.99** |

## 開発ロードマップ

- [x] Phase 1: 論文再現（MOM / PCA_PLAIN / PCA_SUB比較）
- [x] Phase 2: コスト・流動性制約の検証
- [x] Phase 3: パラメータ感度分析（L, λ, K, q, Cfull, リバランス頻度）
- [x] Phase 4: シグナル情報構造分解、条件付き取引、ブレンド
- [x] Phase 5: OOS検証、Walk-forward K、個別株バスケット設計
- [x] Phase 6: 個別株バスケット実バックテスト
- [ ] Phase 7: 実運用パイプライン構築

## プロジェクト構成

```
├── configs/              # YAML実験設定
├── data/raw/             # ETF価格データ
├── data/raw/stocks/      # 個別株価格データ（62銘柄）
├── docs/
│   ├── spec.md               # 実装指示書
│   ├── next_steps.md         # 次フェーズ戦略文書
│   └── practical_poc.md      # 実務PoC仕様書
├── src/                  # コアライブラリ
│   ├── pca_sub.py            # 部分空間正則化PCA（本命）
│   ├── factor_analysis.py    # ファクター分解・α帰属
│   ├── conditional_trading.py # 条件付き取引
│   ├── basket_backtest.py    # 個別株バスケットBT
│   ├── stock_basket.py       # 業種→個別株マッピング
│   ├── walk_forward.py       # Walk-forward検証
│   ├── experiments.py        # 感度分析
│   └── cli.py                # CLI (12コマンド)
├── tests/                # 23テスト
├── results/              # 出力
└── reports/              # 検証レポート
```
