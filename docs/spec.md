# PCA_SUB - Claude Code向け実装指示書
## 論文「米国業種ETF → 日本業種ETFの日中リターン予測（PCA_SUB）」再現・検証プロジェクト

## 目的

以下の論文の内容を、まずは**再現性重視**で実装し、その後**実務で使えるかどうか**を検証できる状態にしてください。

目標は次の3段階です。

1. **論文の再現**
   - 米国11業種ETFの前日リターンから、日本17業種ETFの翌営業日 open-to-close リターンを予測する
   - MOM / PCA_PLAIN / PCA_SUB を再現する

2. **実務検証**
   - 日本側ETFの流動性制約
   - 売買コスト
   - スプレッド
   - 寄り付き・引け執行の不利
   を考慮した検証を行う

3. **改良実験**
   - L, λ, K, Cfull構成期間などの感度分析
   - rolling / expanding Cfull
   - 銘柄フィルタ
   - シグナル→ポートフォリオ変換方法の改善

---

## 実装方針

### 最重要方針
- まずは**論文どおりの最小再現**を優先する
- 改良はそのあと
- いきなり最適化しない
- コードは**再現性・比較可能性・差分検証しやすさ**を重視する

### 禁止事項
- 最初からパラメータ最適化をしない
- 論文再現前に独自改造を大量に入れない
- 未来情報を混ぜない
- 見た目だけのバックテスト結果を出して終わらない

---

## 技術要件

### 推奨スタック
- Python
- pandas
- numpy
- scipy
- scikit-learn（必要なら）
- matplotlib
- pyyaml
- typer or click（CLI用）
- pytest

### 開発原則
- 型ヒントを付ける
- 関数分割をしっかり行う
- ロジックと設定を分離する
- 乱数を使う場合は seed 固定
- 実験条件は YAML で管理
- 出力結果は CSV / JSON / Markdown で保存
- グラフは PNG 保存

---

## ディレクトリ構成案

```text
pca-sub-project/
├─ README.md
├─ CLAUDE.md
├─ pyproject.toml
├─ requirements.txt
├─ configs/
│  ├─ base.yaml
│  ├─ paper_reproduction.yaml
│  ├─ sensitivity_l.yaml
│  ├─ sensitivity_lambda.yaml
│  ├─ sensitivity_k.yaml
│  ├─ cfull_rolling.yaml
│  └─ liquidity_filter.yaml
├─ data/
│  ├─ raw/
│  ├─ interim/
│  └─ processed/
├─ notebooks/
├─ reports/
│  ├─ paper_reproduction/
│  ├─ cost_analysis/
│  └─ sensitivity/
├─ results/
│  ├─ signals/
│  ├─ portfolios/
│  ├─ metrics/
│  └─ plots/
├─ src/
│  ├─ __init__.py
│  ├─ config.py
│  ├─ constants.py
│  ├─ data_loader.py
│  ├─ calendar_align.py
│  ├─ returns.py
│  ├─ standardize.py
│  ├─ prior_factors.py
│  ├─ covariance.py
│  ├─ pca_plain.py
│  ├─ pca_sub.py
│  ├─ momentum.py
│  ├─ signal_builder.py
│  ├─ portfolio.py
│  ├─ backtest.py
│  ├─ metrics.py
│  ├─ cost_model.py
│  ├─ liquidity.py
│  ├─ experiments.py
│  └─ cli.py
├─ tests/
│  ├─ test_returns.py
│  ├─ test_standardize.py
│  ├─ test_prior_factors.py
│  ├─ test_covariance.py
│  ├─ test_pca_plain.py
│  ├─ test_pca_sub.py
│  ├─ test_portfolio.py
│  └─ test_backtest.py
└─ scripts/
   ├─ run_paper_reproduction.sh
   ├─ run_sensitivity.sh
   └─ generate_report.sh
```

---

## 対象銘柄

### 米国側（Select Sector SPDR ETF）
- XLC
- XLY
- XLP
- XLE
- XLF
- XLV
- XLI
- XLB
- XLRE
- XLK
- XLU

### 日本側（TOPIX-17 ETF）
- 1617
- 1618
- 1619
- 1620
- 1621
- 1622
- 1623
- 1624
- 1625
- 1626
- 1627
- 1628
- 1629
- 1630
- 1631
- 1632
- 1633

### 備考
- まずは価格データが揃うことを優先
- 後で流動性フィルタを追加する
- 日本側は流動性が薄い銘柄がある前提で設計する

---

## データ要件

### 必須データ
各銘柄について、少なくとも以下を取得できるようにする

- date
- open
- high
- low
- close
- adjusted close（あれば）
- volume

### 推奨
- split / dividend を反映した価格
- 売買代金推定用に close × volume を計算できるようにする
- 日本と米国でタイムゾーンを明確に分ける

### データソース
最初は簡便な取得方法でよいが、将来差し替えやすくすること

例:
- yfinance
- stooq
- JPX系データ
- 証券会社データ
- CSV 手動配置

### 注意
- データ取得処理は provider abstraction を作る
- vendor 依存を最小化する

---

## 実装する戦略

## 1. 日次リターン定義

### 米国側説明変数
米国側は当日 close-to-close リターン

$$
r^{cc}_{i,t} = \frac{P^{close}_{i,t}}{P^{close}_{i,t-1}} - 1
$$

### 日本側予測対象
日本側は翌営業日 open-to-close リターン

$$
r^{oc}_{j,t+1} = \frac{P^{close}_{j,t+1}}{P^{open}_{j,t+1}} - 1
$$

### 重要
- 米国の t 日引け情報から
- 日本の t+1 日の日中リターンを予測する
- カレンダー整列が重要
- 金曜米国 → 月曜日本 の対応を必ず正しく処理する

---

## 2. ローリング標準化

推定窓長 L を使い、直近 L 営業日で標準化する

$$
\mu_{i,t} = \frac{1}{L}\sum_{\tau \in W_t} r^{cc}_{i,\tau}
$$

$$
\sigma_{i,t} = \sqrt{\frac{1}{L}\sum_{\tau \in W_t}(r^{cc}_{i,\tau} - \mu_{i,t})^2}
$$

$$
z_{i,\tau} = \frac{r^{cc}_{i,\tau} - \mu_{i,t}}{\sigma_{i,t}}
$$

ここで

$$
W_t = \{t-L, \dots, t-1\}
$$

### デフォルト
- L = 60

### 実装注意
- ゼロ分散回避
- 欠損処理
- 過去データのみ使用

---

## 3. 相関行列の構築

日米全業種を縦に結合した標準化リターンから相関行列を作る

$$
C_t \in \mathbb{R}^{N \times N}
$$

ここで
- N = 米国11 + 日本17 = 28

### 実装注意
- 行列次元が期待通りか検証
- 対称性を確認
- 対角が1に近いか確認
- 数値安定性を確保

---

## 4. ベースライン実装

### MOM
シンプルなベースラインも実装する
論文どおりのルールに沿ってシグナルを作る

### PCA_PLAIN
通常のPCAを相関行列 C_t に対して実行し、上位 K 成分を利用する

---

## 5. PCA_SUB 実装

これが本命

### 5.1 事前部分空間 V0 を作る

#### v1: グローバル因子
全業種に同じ重み

$$
v_1 \propto \mathbf{1}
$$

#### v2: 国スプレッド因子
米国を正、日本を負

粗いベクトル a を作る

$$
a =
\begin{bmatrix}
1_{N_U}\\
-1_{N_J}
\end{bmatrix}
$$

直交化して正規化

$$
v_2 = \frac{a - (v_1^\top a)v_1}{\|a - (v_1^\top a)v_1\|}
$$

#### v3: シクリカル / ディフェンシブ因子
業種ごとに cyclical = +1, defensive = -1 の符号ベクトル b を作る
その後 v1, v2 に直交化

$$
\tilde b = b - (v_1^\top b)v_1 - (v_2^\top b)v_2
$$

$$
v_3 = \frac{\tilde b}{\|\tilde b\|}
$$

#### 実装要件
- v1, v2, v3 が互いに直交していることを確認
- 正規化されていることを確認
- cyclical / defensive の分類表を constants.py で管理

---

## 5.2 Cfull の構築

長期相関行列 Cfull を作る

### 再現用デフォルト
- 2010-01-01 〜 2014-12-31

### 重要
- なぜこの期間か論文で明示されていないので、コード上は容易に差し替え可能にする
- 固定期間 / rolling / expanding を切り替えられる設計にする

---

## 5.3 C0 の構築

$$
V_0 = [v_1, v_2, v_3]
$$

$$
D_0 = \mathrm{diag}(V_0^\top C_{\text{full}} V_0)
$$

$$
C_0^{raw} = V_0 D_0 V_0^\top
$$

$$
\Delta = \mathrm{diag}(C_0^{raw})
$$

$$
C_0 = \Delta^{-1/2} C_0^{raw} \Delta^{-1/2}
$$

### 実装注意
- C0 が対称か
- 対角が1になっているか
- 半正定値近傍か
- 数値誤差で固有値が微小マイナスになる場合の処理方針を用意

---

## 5.4 正則化相関行列

$$
C_t^{reg} = (1-\lambda)C_t + \lambda C_0
$$

### 再現用デフォルト
- λ = 0.9

---

## 5.5 固有分解

$$
C_t^{reg} = V_t \Lambda_t V_t^\top
$$

上位 K 本を使う

### 再現用デフォルト
- K = 3

### 実装注意
- 固有値降順
- 実数成分化
- 符号反転の安定性に配慮
- 毎回の分解結果を追跡可能にする

---

## 5.6 日米ブロック分解

$$
V_t^{(K)} =
\begin{bmatrix}
V_{U,t}^{(K)} \\
V_{J,t}^{(K)}
\end{bmatrix}
$$

---

## 5.7 予測シグナル生成

米国当日の標準化ベクトルを使う

$$
f_t = (V_{U,t}^{(K)})^\top z_{U,t}
$$

$$
\hat z_{J,t+1} = V_{J,t}^{(K)} f_t
$$

すなわち

$$
\hat z_{J,t+1} = V_{J,t}^{(K)}(V_{U,t}^{(K)})^\top z_{U,t}
$$

### 出力
- 各営業日
- 各日本ETF
- 予測スコア
を保存

---

## 6. 売買ルール

論文再現では以下で実装

1. 予測スコアを日本17業種で順位付け
2. 上位30%をロング
3. 下位30%をショート
4. 等ウェイト
5. 日本寄りで建てる
6. 日本引けで閉じる

### デフォルト
- q = 0.3

### 実装上の注意
- 17本の30%は端数が出るため、件数決定ルールを明示
- 同順位がある場合の処理を固定
- ネットエクスポージャーが0になるようにする

---

## 7. 評価指標

最低限、以下を実装

- AR: annualized return
- RISK: annualized volatility
- R/R
- MDD: maximum drawdown
- hit ratio
- turnover
- average holding period
- long leg return
- short leg return
- sub-period metrics

### MDD
累積資産曲線から計算すること

$$
\text{Drawdown}_t = 1 - \frac{V_t}{\max_{s \le t} V_s}
$$

$$
\text{MDD} = \max_t \text{Drawdown}_t
$$

---

## 8. レポート出力

各実験ごとに以下を自動生成

### 必須表
- パラメータ一覧
- リターン指標一覧
- ベースライン比較表
- サブ期間別比較表

### 必須グラフ
- 累積リターン曲線
- ドローダウン曲線
- rolling Sharpe 相当
- turnover 時系列
- 流動性不足銘柄比率
- 各ETFの平均予測スコア分布

### 形式
- Markdown レポート
- CSV
- PNG

---

## 実務検証で追加実装すべきもの

## 1. コストモデル

以下を設定可能にする

- 売買手数料
- 片道スプレッドコスト
- 寄り・引け追加スリッページ
- 空売りコスト
- borrow fee proxy

### 実装方針
$$
r^{net}_t = r^{gross}_t - c^{trade}_t - c^{short}_t
$$

### コストはまず簡易モデルでよい
例:
- 片道 10bp
- ショート追加 5bp
- 薄いETFにはさらに追加コスト

---

## 2. 流動性フィルタ

日本ETFについて、以下で除外可能にする

- 直近平均売買代金が閾値未満
- 直近平均出来高が閾値未満
- 平均スプレッドが閾値超過
- 寄り付き板不足 proxy

### 出力
- どの日にどの銘柄を除外したか
- 除外後に候補数が足りるか

---

## 3. 執行の現実化

論文では open-to-close 前提だが、実務では以下も試せるようにする

- open で建てて close で閉じる
- open+5min で建てて close auction 前に閉じる
- VWAP 近似
- 引け板寄せ近傍で一部未約定リスクを反映

---

## 感度分析でやるべき実験

## 1. L 感度
候補:
- 20
- 40
- 60
- 120

## 2. λ 感度
候補:
- 0
- 0.3
- 0.6
- 0.9
- 0.95

## 3. K 感度
候補:
- 1
- 2
- 3
- 4
- 5

## 4. Cfull 構成
候補:
- 2010-2014 固定
- 直近5年 rolling
- expanding
- crisis 除外

## 5. ポートフォリオ構築
候補:
- 等ウェイト
- スコア比例
- ボラ調整
- q = 10%, 20%, 30%

---

## 検証設計

### 絶対ルール
パラメータ探索と最終評価を同じ期間でやらない

### 推奨
- train
- validation
- test
を分ける

例:
- train: 2010-2018
- validation: 2019-2022
- test: 2023-2025

さらに可能なら walk-forward も実装する

---

## 最低限の CLI

以下のCLIを作る

```bash
python -m src.cli fetch-data --config configs/base.yaml
python -m src.cli build-dataset --config configs/base.yaml
python -m src.cli run-backtest --config configs/paper_reproduction.yaml
python -m src.cli run-experiment --config configs/sensitivity_l.yaml
python -m src.cli generate-report --config configs/paper_reproduction.yaml
```

### 望ましい追加コマンド
```bash
python -m src.cli validate-data
python -m src.cli inspect-signal
python -m src.cli compare-models
python -m src.cli run-cost-analysis
python -m src.cli run-liquidity-filter
```

---

## テスト要件

最低限以下をテストする

### 数学・実装整合
- 標準化結果が期待通り
- 相関行列が対称
- V0 が直交
- C0 の対角が1
- λ=0 なら PCA_SUB が PCA_PLAIN に近い挙動になるか確認
- K=全次元 のときの挙動確認
- ブロック分割次元が正しい

### バックテスト整合
- シグナル順位ロジック
- long / short 本数
- 等ウェイトの合計
- 日次収益計算
- 累積資産計算
- MDD 計算

### データ整合
- 米国営業日と日本営業日の対応
- 金曜→月曜の対応
- 欠損時のスキップ / 補完ルール

---

## 実装順序

## Phase 1: 再現最小版
- データ取得
- カレンダー整列
- リターン計算
- 標準化
- MOM
- PCA_PLAIN
- PCA_SUB
- バックテスト
- 指標算出
- 最低限レポート

## Phase 2: 検証強化
- コストモデル
- 流動性フィルタ
- サブ期間評価
- 感度分析
- 実験管理

## Phase 3: 改良
- Cfull rolling / expanding
- prior factor 再設計
- スコア比例ウェイト
- regime 分岐
- 複数ラグ

---

## prior factor の実務的拡張候補

将来的には V0 の軸を追加できる設計にしておくこと

候補:
- 半導体 vs 非半導体
- 金利敏感 vs 非敏感
- 円安恩恵 vs 円高恩恵
- エネルギー感応度
- 輸出主導 vs 内需主導

ただし初期段階では論文どおりの3本で固定すること

---

## 成果物としてほしいもの

## 1. コード
- 再現可能なPython実装
- CLI付き
- テスト付き

## 2. レポート
- 論文再現レポート
- ベースライン比較
- パラメータ感度分析
- コスト反映版
- 流動性制約版

## 3. 出力データ
- 日次シグナル
- 日次ポートフォリオ
- 日次損益
- 指標集計表
- 実験設定ログ

---

## 期待する最初のゴール

最初のマイルストーンはこれ

### Milestone 1
「論文再現の最小版」
- データが取得できる
- モデルが動く
- MOM / PCA_PLAIN / PCA_SUB の比較が出る
- 累積リターン、AR、RISK、R/R、MDD が出る

### Milestone 2
「現実的な検証」
- コストを入れる
- 日本ETF流動性制約を入れる
- 成績がどれだけ崩れるか確認する

### Milestone 3
「改良実験」
- L, λ, K, Cfull を系統比較
- out-of-sample で評価する

---

## Claude Code への具体指示

以下の方針で作業してください。

1. まずはリポジトリ初期化とディレクトリ構成を作る
2. 最小限のデータ取得・保存パイプラインを作る
3. 日米営業日の整列ロジックを実装する
4. リターン計算・標準化を実装する
5. MOM を実装する
6. PCA_PLAIN を実装する
7. PCA_SUB を実装する
8. バックテストと評価指標を実装する
9. 論文再現レポートを出力する
10. その後、コストと流動性制約を追加する
11. その後、感度分析を追加する

### 重要
- 各段階で小さく動かして確認する
- 実装完了ごとに README と関連 docs を更新する
- 実験条件を config に残す
- 途中の仮定は必ず明文化する
- 未来情報混入がないか毎回チェックする

---

## 最後に

このプロジェクトで大事なのは、
**「論文の見た目の数字を再現すること」より、「その数字が実務で残るのかを、段階的に壊しながら検証すること」** です。

したがって、実装は以下の順番で進めてください。

1. 論文を素直に再現
2. コストを入れる
3. 流動性を入れる
4. パラメータ感度を見る
5. それでも残るかを確認する

この順番を守って進めてください。
