# 最終判断レポート: 主戦力候補に届くか

## Experiment 1: Walk-Forward K
K distribution: {3: 1484, 4: 411, 5: 717}

## Exp1 Results

| Config | Full Net AR (%) | Full Net R/R | Test Net AR (%) | Test Net R/R | BE (bp) |
|--------|---------------:|------------:|---------------:|------------:|--------:|
| K=3 fixed | 4.78 | 1.11 | 3.91 | 0.99 | 8.6 |
| WF-K | 1.22 | 0.34 | -0.26 | -0.06 | 4.9 |

## Experiment 2: Composite Filters

| Config | Full Net AR (%) | Full Net R/R | Test Net AR (%) | Test Net R/R | BE (bp) |
|--------|---------------:|------------:|---------------:|------------:|--------:|
| A (strength only) | 3.75 | 0.93 | 3.34 | 0.86 | 7.8 |
| B (strength+spread) | 3.75 | 0.93 | 3.34 | 0.86 | 7.8 |
| C (strength+factor) | 3.06 | 1.02 | 3.04 | 1.13 | 10.5 |
| D (all combined) | 3.06 | 1.02 | 3.04 | 1.13 | 10.5 |

## Experiment 3: Weight Schemes

| Config | Full Net AR (%) | Full Net R/R | Test Net AR (%) | Test Net R/R | BE (bp) |
|--------|---------------:|------------:|---------------:|------------:|--------:|
| A (equal) | 4.78 | 1.11 | 3.91 | 0.99 | 8.6 |
| B (rank) | 5.58 | 1.19 | 4.54 | 1.04 | 9.4 |
| C (sig/vol) | 5.41 | 1.21 | 3.86 | 0.94 | 9.3 |
| A (equal q0.2) | 6.48 | 1.31 | 5.51 | 1.16 | 10.4 |
| B (rank q0.2) | 6.55 | 1.25 | 5.50 | 1.10 | 10.5 |

## Final Integration

| Config | Full Net AR (%) | Full Net R/R | Test Net AR (%) | Test Net R/R | BE (bp) |
|--------|---------------:|------------:|---------------:|------------:|--------:|
| WF-K + rank q0.2 | 2.51 | 0.54 | 2.31 | 0.41 | 6.2 |
| WF-K + equal q0.2 | 2.45 | 0.55 | 2.54 | 0.47 | 6.2 |
| K3 + rank q0.2 | 6.55 | 1.25 | 5.50 | 1.10 | 10.5 |

## 最終判断

Best combination: **K3 + rank q0.2**
- Test Net AR: 5.50%
- Test Net R/R: 1.10

**判定: PARTIAL — サブ戦略候補として有効。主戦力には不足。**