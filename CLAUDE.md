# PCA_SUB Project

## Overview
Reproduction and practical validation of "Lead-lag strategies for Japanese and U.S. sectors using subspace regularization PCA" (Nakagawa et al., 2026, SIG-FIN-036).

## Key Commands
```bash
pip install -e ".[dev]"
pytest tests/
python -m src.cli fetch-data --config configs/base.yaml
python -m src.cli run-backtest --config configs/paper_reproduction.yaml
```

## Architecture
- `src/` — core library modules
- `configs/` — YAML experiment configs
- `tests/` — pytest tests
- `results/` — output signals, portfolios, metrics, plots
- `reports/` — generated experiment reports

## Critical Implementation Notes
- **No future information leakage**: all signals use only past data (window W_t = {t-L, ..., t-1})
- **Calendar alignment**: US t-day close → JP t+1 open-to-close. Friday US → Monday JP.
- **Paper defaults**: L=60, λ=0.9, K=3, q=0.3, Cfull=2010-2014
- US ETFs: 11 Select Sector SPDR | JP ETFs: 17 TOPIX-17
- Eigenvectors ordered by descending eigenvalue
- Equal-weight long/short with zero net exposure
