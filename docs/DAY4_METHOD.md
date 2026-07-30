# Day 4 Method — Corn / Wheat Pairs Backtest

**Primary data:** Bloomberg clean Corn/Wheat (`data/clean/corn_wheat_panel_bloomberg.*`)  
**Features:** `data/features/features.parquet`  
**Scripts:** `scripts/build_day4_backtest.py`, `scripts/run_day4_diagnostics.py`  
**Calendar:** `CME_ag_futures_trading_date`

Yahoo continuous futures remain a **provisional Day-1 comparison** source only.

## Validation (Day 3 finalize)

- Chronological **80% development / 20% holdout** on rows with complete features + target.
- Holdout is **untouched** for hyperparameters, thresholds, hedge window, and feature selection.
- Outer validation: `TimeSeriesSplit(n_splits=5)` on development only (expanding train, forward validation, **no shuffle**).
- Inner Elastic Net search: `ElasticNetCV` with `TimeSeriesSplit(n_splits=3)`.
- LightGBM uses fixed Day-3 hyperparameters (no nested LGBM search).
- **Label overlap:** the target is a one-day forward return, so adjacent labels share a price level. We preserve strict chronology and do **not** purge (purging would discard most daily pairs observations). This is a disclosed limitation.

## Predictions

Holdout predictions from:

1. Elastic Net (fit on full development)  
2. LightGBM (fit on full development)  
3. Ensemble: equal-weight average of **training-standardized** EN and LGBM predictions (moments from development predictions only)

## Portfolio construction (two-asset, not cross-sectional)

At decision date \(t\):

- Positive prediction → long Corn/Wheat ratio; negative → short the ratio.
- Raw weights: \(w^{c,\text{raw}}_t = s_t\), \(w^{w,\text{raw}}_t = -s_t \cdot \beta_t\)
- \(\beta_t\) = trailing 60-day OLS hedge ratio of log Corn on log Wheat (point-in-time).
- Normalize: \(|w^c_t| + |w^w_t| = 1\) when active.
- **Lag weights by one day** so the signal at \(t\) earns asset returns at \(t+1\).

Gross return:

\[
r^{p}_{t} = w^{c}_{t-1}\, r^{\text{Corn}}_{t} + w^{w}_{t-1}\, r^{\text{Wheat}}_{t}
\]

## Signal rules (pre-registered)

| Rule | Definition |
|---|---|
| A Sign | \(s=\mathrm{sign}(\hat y)\) |
| B Threshold | \(s=0\) if \(\|\hat y\|<\tau\); else sign. \(\tau\) chosen from {p25,p50,p75} of development OOS \(\|\hat y\|\) by net Sharpe@2bps |
| C Capped | train-standardized prediction clipped to \([-1,1]\) |
| D Ensemble | sign/threshold on ensemble score |
| Bench MR | \(s=-\mathrm{sign}(\texttt{ratio\_zscore\_50})\) |
| Bench flat | \(s=0\) |

## Transaction costs

\[
\text{turnover}_t = \tfrac12\big(|\Delta w^c_t| + |\Delta w^w_t|\big)
\]

\[
r^{\text{net}}_t = r^{\text{gross}}_t - \text{turnover}^{\text{aligned}}_t \cdot c
\]

Cost scenarios: **0 / 2 / 5 bps** per unit turnover.

This is a **linear implementation-shortfall approximation**. We do **not** invent square-root market impact without ADV/volume.

### Cost discussion (qualitative)

- **Commissions:** futures commissions are typically small vs equity; linear bps proxy includes them roughly.
- **Bid–ask:** agricultural futures spreads vary by contract month/liquidity; linear costs understate stress regimes.
- **Slippage:** market orders around economic releases can exceed 2–5 bps.
- **Contract rolls:** continuous Bloomberg series embed vendor roll rules; true roll P&L may differ from backtest.
- **Margin / leverage:** futures are margined; reported returns assume notional weights sum to 1 in absolute value, not capital-at-risk Sharpe under margin.
- **Vendor continuous rolls:** Bloomberg sheet construction remains opaque — a core limitation carried from Day 1–2.

## Metrics

Annualization uses 252 trading days. For this market-neutral pairs book, Sharpe vs a zero benchmark is approximately the Information Ratio.

## Residual diagnostics

Run on holdout Elastic Net residuals (ensemble optional). Any rejected null is interpreted as a **future hypothesis**, not a license to add features using the same holdout.

## Reproducibility

```bash
python scripts/build_day3_features.py
python scripts/build_day4_backtest.py
python scripts/run_day4_diagnostics.py
python -m pytest tests/test_day4_pipeline.py -q
```
