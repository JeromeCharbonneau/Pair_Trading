# Feature Set — Corn / Wheat (Day 3)

**Primary panel:** Bloomberg clean Corn/Wheat (`data/clean/corn_wheat_panel_bloomberg.*`)  
**Outputs:** `data/features/features.parquet`, `features.csv`, `feature_manifest.json`  
**Rebuild:** `python scripts/build_day3_features.py` (deterministic given the same clean input)  
**Calendar:** `CME_ag_futures_trading_date`

Yahoo continuous remains a Day-1/2 provisional comparison series only. Day-3 features and models use Bloomberg.

## Paper-derived feature

| Field | Value |
|---|---|
| **Feature name** | `ratio_zscore_50` |
| **Category** | Internal / Endogenous |
| **Type** | Statistical mean-reversion feature |
| **Source** | van Unen (2023) |
| **Citation** | van Unen, Q. (2023). *Pairs Trading in Agricultural Commodity Futures Markets*. BSc thesis, Erasmus University Rotterdam, §4.2. Local copy: `Context/Thesis-Final-Draft (1).pdf` |

### Construction (reproduced)

```text
ratio_t = Corn_t / Wheat_t
rolling_mean_50_t = mean(ratio_{t-49:t})
rolling_std_50_t  = std(ratio_{t-49:t})
ratio_zscore_50_t = (ratio_t - rolling_mean_50_t) / rolling_std_50_t
```

- Trailing 50 **trading** days via `pandas.Series.rolling(50)` — **never** `center=True`.
- `min_periods=50` → first 49 observations are NaN (no fill).
- Stored as a **continuous** numeric feature. The paper’s ±1.5σ entry rules and dual-signal-within-10-days filter are trading rules, not this feature.

## Feature table

| Feature | Category | Risk-model bucket | Construction | Rationale |
|---|---|---|---|---|
| `ratio_zscore_50` | Internal / Endogenous | Statistical | 50d rolling z-score of Corn/Wheat price ratio | Paper-derived mean-reversion signal (van Unen 2023 §4.2) |
| `ratio` | Internal / Endogenous | Fundamental-analog | Corn / Wheat | Relative price level of the pair |
| `log_spread` | Internal / Endogenous | Statistical | log(Corn) − β̂₆₀·log(Wheat), trailing OLS β | Cointegration residual proxy; **no full-sample β** |
| `spread_zscore_60` | Internal / Endogenous | Statistical | 60d z-score of `log_spread` | Normalized spread for mean-reversion |
| `rv_corn_20` | Internal / Endogenous | Statistical | 20d std of Corn returns | Leg risk-regime proxy |
| `rv_wheat_20` | Internal / Endogenous | Statistical | 20d std of Wheat returns | Leg risk-regime proxy |
| `ret_ratio_1d` | Internal / Endogenous | Statistical | 1d pct change of `ratio` | Short-horizon ratio momentum (known at close of *t*) |

**External / weather features:** not included tonight. The project thesis is pairs mean-reversion on prices; weather/ENSO would be a natural Day-4+ overlay, not required for the paper feature.

## No look-ahead

- All rolling statistics are trailing (include *t*, exclude *t+1…*).
- Rolling hedge β uses only the trailing 60 observations ending at *t*.
- Modeling target `target_ret_ratio_1d_fwd` = `ratio_{t+1}/ratio_t − 1` is created with `.shift(-1)` and is **excluded** from the feature matrix `X` at fit time.
- Warm-up NaNs are dropped only when fitting models; they are not forward-filled in the feature file.

## Aux column (not in X)

| Column | Role |
|---|---|
| `rolling_beta_60` | Trailing OLS hedge ratio for Day-4 portfolio weights — **excluded from model features** |

## Modeling / validation choices (Day 3 finalized)

| Choice | Decision | Why |
|---|---|---|
| Universe / pooling | Specialized single-pair time-series model | N=2 Corn/Wheat; not a large homogeneous cross-section |
| Target | Next-day ratio return (raw) | Directional futures-style signal |
| Primary metric | Hit rate / directional accuracy | Matches Day-3 futures positioning guidance |
| Secondary metric | Time-series Spearman IC | Signal quality independent of sizing |
| Models | Elastic Net + LightGBM (+ equal-weight ensemble on Day 4) | Interpretable β′F vs nonlinear check |
| Validation | Expanding `TimeSeriesSplit` (5 outer folds) on development; EN inner `TimeSeriesSplit(3)` | No random k-fold / no shuffle |
| Holdout | Final 20% chronological | Untouched for hyperparams, thresholds, features |
| Label overlap | Disclosed; not purged | 1-day forward labels share price levels |

Day-4 portfolio construction, costs, and results: `docs/DAY4_METHOD.md`, `docs/DAY4_RESULTS.md`.

Cointegration analysis from Days 1–2 is unchanged and remains a separate diagnostic — these features are additive.
