# Day 4 Results — Corn / Wheat Pairs

**Run artifacts:** `data/results/day4_metrics.json`, `day4_predictions.*`, `day4_backtest_daily.*`, `day4_residual_diagnostics.json`  
**Preferred config (stability rule):** `en_threshold` — Elastic Net + no-trade threshold (dev-selected p75)

## 1. Time-series CV (development only)

| Metric | Mean | Std |
|---|---|---|
| Elastic Net hit rate | 51.45% | 1.96% |
| Elastic Net Spearman IC | 0.059 | 0.028 |
| LightGBM hit rate | 50.68% | 1.72% |
| LightGBM Spearman IC | 0.019 | 0.015 |

Holdout window: **2021-05-05 → 2026-07-28** (n=1,315). Not used for threshold or hyperparameter selection.

Fold-level dates and metrics are in `data/features/day3_model_metrics.json` and `data/results/day4_metrics.json`.

## 2. Holdout portfolio results (net of 2 bps)

| Config | Net Sharpe | Max DD | Ann. turnover | Active hit rate | Cum. return |
|---|---|---|---|---|---|
| **en_threshold (preferred)** | **0.42** | **-7.5%** | **7.5** | 51.2% | +16.1% |
| en_sign | -0.12 | -31.4% | 50.4 | 50.0% | -18.2% |
| en_capped | 0.05 | -34.6% | 51.2 | 49.8% | -3.6% |
| lgbm_sign | 0.27 | -34.8% | 82.2 | 49.8% | +18.4% |
| lgbm_threshold | -0.10 | -42.4% | 66.2 | 49.6% | -14.3% |
| lgbm_capped | 0.26 | -36.5% | 84.2 | 49.8% | +18.2% |
| ensemble_sign | 0.01 | -48.8% | 55.4 | 49.9% | -8.0% |
| ensemble_threshold | 0.01 | -48.8% | 55.6 | 49.9% | -7.7% |
| bench_zscore_mr | 0.12 | -28.0% | 22.9 | 50.0% | +2.6% |
| bench_flat | n/a | 0 | 0 | n/a | 0 |

### Cost sensitivity (preferred `en_threshold`)

| Cost | Sharpe |
|---|---|
| 0 bps (gross) | 0.44 |
| 2 bps | 0.42 |
| 5 bps | 0.39 |

Low turnover from the threshold rule is why net Sharpe decays only mildly with costs.

### Sub-period net Sharpe @ 2 bps (`en_threshold`)

| Year | Sharpe |
|---|---|
| 2021 (partial) | 1.19 |
| 2022 | 0.53 |
| 2023 | **-0.81** |
| 2024 | **-0.22** |
| 2025 | unstable / undefined in sample slice |
| 2026 (partial) | 1.76 |

Performance is **not stable across years**.

## 3. Hit rate → IC and breadth

For preferred active days:

- Active hit rate ≈ 51.2% → directional IC \(= 2\cdot\text{hit}-1 ≈ 0.024\)
- Effective breadth (position changes / year) ≈ **14.8**
- IR approx \(= \mathrm{IC}\sqrt{B} ≈ 0.09\)

Realized net Sharpe @2bps ≈ **0.42**, which is **higher** than the IR approximation. Reasons they differ:

- IR approx ignores magnitude, volatility timing, and the hedge-ratio portfolio transform.
- Breadth based on position changes is conservative vs AR(1) \(n_{\text{eff}}\) (~241/year).
- A few large favorable moves can inflate Sharpe relative to sign-only IC math.
- Multiple-testing: Deflated Sharpe Ratio for the preferred holdout Sharpe across ~13 trials is ≈ **0** (not statistically distinguishable from the expected maximum under the null).

## 4. Residual diagnostics (holdout Elastic Net)

| Test | Result |
|---|---|
| Durbin–Watson | 1.90 — no strong AR(1) |
| Ljung–Box | reject at lags 5 and 10 (p<0.05); lag 1 and 20 weaker |
| Breusch–Pagan | borderline (p≈0.08) |
| White | reject (p≪0.05) → heteroskedasticity / changing volatility |
| Jarque–Bera | reject (p≈0) → fat tails |

**Interpretation:** residuals show fat tails and heteroskedasticity, plus some multi-lag serial structure. That may indicate model misspecification or volatility clustering — **not** an automatic license to add features on this holdout. Any new feature is a future hypothesis needing a new untouched period.

Figures: `data/results/figures/elastic_net_*.png`.

## 5. Robustness (summary)

- EN vs LGBM vs ensemble: thresholded EN dominates on stability and turnover; LGBM sign has positive Sharpe but very high turnover and deep drawdowns.
- Sign vs threshold: sign strategies churn and often lose after costs; threshold is essential.
- Hedge windows 40/60/90: preferred stays positive (Sharpes ≈ 0.46 / 0.42 / 0.58) — directionally robust, point estimates move.
- Extra 1-day prediction delay: Sharpe falls to ≈ 0.24 (edge decays with latency).
- Drop top 1% |gross| days: still evaluated in metrics JSON (sensitivity to outliers).

## 6. Final answer

### Does the small predictive advantage survive hedge-ratio-neutral construction, turnover, and transaction costs?

**Partially, and only for a low-turnover Elastic Net threshold rule — with important caveats.**

On the untouched holdout, `en_threshold` delivers a modest **net Sharpe ≈ 0.42** at 2 bps and ≈ 0.39 at 5 bps, with a relatively contained max drawdown (~7.5%) and low annualized turnover (~7.5). That is better than always-flat and better than the simple z-score mean-reversion benchmark on this window.

However:

1. **Hit rate alone is not success** — active hit rate is only ~51%, barely above a coin flip.
2. **Sub-period instability** — 2023–2024 net Sharpes are negative.
3. **Multiple-testing** — Deflated Sharpe is ~0 after accounting for the pre-registered grid.
4. **High-turnover rules fail** — EN/LGBM/ensemble sign and capped strategies do **not** reliably survive costs.
5. **Residuals** still show fat tails, heteroskedasticity, and some serial structure.

### Preferred model and rule

- **Preferred:** Elastic Net + **p75 no-trade threshold** (`en_threshold`)
- **Why:** best stability score among pre-registered configs (sub-period behavior, drawdown, turnover, cost robustness), not because it had the single highest Sharpe among all tinkering.

### What did not work

- Always-on sign strategies (especially LightGBM) — turnover overwhelms a weak edge.
- Ensemble averaging did not improve holdout net performance here.
- Relying on hit rate > 50% without portfolio/cost analysis.

### What remains uncertain

- Whether the thresholded EN edge persists on a **new** post-2026 sample.
- True economic costs including rolls and stress-period spreads.
- Sensitivity to Bloomberg’s opaque continuous-contract construction.

### What to test next (new untouched dataset required)

- Re-estimate threshold and models ending before a fresh OOS window.
- Explicit Panama/per-contract rolls if available.
- Volatility targeting / heteroskedasticity-aware sizing as a **pre-registered** hypothesis (not fit on this holdout).
- Optional external ag weather features as a separate experiment with its own holdout.
