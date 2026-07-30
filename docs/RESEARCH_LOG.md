# Research Log — Days 3–4

**Project:** Corn / Wheat pairs trading  
**Primary data:** Bloomberg continuous clean panel  
**Last updated:** 2026-07-30  

## Experiment design (locked before holdout reporting)

| Item | Setting |
|---|---|
| Features | ratio, ratio_zscore_50, log_spread, spread_zscore_60, rv_corn_20, rv_wheat_20, ret_ratio_1d |
| Target | target_ret_ratio_1d_fwd |
| Split | 80% development / 20% holdout chronological |
| Outer CV | TimeSeriesSplit n_splits=5 on development |
| Inner EN CV | TimeSeriesSplit n_splits=3 |
| LGBM | fixed params (n_estimators=200, num_leaves=15, lr=0.05, …) |
| Threshold grid | p25 / p50 / p75 of \|dev OOS EN prediction\| |
| Threshold selection metric | net Sharpe @ 2 bps on development OOS stack |
| Hedge beta | trailing 60d OLS (robustness 40/60/90) |
| Costs | 0 / 2 / 5 bps per unit turnover (linear) |
| Preferred-selection rule | stability score (not max Sharpe alone) |

## Total configurations tested

**Primary reported configs: 13**

- 10 core holdout strategies (EN×3, LGBM×3, Ensemble×2, 2 benchmarks)
- 3 threshold probes on development (p25/p50/p75)

Robustness extras (logged, not primary selection): hedge windows 40/60/90; delay-1; exclude top 1% |gross|; nearby thresholds on holdout (reporting only).

### Multiple-testing

With ~13 primary trials, a single holdout Sharpe is inflated under the null. We compute a Bailey–López de Prado **Deflated Sharpe Ratio** for the preferred strategy; it is ≈ 0, i.e. **not** statistically distinguished from the expected maximum Sharpe under noise given the trial count. This does **not** erase economic discussion of turnover/drawdown, but it blocks strong claims of discovery.

## Threshold probes (development only)

| Probe | Threshold | Dev net Sharpe@2bps | Decision |
|---|---|---|---|
| p25 | (see metrics JSON) | lower | rejected |
| p50 | (see metrics JSON) | lower | rejected |
| **p75** | **≈ 0.001108** | **best** | **selected** |

## Holdout evaluations

| Config | Decision | Reason |
|---|---|---|
| en_threshold | **preferred** | Best stability (turnover, MDD, cost robustness, sub-period mix) |
| en_sign | rejected_vs_preferred | High turnover; negative net Sharpe@2bps |
| en_capped | rejected_vs_preferred | High turnover; weak Sharpe; deep DD |
| lgbm_sign | rejected_vs_preferred | Positive Sharpe but extreme turnover/DD |
| lgbm_threshold | rejected_vs_preferred | Negative net Sharpe; deep DD |
| lgbm_capped | rejected_vs_preferred | High turnover/DD |
| ensemble_sign | rejected_vs_preferred | ~0 Sharpe; deep DD |
| ensemble_threshold | rejected_vs_preferred | ~0 Sharpe; deep DD |
| bench_zscore_mr | benchmark | Beats flat slightly; loses to preferred |
| bench_flat | benchmark | Zero return baseline |

Exact numeric results: `data/results/day4_metrics.json` and `data/results/day4_research_log.json`.

## Machine-readable log

See `data/results/day4_research_log.json` for dated entries with metrics and keep/reject decisions.
