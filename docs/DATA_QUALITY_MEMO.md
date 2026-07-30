# Data Quality Memo — Corn / Wheat

**Date:** 2026-07-30 (updated for Bloomberg primary)  
**Pair:** Corn / Wheat  
**Primary research source:** Bloomberg continuous workbook (`DATA_SOURCE=bloomberg`)  
**Provisional comparison source:** Yahoo continuous `ZC=F` / `ZW=F`  
**Calendar convention:** `CME_ag_futures_trading_date`

## Bloomberg clean panel (primary)

| Item | Value |
|---|---|
| Raw pull | `2026-07-29T234058Z_bloomberg` → `data_trading.xlsx` |
| Raw rows | 6,836 |
| Clean rows | **6,692** |
| Cleaning rules | both legs finite & positive; **no** forward-fill; **no** interpolation; **no** price winsorization |
| Extreme returns | flagged, not auto-dropped |
| Cointegration (full clean sample) | Engle–Granger p ≈ 1.0e-4; hedge β̂ (log Corn on Wheat) ≈ 1.03; return corr ≈ 0.59 |
| Lineage | `data/clean/lineage_bloomberg.json` |
| Outputs | `corn_wheat_panel_bloomberg.*`, `corn_wheat_log_prices_bloomberg.*` |

**Roll limitation (disclosed):** Bloomberg sheets are vendor-defined continuous series. Roll dates/method are not reconstructed from per-contract Panama rules.

## Yahoo clean panel (provisional comparison)

| Item | Value |
|---|---|
| Raw pull | `2026-07-29T152437Z_continuous` |
| Clean rows | 2,510 (2016-07-29 → 2026-07-29) |
| Dropped | 4 rows (vendor holes / holiday misalignment) |
| Lineage | `data/clean/lineage.json` |

Yahoo was used for the Day-1 pair screen. It is **not** the Day 3–4 research series.

### Yahoo missing values
- Mechanism: sparse vendor holes / holiday misalignment (MCAR-style feed gaps).
- Method: listwise delete (both legs present, finite, positive).
- No forward-fill and no interpolation.

### Outliers
- Extreme daily returns (|r| > 10%) flagged and kept on both sources.
- No price winsorization (pairs/cointegration use full path; N=2 cross-sectional winsorize N/A).

## Bottom line

Bloomberg is the **primary** cleaned research panel for feature engineering, modeling, and Day-4 backtests. Yahoo remains a provisional comparison. Do not reuse full-sample hedge ratios in a backtest without in-sample / trailing re-estimation (`rolling_beta_60` is used in Day 4).
