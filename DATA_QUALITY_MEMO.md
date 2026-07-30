# Data Quality Memo — Corn / Wheat (Day 2)

**Date:** 2026-07-29  
**Pair:** Corn (`ZC=F`) / Wheat (`ZW=F`)  
**Data source flag:** `yahoo_continuous_provisional`  
**Raw pull:** `2026-07-29T152437Z_continuous` → `commodity_prices.csv`  
**Calendar convention:** `CME_ag_futures_trading_date` (CME ag futures trading/settlement dates). External data must be asof-aligned to this calendar before joining.

## What we used
Daily Close prices from Yahoo continuous futures, frozen on disk (not re-downloaded over the same path). This is a **provisional** research series until Bloomberg (or another vendor) provides an explicitly rolled continuous history.

## Missing values
- Mechanism (practical call): sparse vendor holes / holiday misalignment — treated like **MCAR-style feed gaps**, not informative MNAR silence.
- **Method:** listwise delete (keep a row only if both legs are present, finite, and positive).
- **No forward-fill and no interpolation** (avoids inventing prices and avoids non-causal fills).
- Dropped 4 row(s): 2018-12-17 (Corn=NaN), 2018-12-18 (Corn=NaN), 2023-11-23 (Corn=NaN, Wheat=NaN), 2025-07-04 (Corn=NaN, Wheat=NaN).
- Clean sample: **2,510** rows from 2016-07-29 to 2026-07-29.

## Outliers / winsorization
- Extreme daily returns (|r| > 10%) were **flagged** and **kept**.
- **No winsorization** on prices: the pairs spread and cointegration test use the full return path; capping tails would distort the signal. Cross-sectional winsorization from the slides does not apply to N=2.

## Futures rolls
- Yahoo `=F` roll method/dates are opaque.
- Per-contract Yahoo history is too incomplete for a full Panama rebuild (expired months missing).
- **Intended later rule (Bloomberg):** Panama / back-adjusted continuous with one consistent roll rule (e.g. volume crossover or fixed days-to-expiry), applied identically to Corn and Wheat.

## Outputs / lineage
- `data/clean/corn_wheat_panel.csv` and `.parquet`
- `data/clean/corn_wheat_log_prices_clean.csv` and `.parquet`
- `data/clean/lineage.json` (relative paths, rules, dropped dates, extreme flags, calendar convention)

## Bottom line
We have an analysis-ready **provisional** Corn/Wheat panel suitable to proceed, with limitations disclosed. After Bloomberg arrives, save a new raw folder, flip `DATA_SOURCE`, re-run this cleaning, and re-test cointegration. Do not reuse full-sample hedge ratios in a backtest without in-sample re-estimation.
