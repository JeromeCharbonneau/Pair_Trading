# Provenance log — MMF Workshop pairs-trading project

Last updated: 2026-07-29 (local) — Bloomberg workbook ingest + Yahoo comparison

## What was pulled

| Pull ID | Kind | Source | When | What |
|---|---|---|---|---|
| `2026-07-29T152437Z_continuous` | Vendor continuous futures (`=F`) | Yahoo Finance via `yfinance` | 2026-07-29 ~15:24 local | Daily Close for 14 commodity continuous symbols, ~10y |
| `2026-07-29T203057Z_contracts` | Per-contract CBT months | Yahoo Finance via `yfinance` | 2026-07-29 ~20:30 UTC | Individual Corn (`ZC*.CBT`) / Wheat (`ZW*.CBT`) contracts with data at pull time |
| `2026-07-29T234058Z_bloomberg` | Multi-sheet continuous futures workbook | Bloomberg export (`data_trading.xlsx`) | 2026-07-29 | 15 commodity sheets (~2000–2026); Corn uses Last Price, others Close |

Canonical paths:
- Continuous screen: `data/raw/2026-07-29T152437Z_continuous/`
- Per-contract (preferred for Day-2 rolls): `data/raw/2026-07-29T203057Z_contracts/`
- Bloomberg workbook: `data/raw/2026-07-29T234058Z_bloomberg/data_trading.xlsx`
- Cleaning outputs (not raw): `data/clean/`

Endpoint pattern:
- Continuous: `yf.download(symbols, period="10y", interval="1d", auto_adjust=False)`
- Contracts: `yf.download("<ROOT><MONTH><YY>.CBT", period="max", auto_adjust=False)`

## Immutability rule

Raw pulls are written once under a unique pull-id folder and are **not** overwritten on notebook re-runs. Analysis loads from these folders; cleaning writes new files under `data/clean/`.

## Point-in-time (PIT) traps identified (unresolved unless noted)

1. **As-of-today continuous series**  
   `ZC=F` / `ZW=F` (and the other `=F` symbols) are whatever Yahoo serves *today*. Re-pulling later can revise history. We freeze one pull on disk; that freezes *our copy*, not Yahoo’s construction methodology.

2. **Opaque continuous-contract construction**  
   Roll dates, roll method (front-month concat vs Panama / ratio back-adjust), and which deferred month is “front” are vendor-defined. Day-1 continuous screen is **not** a point-in-time reconstructible continuous series.

3. **Calendar / session convention (canonical)**  
   Canonical date convention for this project: **CME agricultural futures trading/settlement date** (`CME_ag_futures_trading_date` in `pipeline_utils.py` and `lineage.json`).  
   Any future external series (weather, macro, equities, Bloomberg fields) must be **point-in-time asof-aligned** to this trading-date calendar before joining — never a naive same-`date` merge across clocks.

4. **No vendor “as-of” / vintage stamp in the API response**  
   `yfinance` does not return a filing/as-of vintage for these closes. Provenance is only our pull timestamp + cached files.

## Survivorship / listing traps identified (unresolved)

1. **Expired contract deletion on Yahoo (critical for rolls)**  
   Scanning Corn/Wheat months `H/K/N/U/Z` for years 2016–2027 shows Yahoo returns history only for **currently listed** (and a few recently expired) contracts. Older expired months 404 / empty.  
   **Consequence:** a full 10-year Panama/back-adjusted continuous curve **cannot** be rebuilt from this Yahoo per-contract pull alone. Unresolved without Bloomberg / CME / another archive.

2. **Contracts with data in the 2026-07-29 contracts pull (universe as-of pull time)**  
   Corn: `ZCN25.CBT`, `ZCU26.CBT`, `ZCZ26.CBT`, `ZCH27.CBT`, `ZCK27.CBT`, `ZCN27.CBT`, `ZCU27.CBT`, `ZCZ27.CBT`  
   Wheat: `ZWN25.CBT`, `ZWU26.CBT`, `ZWZ26.CBT`, `ZWH27.CBT`, `ZWK27.CBT`, `ZWN27.CBT`, `ZWU27.CBT`, `ZWZ27.CBT`  
   This is an **as-of-today listed-contract universe**, not the historical set of all contracts that traded over 2016–2026.

3. **Convenience commodity universe**  
   The 14-name continuous screen is a hand-picked liquid set (no delisted commodity products, no systematic inclusion rule). Pair discovery is conditioned on that convenience sample.

4. **Backfill / instant-history risk on listed deferreds**  
   Deferred months (e.g. 2027) appear with multi-year Yahoo histories. We have not independently verified when each contract became listed/tradable; early prints may be backfilled or thinly traded. Treat pre-liquidity history as suspect until Day-2 QC.

## Likely direction / magnitude of bias (disclosure)

| Bias | Direction if ignored | Rough note |
|---|---|---|
| Using `=F` continuous without documenting rolls | Distorts returns/spreads around roll dates; can manufacture or kill mean-reversion in the Corn–Wheat spread | Material for pairs; methodology unknown |
| Building rolls only from today’s listed contracts | Short / incomplete continuous history; selection toward currently liquid months | Severe for 10y backtest length |
| Convenience 14-name universe | Pair screen may miss better cointegrated pairs; cherry-picking risk | Moderate for narrative; disclose |

## Future Bloomberg drop-in (planned; not pulled yet)

When Bloomberg Corn/Wheat (preferably per-contract or explicitly rolled continuous) is available:

1. Write to a **new** immutable folder: `data/raw/<UTC-timestamp>_bloomberg/` (never overwrite Yahoo pulls).
2. Include a `manifest.json` with pull time, fields, contract/roll conventions.
3. Set notebook `DATA_SOURCE = "bloomberg"` and point the loader at that pull id.
4. Re-run Day 2 cleaning unchanged; then re-run cointegration / spread work on the new clean panel.
5. Append a row to this provenance table for the Bloomberg pull.

Environment pin: see `requirements.txt` (Day 1 reproducibility).

## Day 2 status

- Clean Corn/Wheat panel + lineage: `data/clean/` (see `lineage.json`)
- Data-quality memo: `docs/DATA_QUALITY_MEMO.md`
- Full Panama roll on expired contracts: still blocked on Yahoo; deferred to Bloomberg re-run

## Pair selection note

Working pair for the project: **Corn / Wheat**, chosen for economic link + cointegration + return correlation on the frozen continuous screen — provisional until a Bloomberg (or otherwise explicitly rolled) continuous series replaces the Yahoo `=F` feed.
