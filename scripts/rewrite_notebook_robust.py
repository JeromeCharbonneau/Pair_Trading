#!/usr/bin/env python3
"""Rewrite pairs_trading_investigation.ipynb into clear Day1/Day2 sections."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
nb_path = ROOT / "pairs_trading_investigation.ipynb"


def lines(text: str) -> list[str]:
    if not text.endswith("\n"):
        text += "\n"
    return [ln + "\n" for ln in text.splitlines(True)]


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": lines(text)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": lines(text),
    }


cells: list[dict] = []

cells.append(
    md(
        """# Corn / Wheat pairs trading — Days 1–2

Pipeline: **Source → Validate → Screen → Clean → Validate clean → Cointegration**.

- Raw data under `data/raw/` is **immutable** (never overwrite).
- Clean outputs under `data/clean/` (CSV + Parquet).
- Canonical calendar: **CME ag futures trading date** (`pipeline_utils.CALENDAR_CONVENTION`).
- Yahoo continuous `ZC=F` / `ZW=F` is **provisional** until Bloomberg rolled series arrives.
"""
    )
)

# --- 1. Day 1 sourcing ---
cells.append(md("## 1. Day 1 — Data sourcing"))
cells.append(
    code(
        '''!pip install -q -r requirements.txt

from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure project root is importable when the kernel cwd differs slightly
ROOT = Path.cwd()
if not (ROOT / "pipeline_utils.py").exists():
    raise FileNotFoundError(
        "Run this notebook with working directory = project root "
        "(folder containing pipeline_utils.py and data/)."
    )
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline_utils import (
    CALENDAR_CONVENTION,
    CALENDAR_NOTE,
    CLEAN_ROOT,
    DEFAULT_CONTINUOUS_PULL,
    DEFAULT_CONTRACTS_PULL,
    PAIR_COLS,
    RAW_ROOT,
    clean_corn_wheat,
    flag_extreme_returns,
    load_clean_log_pair,
    load_clean_pair,
    load_continuous_prices,
    load_contracts_manifest,
    to_rel,
    validate_clean_pair,
    validate_price_panel,
)

# Flip to "bloomberg" later and point ACTIVE_CONTINUOUS_PULL at the new raw folder.
DATA_SOURCE = "yahoo_continuous_provisional"
ACTIVE_CONTINUOUS_PULL = DEFAULT_CONTINUOUS_PULL
ACTIVE_CONTRACTS_PULL = DEFAULT_CONTRACTS_PULL

print(f"DATA_SOURCE: {DATA_SOURCE}")
print(f"Calendar:    {CALENDAR_CONVENTION}")
print(CALENDAR_NOTE)
print(f"Raw root:    {to_rel(RAW_ROOT)}")
print(f"Clean root:  {to_rel(CLEAN_ROOT)}")

# --- Layer A: raw continuous futures ---
prices = load_continuous_prices(ACTIVE_CONTINUOUS_PULL)
returns = prices.pct_change(fill_method=None)
print(f"\\n[continuous] pull={ACTIVE_CONTINUOUS_PULL} shape={prices.shape}")
print(f"  window={prices.index.min().date()} → {prices.index.max().date()}")

# --- Layer B: individual contracts inventory (for Day-2 rolls / Bloomberg later) ---
try:
    contracts_manifest = load_contracts_manifest(ACTIVE_CONTRACTS_PULL)
    n_ok = len(contracts_manifest.get("symbols_with_data", []))
    print(f"\\n[contracts] pull={ACTIVE_CONTRACTS_PULL} symbols_with_data={n_ok}")
except FileNotFoundError as exc:
    contracts_manifest = None
    print(f"\\n[contracts] not loaded: {exc}")

PAIR = list(PAIR_COLS)
print(f"\\nSelected pair: {PAIR}")
'''
    )
)

# --- 2. Raw validation ---
cells.append(md("## 2. Raw-data validation"))
cells.append(
    code(
        '''# Validate full continuous panel + Corn/Wheat slice
report_all = validate_price_panel(prices, name="raw_continuous_all", allow_na=True)
print(report_all.summary())
report_all.raise_if_failed()

pair_raw = prices[PAIR].copy()
pair_raw.index = pd.to_datetime(pair_raw.index)
report_pair_raw = validate_price_panel(
    pair_raw, required_cols=PAIR, name="raw_corn_wheat", allow_na=True
)
print(report_pair_raw.summary())
report_pair_raw.raise_if_failed()

print("\\nMissing counts (Corn/Wheat):")
print(pair_raw.isna().sum())
print(f"Rows with any NA: {int(pair_raw.isna().any(axis=1).sum())}")

# Compact EDA plot (single figure block)
pair_ret = pair_raw.pct_change(fill_method=None)
both = pair_raw.dropna()
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)
axes[0].plot(pair_raw.index, pair_raw["Corn"], label="Corn", lw=1)
axes[0].plot(pair_raw.index, pair_raw["Wheat"], label="Wheat", lw=1)
axes[0].set_title("Raw continuous Closes (provisional Yahoo =F)")
axes[0].legend()
for dt in pair_raw.index[pair_raw.isna().any(axis=1)]:
    axes[0].axvline(dt, color="red", alpha=0.3, lw=1)
axes[1].hist(pair_ret["Corn"].dropna(), bins=50, alpha=0.55, label="Corn")
axes[1].hist(pair_ret["Wheat"].dropna(), bins=50, alpha=0.55, label="Wheat")
axes[1].set_title("Daily return distributions (raw)")
axes[1].legend()
plt.tight_layout()
plt.show()
'''
    )
)

# --- 3. Pair screening ---
cells.append(
    md(
        """## 3. Pair screening (exploratory)

Engle-Granger screen on the **raw continuous** panel to motivate Corn/Wheat.
Final inference uses the **cleaned** panel in section 6 (still provisional on Yahoo).
"""
    )
)
cells.append(
    code(
        '''from itertools import combinations
from statsmodels.tsa.stattools import coint

log_raw = np.log(prices.clip(lower=1e-12))
rows = []
for a, b in combinations(log_raw.columns, 2):
    pair = log_raw[[a, b]].dropna()
    if len(pair) < 252:
        continue
    t_stat, p_value, crit = coint(pair[a], pair[b])
    corr = returns[[a, b]].corr().iloc[0, 1]
    rows.append({"Pair": f"{a} / {b}", "P-value": p_value, "T-stat": t_stat, "Corr": corr, "N": len(pair)})

coint_screen = pd.DataFrame(rows).sort_values("P-value").reset_index(drop=True)
print("Exploratory cointegration screen (raw continuous):")
print(coint_screen.round(4).to_string(index=False))
chosen = coint_screen.loc[coint_screen["Pair"].isin(["Corn / Wheat", "Wheat / Corn"])]
print("\\nProject choice: Corn / Wheat")
print(chosen.round(4).to_string(index=False) if len(chosen) else "Corn/Wheat not in screen.")
print(f"DATA_SOURCE={DATA_SOURCE} — exploratory only; see section 6 for clean-panel test.")
'''
    )
)

# --- 4. Day 2 cleaning ---
cells.append(
    md(
        """## 4. Day 2 — Cleaning (rules unchanged)

1. Both Corn and Wheat present, finite, and `> 0`
2. No forward-fill / no interpolation
3. No winsorization of prices
4. Extreme returns flagged, not auto-deleted
5. Write CSV + Parquet under `data/clean/` + `lineage.json` (relative paths)
"""
    )
)
cells.append(
    code(
        '''pair_clean, dropped_detail = clean_corn_wheat(pair_raw)
pair_log = np.log(pair_clean)
extreme_flags = flag_extreme_returns(pair_clean, threshold=0.10)

n_raw, n_clean = len(pair_raw), len(pair_clean)
print("Day 2 cleaning summary")
print(f"  DATA_SOURCE:  {DATA_SOURCE}")
print(f"  Calendar:     {CALENDAR_CONVENTION}")
print(f"  Raw rows:     {n_raw:,}")
print(f"  Clean rows:   {n_clean:,}")
print(f"  Dropped:      {n_raw - n_clean:,} ({(n_raw - n_clean) / n_raw:.2%})")
print("  Forward-fill: NO | Interpolate: NO | Winsorize: NO")
if dropped_detail:
    print("\\nDropped dates:")
    for d in dropped_detail:
        print(f"  {d['date']}: {d['reason']}")
print(f"\\nExtreme |r|>10% flags kept: {len(extreme_flags)}")

CLEAN_ROOT.mkdir(parents=True, exist_ok=True)
panel_csv = CLEAN_ROOT / "corn_wheat_panel.csv"
panel_pq = CLEAN_ROOT / "corn_wheat_panel.parquet"
log_csv = CLEAN_ROOT / "corn_wheat_log_prices_clean.csv"
log_pq = CLEAN_ROOT / "corn_wheat_log_prices_clean.parquet"
alias_csv = CLEAN_ROOT / "corn_wheat_prices_clean.csv"

pair_clean.to_csv(panel_csv)
pair_clean.to_parquet(panel_pq)
pair_log.to_csv(log_csv)
pair_log.to_parquet(log_pq)
pair_clean.to_csv(alias_csv)  # CSV alias only

lineage = {
    "created_at_utc": pd.Timestamp.utcnow().isoformat(),
    "DATA_SOURCE": DATA_SOURCE,
    "calendar_convention": CALENDAR_CONVENTION,
    "calendar_note": CALENDAR_NOTE,
    "pair": PAIR,
    "raw_pull_id": ACTIVE_CONTINUOUS_PULL,
    "raw_input_file": to_rel(RAW_ROOT / ACTIVE_CONTINUOUS_PULL / "commodity_prices.csv"),
    "rules": {
        "require_both_legs_finite_positive": True,
        "forward_fill": False,
        "interpolation": False,
        "winsorize_prices": False,
        "winsorize_rationale": "Pairs spread/cointegration uses tails; N=2 cross-sectional winsorize N/A",
        "drop_extreme_returns": False,
    },
    "n_raw_rows": n_raw,
    "n_clean_rows": n_clean,
    "dropped": dropped_detail,
    "extreme_return_flags_abs_gt_10pct": extreme_flags,
    "outputs": {
        "panel_csv": to_rel(panel_csv),
        "panel_parquet": to_rel(panel_pq),
        "log_prices_csv": to_rel(log_csv),
        "log_prices_parquet": to_rel(log_pq),
        "legacy_clean_alias_csv": to_rel(alias_csv),
    },
    "provisional": True,
    "bloomberg_rerun_required_for_explicit_rolls": True,
}
lineage_path = CLEAN_ROOT / "lineage.json"
lineage_path.write_text(json.dumps(lineage, indent=2))
print("\\nWrote:")
for k, v in lineage["outputs"].items():
    print(f"  {k}: {v}")
print(f"  lineage: {to_rel(lineage_path)}")
'''
    )
)

# --- 5. Clean validation ---
cells.append(md("## 5. Cleaned-data validation"))
cells.append(
    code(
        '''# Reload via clean loader to prove the artifact round-trips
pair_clean_reloaded = load_clean_pair()
pair_log_reloaded = load_clean_log_pair()

report_clean = validate_clean_pair(pair_clean_reloaded)
print(report_clean.summary())
report_clean.raise_if_failed()
assert pair_clean_reloaded.shape[0] == pair_clean.shape[0]

fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
axes[0].plot(pair_clean.index, pair_clean["Corn"], label="Corn", lw=1)
axes[0].plot(pair_clean.index, pair_clean["Wheat"], label="Wheat", lw=1)
axes[0].set_title("Clean panel — Corn & Wheat")
axes[0].legend()

# Illustrative full-sample hedge ratio ONLY for visualization — not for backtest reuse
beta_viz = np.polyfit(pair_log["Wheat"], pair_log["Corn"], 1)[0]
spread_viz = pair_log["Corn"] - beta_viz * pair_log["Wheat"]
axes[1].plot(spread_viz.index, spread_viz, color="tab:green", lw=1)
axes[1].axhline(spread_viz.mean(), color="black", lw=0.8, ls="--")
axes[1].set_title(f"Illustrative log spread (full-sample beta={beta_viz:.3f}) — NOT for backtest")
plt.tight_layout()
plt.show()
print("WARNING: full-sample hedge ratio above is illustrative only; estimate beta in-sample / rolling in any backtest.")
'''
    )
)

# --- 6. Final cointegration ---
cells.append(
    md(
        """## 6. Final cointegration test (cleaned log prices)

Uses official Day-2 `pair_log`. Yahoo continuous remains **provisional**.
"""
    )
)
cells.append(
    code(
        '''from statsmodels.tsa.stattools import coint

# Hedge ratio: OLS Corn on Wheat in logs (report only; do not reuse blindly in backtests)
x = pair_log["Wheat"].to_numpy()
y = pair_log["Corn"].to_numpy()
beta_hat = np.polyfit(x, y, 1)[0]

t_stat, p_value, crit = coint(pair_log["Corn"], pair_log["Wheat"])
corr = pair_clean.pct_change(fill_method=None).corr().iloc[0, 1]
n = len(pair_log)

print("=" * 60)
print("FINAL COINTEGRATION REPORT — Corn / Wheat")
print("=" * 60)
print(f"DATA_SOURCE:          {DATA_SOURCE}  [PROVISIONAL]")
print(f"Calendar convention:  {CALENDAR_CONVENTION}")
print(f"Sample:               {pair_log.index.min().date()} → {pair_log.index.max().date()}  N={n}")
print(f"Hedge ratio (OLS):    {beta_hat:.6f}   (log Corn on log Wheat)")
print(f"EG test statistic:    {t_stat:.4f}")
print(f"EG p-value:           {p_value:.4f}")
print(f"Critical values:      {crit}")
print(f"Return correlation:   {corr:.4f}")
if p_value < 0.05:
    print("Result: evidence of cointegration at 5% on the clean provisional panel.")
else:
    print("Result: no strong cointegration evidence at 5% on this panel.")
print("-" * 60)
print("BACKTEST WARNING: do not reuse this full-sample hedge ratio out of sample.")
print("Re-estimate beta on in-sample / expanding / rolling windows only.")
print("Re-run this cell after Bloomberg replaces Yahoo continuous.")

selected_pair = pair_log.copy()
'''
    )
)

# --- 7. Outputs / limitations ---
cells.append(
    md(
        """## 7. Saved outputs and limitations

**Outputs:** `data/clean/corn_wheat_panel.{csv,parquet}`, `data/clean/corn_wheat_log_prices_clean.{csv,parquet}`, `data/clean/lineage.json`

**Limitations**
- Yahoo `=F` roll method is opaque; treat results as provisional.
- Expired CBT months are mostly unavailable on Yahoo — full Panama rolls need Bloomberg.
- Canonical join calendar is CME ag futures trading dates; align any external data with asof joins.
- Next: Bloomberg raw pull into a new `data/raw/<timestamp>_bloomberg/` folder (never overwrite Yahoo), then re-run cleaning + cointegration.
"""
    )
)
cells.append(
    code(
        '''print("Artifacts:")
for p in sorted(CLEAN_ROOT.glob("corn_wheat*")):
    print(f"  {to_rel(p)}")
print(f"  {to_rel(CLEAN_ROOT / 'lineage.json')}")
print(f"\\nDATA_SOURCE={DATA_SOURCE} | calendar={CALENDAR_CONVENTION}")
print("Raw remains immutable under data/raw/. Cleaning never writes into data/raw/.")
'''
    )
)

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "cells": cells,
}
nb_path.write_text(json.dumps(nb, indent=2))
print(f"Wrote {nb_path} with {len(cells)} cells")
