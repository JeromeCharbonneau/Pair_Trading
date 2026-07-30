#!/usr/bin/env python3
"""Ingest Bloomberg workbook, rebuild Corn/Wheat clean panel, compare to Yahoo."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline_utils import (  # noqa: E402
    CALENDAR_CONVENTION,
    CALENDAR_NOTE,
    CLEAN_ROOT,
    DEFAULT_BLOOMBERG_PULL,
    DEFAULT_CONTINUOUS_PULL,
    PAIR_COLS,
    clean_corn_wheat,
    flag_extreme_returns,
    load_bloomberg_workbook,
    load_continuous_prices,
    to_rel,
    validate_clean_pair,
    validate_price_panel,
)


def pair_stats(prices: pd.DataFrame, label: str) -> dict:
    cols = list(PAIR_COLS)
    raw = prices[cols].copy()
    clean, dropped = clean_corn_wheat(raw)
    validate_clean_pair(clean).raise_if_failed()
    logs = np.log(clean)
    x = logs["Wheat"].to_numpy()
    y = logs["Corn"].to_numpy()
    beta = float(np.polyfit(x, y, 1)[0])
    t_stat, p_value, crit = coint(logs["Corn"], logs["Wheat"])
    corr = float(clean.pct_change(fill_method=None).corr().iloc[0, 1])
    extremes = flag_extreme_returns(clean, 0.10)
    return {
        "label": label,
        "n_raw": int(len(raw)),
        "n_clean": int(len(clean)),
        "start": str(clean.index.min().date()),
        "end": str(clean.index.max().date()),
        "dropped": dropped,
        "n_dropped": len(dropped),
        "corr": corr,
        "beta": beta,
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "crit": [float(c) for c in crit],
        "n_extreme_flags": len(extremes),
        "cointegrated_5pct": bool(p_value < 0.05),
        "clean": clean,
        "logs": logs,
        "extremes": extremes,
    }


def screen_pairs(prices: pd.DataFrame, min_n: int = 252) -> pd.DataFrame:
    logs = np.log(prices.where(prices > 0))
    rets = prices.pct_change(fill_method=None)
    rows = []
    for a, b in combinations(prices.columns, 2):
        pair = logs[[a, b]].dropna()
        if len(pair) < min_n:
            continue
        t_stat, p_value, _ = coint(pair[a], pair[b])
        corr = rets[[a, b]].corr().iloc[0, 1]
        rows.append(
            {
                "Pair": f"{a} / {b}",
                "P-value": p_value,
                "T-stat": t_stat,
                "Corr": corr,
                "N": len(pair),
            }
        )
    return pd.DataFrame(rows).sort_values("P-value").reset_index(drop=True)


def save_bloomberg_clean(stats: dict, pull_id: str) -> dict:
    CLEAN_ROOT.mkdir(parents=True, exist_ok=True)
    clean = stats["clean"]
    logs = stats["logs"]
    panel_csv = CLEAN_ROOT / "corn_wheat_panel_bloomberg.csv"
    panel_pq = CLEAN_ROOT / "corn_wheat_panel_bloomberg.parquet"
    log_csv = CLEAN_ROOT / "corn_wheat_log_prices_bloomberg.csv"
    log_pq = CLEAN_ROOT / "corn_wheat_log_prices_bloomberg.parquet"
    clean.to_csv(panel_csv)
    clean.to_parquet(panel_pq)
    logs.to_csv(log_csv)
    logs.to_parquet(log_pq)

    lineage = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "DATA_SOURCE": "bloomberg",
        "calendar_convention": CALENDAR_CONVENTION,
        "calendar_note": CALENDAR_NOTE,
        "pair": list(PAIR_COLS),
        "raw_pull_id": pull_id,
        "raw_input_file": to_rel(
            ROOT / "data" / "raw" / pull_id / "data_trading.xlsx"
        ),
        "rules": {
            "require_both_legs_finite_positive": True,
            "forward_fill": False,
            "interpolation": False,
            "winsorize_prices": False,
            "drop_extreme_returns": False,
        },
        "n_raw_rows": stats["n_raw"],
        "n_clean_rows": stats["n_clean"],
        "dropped": stats["dropped"],
        "extreme_return_flags_abs_gt_10pct": stats["extremes"],
        "outputs": {
            "panel_csv": to_rel(panel_csv),
            "panel_parquet": to_rel(panel_pq),
            "log_prices_csv": to_rel(log_csv),
            "log_prices_parquet": to_rel(log_pq),
        },
        "cointegration": {
            "hedge_ratio_log_corn_on_wheat": stats["beta"],
            "t_stat": stats["t_stat"],
            "p_value": stats["p_value"],
            "critical_values": stats["crit"],
            "return_corr": stats["corr"],
        },
        "provisional_note": "Bloomberg continuous sheets; roll method still vendor-defined unless rebuilt from contracts.",
    }
    lineage_path = CLEAN_ROOT / "lineage_bloomberg.json"
    lineage_path.write_text(json.dumps(lineage, indent=2))
    return lineage


def main() -> None:
    print("Loading Bloomberg workbook...")
    bb = load_bloomberg_workbook(DEFAULT_BLOOMBERG_PULL)
    print(f"  shape={bb.shape} cols={list(bb.columns)}")
    print(f"  window={bb.index.min().date()} → {bb.index.max().date()}")
    rep = validate_price_panel(bb, name="bloomberg_all", allow_na=True)
    print(rep.summary())

    print("\nLoading Yahoo continuous...")
    yahoo = load_continuous_prices(DEFAULT_CONTINUOUS_PULL)

    bb_stats = pair_stats(bb, "Bloomberg")
    y_stats = pair_stats(yahoo, "Yahoo")

    lineage = save_bloomberg_clean(bb_stats, DEFAULT_BLOOMBERG_PULL)
    print(f"\nWrote Bloomberg clean artifacts + {to_rel(CLEAN_ROOT / 'lineage_bloomberg.json')}")

    print("\nScreening Bloomberg pairs (exploratory)...")
    bb_screen = screen_pairs(bb)
    print(bb_screen.head(15).round(4).to_string(index=False))

    print("\nScreening Yahoo pairs (exploratory)...")
    y_screen = screen_pairs(yahoo)
    print(y_screen.head(15).round(4).to_string(index=False))

    # Overlap window comparison (fairer)
    common_idx = bb_stats["clean"].index.intersection(y_stats["clean"].index)
    bb_overlap = bb_stats["clean"].loc[common_idx]
    y_overlap = y_stats["clean"].loc[common_idx]
    # recompute EG on overlap
    def eg(clean: pd.DataFrame) -> tuple[float, float, float, float]:
        logs = np.log(clean)
        beta = float(np.polyfit(logs["Wheat"], logs["Corn"], 1)[0])
        t, p, _ = coint(logs["Corn"], logs["Wheat"])
        corr = float(clean.pct_change(fill_method=None).corr().iloc[0, 1])
        return beta, float(t), float(p), corr

    bb_o = eg(bb_overlap)
    y_o = eg(y_overlap)

    def rank_of(screen: pd.DataFrame, pair: str = "Corn / Wheat") -> int | None:
        hits = screen.index[screen["Pair"].isin([pair, "Wheat / Corn"])].tolist()
        return int(hits[0]) + 1 if hits else None

    comparison = {
        "bloomberg_pull_id": DEFAULT_BLOOMBERG_PULL,
        "yahoo_pull_id": DEFAULT_CONTINUOUS_PULL,
        "full_sample": {
            "bloomberg": {k: bb_stats[k] for k in [
                "n_raw", "n_clean", "start", "end", "n_dropped", "corr", "beta",
                "t_stat", "p_value", "crit", "n_extreme_flags", "cointegrated_5pct",
            ]},
            "yahoo": {k: y_stats[k] for k in [
                "n_raw", "n_clean", "start", "end", "n_dropped", "corr", "beta",
                "t_stat", "p_value", "crit", "n_extreme_flags", "cointegrated_5pct",
            ]},
        },
        "overlap_sample": {
            "n": int(len(common_idx)),
            "start": str(common_idx.min().date()) if len(common_idx) else None,
            "end": str(common_idx.max().date()) if len(common_idx) else None,
            "bloomberg": {
                "beta": bb_o[0], "t_stat": bb_o[1], "p_value": bb_o[2], "corr": bb_o[3],
                "cointegrated_5pct": bb_o[2] < 0.05,
            },
            "yahoo": {
                "beta": y_o[0], "t_stat": y_o[1], "p_value": y_o[2], "corr": y_o[3],
                "cointegrated_5pct": y_o[2] < 0.05,
            },
        },
        "screen_rank_corn_wheat": {
            "bloomberg": rank_of(bb_screen),
            "yahoo": rank_of(y_screen),
        },
        "bloomberg_top5": bb_screen.head(5).round(4).to_dict(orient="records"),
        "yahoo_top5": y_screen.head(5).round(4).to_dict(orient="records"),
    }

    out_json = CLEAN_ROOT / "yahoo_vs_bloomberg_comparison.json"
    out_json.write_text(json.dumps(comparison, indent=2))

    md = f"""# Yahoo vs Bloomberg — Corn / Wheat comparison

Generated: {datetime.now(timezone.utc).isoformat()}

## Full-sample clean Corn/Wheat

| Metric | Yahoo | Bloomberg |
|---|---:|---:|
| Window | {y_stats['start']} → {y_stats['end']} | {bb_stats['start']} → {bb_stats['end']} |
| Clean N | {y_stats['n_clean']:,} | {bb_stats['n_clean']:,} |
| Dropped rows | {y_stats['n_dropped']} | {bb_stats['n_dropped']} |
| Return corr | {y_stats['corr']:.4f} | {bb_stats['corr']:.4f} |
| Hedge ratio (log Corn on Wheat) | {y_stats['beta']:.4f} | {bb_stats['beta']:.4f} |
| EG t-stat | {y_stats['t_stat']:.4f} | {bb_stats['t_stat']:.4f} |
| EG p-value | {y_stats['p_value']:.4f} | {bb_stats['p_value']:.4f} |
| Cointegrated at 5%? | {y_stats['cointegrated_5pct']} | {bb_stats['cointegrated_5pct']} |
| Extreme \\|r\\|>10% flags | {y_stats['n_extreme_flags']} | {bb_stats['n_extreme_flags']} |

## Overlap window (fairer head-to-head)

Common clean dates: **{len(common_idx):,}** ({common_idx.min().date() if len(common_idx) else 'n/a'} → {common_idx.max().date() if len(common_idx) else 'n/a'})

| Metric | Yahoo | Bloomberg |
|---|---:|---:|
| Return corr | {y_o[3]:.4f} | {bb_o[3]:.4f} |
| Hedge ratio | {y_o[0]:.4f} | {bb_o[0]:.4f} |
| EG t-stat | {y_o[1]:.4f} | {bb_o[1]:.4f} |
| EG p-value | {y_o[2]:.4f} | {bb_o[2]:.4f} |
| Cointegrated at 5%? | {y_o[2] < 0.05} | {bb_o[2] < 0.05} |

## Exploratory screen rank of Corn / Wheat

- Yahoo rank: **{rank_of(y_screen)}**
- Bloomberg rank: **{rank_of(bb_screen)}**

### Bloomberg top 5 pairs
{bb_screen.head(5).round(4).to_string(index=False)}

### Yahoo top 5 pairs
{y_screen.head(5).round(4).to_string(index=False)}

## Interpretation notes
- Bloomberg history is much longer (~2000+); full-sample EG results are **not** directly comparable to Yahoo’s ~10y window without the overlap table.
- Cleaning rules identical on both sources.
- Bloomberg sheets are still continuous/vendor series; roll methodology remains a disclosed limitation until true per-contract rolls are built.
"""
    out_md = ROOT / "docs" / "YAHOO_VS_BLOOMBERG.md"
    out_md.write_text(md)
    print("\n" + md)
    print(f"\nWrote {to_rel(out_md)} and {to_rel(out_json)}")


if __name__ == "__main__":
    main()
