"""Create a NEW immutable raw Corn/Wheat CBT contracts pull (never overwrites).

Usage (from project root, with yfinance installed):
  python scripts/pull_raw_contracts.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline_utils import RAW_ROOT, to_rel, validate_contract_ohlc  # noqa: E402

MONTHS = list("HKNUZ")
ROOTS = {"Corn": "ZC", "Wheat": "ZW"}
# Near-dated years only; Yahoo empties most expired months
YEARS = list(range(24, 29))


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    return out


def main() -> None:
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    pull_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ") + "_contracts"
    out = RAW_ROOT / pull_id
    contracts_dir = out / "contracts"
    # Fail if this pull id somehow exists — never overwrite prior raw pulls
    contracts_dir.mkdir(parents=True, exist_ok=False)

    manifest = {
        "pull_id": pull_id,
        "pulled_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Yahoo Finance via yfinance",
        "endpoint": "yf.download(symbol, period='max', auto_adjust=False)",
        "calendar_convention": "CME_ag_futures_trading_date",
        "symbols_attempted": [],
        "symbols_with_data": [],
        "symbols_empty_or_missing": [],
        "symbols_failed": [],
        "known_gaps": [
            "Yahoo drops most expired CBT contract histories; this pull only captures symbols with data at pull time.",
            "Not sufficient alone for a full multi-year Panama continuous series.",
        ],
    }

    rows: list[pd.DataFrame] = []
    for crop, root in ROOTS.items():
        for y in YEARS:
            for m in MONTHS:
                sym = f"{root}{m}{y:02d}.CBT"
                manifest["symbols_attempted"].append(sym)
                try:
                    d = yf.download(
                        sym,
                        period="max",
                        progress=False,
                        auto_adjust=False,
                        threads=False,
                    )
                except Exception as exc:  # network / API errors
                    manifest["symbols_failed"].append(
                        {"symbol": sym, "error": f"{type(exc).__name__}: {exc}"}
                    )
                    print(f"FAIL {sym}: {exc}")
                    time.sleep(0.1)
                    continue

                if d is None or len(d) == 0:
                    manifest["symbols_empty_or_missing"].append(
                        {"symbol": sym, "reason": "empty"}
                    )
                    time.sleep(0.1)
                    continue

                d = _flatten_columns(d)
                report = validate_contract_ohlc(d, name=sym)
                if not report.ok:
                    manifest["symbols_failed"].append(
                        {"symbol": sym, "error": "; ".join(report.errors)}
                    )
                    print(f"INVALID {sym}: {report.errors}")
                    time.sleep(0.1)
                    continue

                file_path = contracts_dir / f"{sym}.csv"
                d.to_csv(file_path)
                meta = {
                    "symbol": sym,
                    "crop": crop,
                    "n_rows": int(len(d)),
                    "start": str(d.index.min().date()),
                    "end": str(d.index.max().date()),
                    "file": to_rel(file_path),
                    "validation_warnings": report.warnings,
                }
                manifest["symbols_with_data"].append(meta)

                tmp = d.reset_index()
                tmp = tmp.rename(columns={tmp.columns[0]: "date"})
                tmp["symbol"] = sym
                tmp["crop"] = crop
                rows.append(tmp)
                print(f"saved {sym} n={len(d)}")
                time.sleep(0.1)

    if rows:
        long_df = pd.concat(rows, ignore_index=True)
        long_path = out / "contracts_long.csv"
        long_df.to_csv(long_path, index=False)
        manifest["contracts_long"] = to_rel(long_path)
        print(f"Wrote long panel: {to_rel(long_path)} rows={len(long_df)}")
    else:
        manifest["contracts_long"] = None
        print("No contract rows downloaded; contracts_long.csv not written.")

    manifest_path = out / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(
        f"Wrote {to_rel(out)} | ok={len(manifest['symbols_with_data'])} "
        f"empty={len(manifest['symbols_empty_or_missing'])} "
        f"failed={len(manifest['symbols_failed'])}"
    )


if __name__ == "__main__":
    main()
