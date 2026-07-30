"""Shared loaders, validators, and path helpers for the Corn/Wheat pairs project.

Import from the notebook (run with cwd = project root) or from scripts/.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

CALENDAR_CONVENTION = "CME_ag_futures_trading_date"
CALENDAR_NOTE = (
    "Canonical dates are CME agricultural futures trading/settlement dates. "
    "Any future external series (weather, macro, Bloomberg fields) must be "
    "point-in-time asof-aligned to this trading-date calendar before joining."
)

DEFAULT_CONTINUOUS_PULL = "2026-07-29T152437Z_continuous"
DEFAULT_CONTRACTS_PULL = "2026-07-29T203057Z_contracts"
DEFAULT_BLOOMBERG_PULL = "2026-07-29T234058Z_bloomberg"
PAIR_COLS = ("Corn", "Wheat")

# Bloomberg sheet name -> project display name
BLOOMBERG_SHEET_MAP = {
    "corn": "Corn",
    "wheat": "Wheat",
    "soya": "Soybeans",
    "oat": "Oats",
    "rice": "Rice",
    "canola": "Canola",
    "coffee": "Coffee",
    "cocoa": "Cocoa",
    "sugar": "Sugar",
    "cotton": "Cotton",
    "oil": "Crude Oil",
    "gas": "Natural Gas",
    "gold": "Gold",
    "silver": "Silver",
    "copper": "Copper",
}


def get_project_root() -> Path:
    """Resolve project root from this file or cwd."""
    here = Path(__file__).resolve().parent
    if (here / "data" / "raw").is_dir() and (here / "docs").is_dir():
        return here
    cwd = Path.cwd().resolve()
    if (cwd / "data" / "raw").is_dir():
        return cwd
    raise FileNotFoundError(
        "Cannot find project root (expected data/raw and docs/). "
        "Open/run from the 'MMF Workshop in Finance' folder."
    )


PROJECT_ROOT = get_project_root()
RAW_ROOT = PROJECT_ROOT / "data" / "raw"
CLEAN_ROOT = PROJECT_ROOT / "data" / "clean"


def to_rel(path: Path | str) -> str:
    """Return path relative to project root (posix)."""
    p = Path(path).resolve()
    try:
        return p.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def ensure_dirs() -> None:
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    CLEAN_ROOT.mkdir(parents=True, exist_ok=True)


def new_raw_pull_dir(kind: str) -> Path:
    """Create a unique immutable raw pull folder (never overwrite)."""
    from datetime import datetime, timezone

    ensure_dirs()
    pull_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M%SZ") + f"_{kind}"
    path = RAW_ROOT / pull_id
    path.mkdir(parents=False, exist_ok=False)
    return path


# ---------------------------------------------------------------------------
# Loaders (distinct layers)
# ---------------------------------------------------------------------------

def load_continuous_prices(pull_id: str = DEFAULT_CONTINUOUS_PULL) -> pd.DataFrame:
    """Load immutable Yahoo continuous Close panel from data/raw/<pull_id>/."""
    path = RAW_ROOT / pull_id / "commodity_prices.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing continuous prices at {to_rel(path)}. "
            f"Available raw pulls: {[p.name for p in RAW_ROOT.glob('*') if p.is_dir()]}"
        )
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def load_contracts_long(pull_id: str = DEFAULT_CONTRACTS_PULL) -> pd.DataFrame:
    """Load long-format individual CBT contract history (if present)."""
    path = RAW_ROOT / pull_id / "contracts_long.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing contracts long file at {to_rel(path)}. "
            "Run scripts/pull_raw_contracts.py to create a new immutable pull."
        )
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def load_contracts_manifest(pull_id: str = DEFAULT_CONTRACTS_PULL) -> dict:
    path = RAW_ROOT / pull_id / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing contracts manifest at {to_rel(path)}")
    return json.loads(path.read_text())


def _coerce_sheet_dates(series: pd.Series) -> pd.Series:
    """Parse sheet dates that may already be timestamps or Excel serials."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")

    # Mixed / object: try datetime first, then Excel serial for remaining
    as_dt = pd.to_datetime(series, errors="coerce")
    nums = pd.to_numeric(series, errors="coerce")
    # Excel serial ~18264 = 1950-01-01; ~73415 = 2100-12-31
    serial_mask = as_dt.isna() & nums.notna() & (nums >= 18264) & (nums <= 73415)
    if serial_mask.any():
        as_dt = as_dt.copy()
        as_dt.loc[serial_mask] = pd.to_datetime(
            nums.loc[serial_mask], unit="D", origin="1899-12-30", errors="coerce"
        )
    return as_dt


def _sheet_close_series(df: pd.DataFrame, sheet_key: str) -> pd.Series:
    """Extract a Close-like series from a Bloomberg sheet DataFrame."""
    cols = {str(c).strip(): c for c in df.columns}
    # Prefer explicit Close; Corn export uses Last Price
    if "Close" in cols:
        price_col = cols["Close"]
    elif "Last Price" in cols:
        price_col = cols["Last Price"]
    elif "PX_LAST" in cols:
        price_col = cols["PX_LAST"]
    else:
        raise KeyError(
            f"Sheet {sheet_key!r}: no Close/Last Price column; got {list(df.columns)}"
        )

    if "Date" not in cols:
        date_col = df.columns[0]
    else:
        date_col = cols["Date"]

    out = pd.DataFrame(
        {
            "date": _coerce_sheet_dates(df[date_col]),
            "close": pd.to_numeric(df[price_col], errors="coerce"),
        }
    ).dropna(subset=["date"])
    out = out.set_index("date").sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out["close"]
def load_bloomberg_workbook(
    pull_id: str = DEFAULT_BLOOMBERG_PULL,
    filename: str = "data_trading.xlsx",
) -> pd.DataFrame:
    """Load Bloomberg multi-sheet workbook into a Close panel (columns = commodities)."""
    path = RAW_ROOT / pull_id / filename
    if not path.exists():
        # fallback: loose file under data/raw/
        alt = RAW_ROOT / filename
        if alt.exists():
            path = alt
        else:
            raise FileNotFoundError(
                f"Missing Bloomberg workbook at {to_rel(RAW_ROOT / pull_id / filename)}"
            )

    xl = pd.ExcelFile(path, engine="openpyxl")
    series_map: dict[str, pd.Series] = {}
    for sheet in xl.sheet_names:
        key = sheet.strip().lower()
        if key not in BLOOMBERG_SHEET_MAP:
            continue
        name = BLOOMBERG_SHEET_MAP[key]
        raw = xl.parse(sheet)
        if raw.empty:
            continue
        series_map[name] = _sheet_close_series(raw, key)

    if not series_map:
        raise ValueError(f"No recognizable commodity sheets in {to_rel(path)}")

    panel = pd.DataFrame(series_map).sort_index()
    panel.index = pd.to_datetime(panel.index)
    return panel


def load_bloomberg_manifest(pull_id: str = DEFAULT_BLOOMBERG_PULL) -> dict:
    path = RAW_ROOT / pull_id / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing Bloomberg manifest at {to_rel(path)}")
    return json.loads(path.read_text())


def load_clean_pair(
    panel_name: str = "corn_wheat_panel.csv",
) -> pd.DataFrame:
    """Load cleaned Corn/Wheat price panel from data/clean/."""
    path = CLEAN_ROOT / panel_name
    if not path.exists():
        alt = CLEAN_ROOT / "corn_wheat_panel.parquet"
        if alt.exists():
            df = pd.read_parquet(alt)
            df.index = pd.to_datetime(df.index)
            return df.sort_index()
        raise FileNotFoundError(
            f"Missing clean panel at {to_rel(path)}. Run the Day 2 cleaning notebook section first."
        )
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def load_clean_log_pair(
    name: str = "corn_wheat_log_prices_clean.csv",
) -> pd.DataFrame:
    path = CLEAN_ROOT / name
    if not path.exists():
        alt = CLEAN_ROOT / "corn_wheat_log_prices_clean.parquet"
        if alt.exists():
            df = pd.read_parquet(alt)
            df.index = pd.to_datetime(df.index)
            return df.sort_index()
        raise FileNotFoundError(f"Missing clean log panel at {to_rel(path)}")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

class ValidationReport:
    def __init__(self, name: str):
        self.name = name
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def error(self, msg: str) -> None:
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def raise_if_failed(self) -> None:
        if self.errors:
            joined = "; ".join(self.errors)
            raise ValueError(f"[{self.name}] validation failed: {joined}")

    def summary(self) -> str:
        parts = [f"[{self.name}] {'PASS' if self.ok else 'FAIL'}"]
        if self.errors:
            parts.append(f"  errors ({len(self.errors)}):")
            parts.extend(f"    - {e}" for e in self.errors)
        if self.warnings:
            parts.append(f"  warnings ({len(self.warnings)}):")
            parts.extend(f"    - {w}" for w in self.warnings)
        if self.ok and not self.warnings:
            parts.append("  no issues")
        return "\n".join(parts)


def _check_datetime_index(df: pd.DataFrame, report: ValidationReport) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        report.error("index is not DatetimeIndex")
        return
    if df.index.hasnans:
        report.error("index contains NaT")
    if not df.index.is_unique:
        report.error(f"duplicate dates: {int(df.index.duplicated().sum())}")
    if not df.index.is_monotonic_increasing:
        report.error("dates are not sorted ascending")


def _check_numeric_cols(df: pd.DataFrame, cols: Sequence[str], report: ValidationReport) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        report.error(f"missing columns: {missing}")
        return
    for c in cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            coerced = pd.to_numeric(df[c], errors="coerce")
            if coerced.isna().sum() > df[c].isna().sum():
                report.error(f"column {c} has non-numeric values")
            else:
                report.warn(f"column {c} is not numeric dtype ({df[c].dtype})")


def _check_gaps(df: pd.DataFrame, report: ValidationReport, max_gap_days: int = 5) -> None:
    if len(df) < 2:
        return
    gaps = df.index.to_series().diff().dt.days.dropna()
    large = gaps[gaps > max_gap_days]
    if len(large):
        top = large.sort_values(ascending=False).head(5)
        detail = ", ".join(f"{idx.date()} (+{int(v)}d)" for idx, v in top.items())
        report.warn(f"{len(large)} calendar gaps > {max_gap_days}d (e.g. {detail})")


def validate_price_panel(
    df: pd.DataFrame,
    required_cols: Iterable[str] | None = None,
    name: str = "continuous_prices",
    allow_na: bool = True,
) -> ValidationReport:
    """Validate a multi-asset or pair Close panel indexed by date."""
    report = ValidationReport(name)
    if df is None or len(df) == 0:
        report.error("panel is empty")
        return report

    cols = list(required_cols) if required_cols is not None else list(df.columns)
    _check_datetime_index(df, report)
    _check_numeric_cols(df, cols, report)

    # Exact duplicate (date + values). Same prices on different dates are fine.
    if df.reset_index().duplicated().any():
        report.error(
            f"duplicated (date, values) rows: {int(df.reset_index().duplicated().sum())}"
        )

    present = [c for c in cols if c in df.columns]
    if present:
        vals = df[present]
        if not allow_na and vals.isna().any().any():
            report.error(f"NaNs present: {vals.isna().sum().to_dict()}")
        nonpos = (vals <= 0) | ~np.isfinite(vals)
        bad = nonpos & vals.notna()
        if bad.any().any():
            msg = f"non-positive or non-finite values: {bad.sum().to_dict()}"
            # Raw multi-asset panels: warn (pair cleaning will drop bad legs).
            # Clean panels (allow_na=False): hard fail.
            if allow_na:
                report.warn(msg)
            else:
                report.error(msg)

    _check_gaps(df, report)
    return report


def validate_clean_pair(
    df: pd.DataFrame,
    cols: Sequence[str] = PAIR_COLS,
    name: str = "clean_corn_wheat",
) -> ValidationReport:
    """Strict validation for the official cleaned Corn/Wheat panel."""
    report = validate_price_panel(df, required_cols=cols, name=name, allow_na=False)
    if len(df) == 0:
        report.error("cleaned dataset is empty")
    # re-emphasize no NaNs on clean pair
    present = [c for c in cols if c in df.columns]
    if present and df[present].isna().any().any():
        report.error("cleaned pair still contains NaNs")
    return report


def validate_contract_ohlc(df: pd.DataFrame, name: str = "contract") -> ValidationReport:
    """Validate a single-contract OHLC download."""
    report = ValidationReport(name)
    if df is None or len(df) == 0:
        report.error("empty download")
        return report
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    if "Close" not in df.columns:
        report.error(f"missing Close column; got {list(df.columns)}")
        return report
    close = pd.to_numeric(df["Close"], errors="coerce")
    if close.notna().sum() == 0:
        report.error("Close column has no numeric values")
    if (close.dropna() <= 0).any():
        report.warn("Close contains non-positive values")
    return report


def flag_extreme_returns(
    prices: pd.DataFrame,
    threshold: float = 0.10,
) -> list[dict]:
    """Flag |daily return| > threshold without dropping rows."""
    rets = prices.pct_change(fill_method=None)
    flags: list[dict] = []
    for col in prices.columns:
        r = rets[col].dropna()
        for dt, val in r[r.abs() > threshold].items():
            flags.append(
                {
                    "date": str(pd.Timestamp(dt).date()),
                    "series": str(col),
                    "return": float(val),
                }
            )
    return flags


def clean_corn_wheat(pair_raw: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Apply Day-2 cleaning rules (unchanged): both legs finite & positive; no fill."""
    cols = list(PAIR_COLS)
    missing = [c for c in cols if c not in pair_raw.columns]
    if missing:
        raise KeyError(f"pair raw missing columns {missing}")
    raw = pair_raw[cols].copy()
    raw.index = pd.to_datetime(raw.index)
    valid = raw.notna().all(axis=1) & np.isfinite(raw).all(axis=1) & (raw > 0).all(axis=1)
    clean = raw.loc[valid].sort_index()
    dropped: list[dict] = []
    for dt in raw.index.difference(clean.index):
        row = raw.loc[dt]
        reasons = []
        for c in cols:
            val = row[c]
            if pd.isna(val):
                reasons.append(f"{c}=NaN")
            elif not np.isfinite(val):
                reasons.append(f"{c}=non-finite")
            elif val <= 0:
                reasons.append(f"{c}<={val}")
        dropped.append({"date": str(pd.Timestamp(dt).date()), "reason": ", ".join(reasons)})
    return clean, dropped
