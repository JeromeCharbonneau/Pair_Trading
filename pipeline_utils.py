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
FEATURES_ROOT = PROJECT_ROOT / "data" / "features"

# Day-3 feature windows (trading days)
RATIO_ZSCORE_WINDOW = 50  # van Unen (2023) §4.2
ROLLING_HEDGE_WINDOW = 60
REALIZED_VOL_WINDOW = 20
SPREAD_ZSCORE_WINDOW = 60

FEATURE_COLUMNS = (
    "ratio",
    "ratio_zscore_50",
    "log_spread",
    "spread_zscore_60",
    "rv_corn_20",
    "rv_wheat_20",
    "ret_ratio_1d",
)


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
    FEATURES_ROOT.mkdir(parents=True, exist_ok=True)


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


def _load_clean_table(path: Path) -> pd.DataFrame:
    """Load a clean CSV or Parquet table with a DatetimeIndex."""
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def load_clean_pair(
    panel_name: str = "corn_wheat_panel.csv",
) -> pd.DataFrame:
    """Load cleaned Corn/Wheat price panel from data/clean/."""
    path = CLEAN_ROOT / panel_name
    if path.exists():
        return _load_clean_table(path)
    # Fallbacks for stem without extension / alternate format
    stem = Path(panel_name).stem
    for candidate in (
        CLEAN_ROOT / f"{stem}.parquet",
        CLEAN_ROOT / f"{stem}.csv",
        CLEAN_ROOT / "corn_wheat_panel.parquet",
        CLEAN_ROOT / "corn_wheat_panel.csv",
    ):
        if candidate.exists():
            return _load_clean_table(candidate)
    raise FileNotFoundError(
        f"Missing clean panel at {to_rel(path)}. Run the Day 2 cleaning notebook section first."
    )


def load_clean_log_pair(
    name: str = "corn_wheat_log_prices_clean.csv",
) -> pd.DataFrame:
    path = CLEAN_ROOT / name
    if path.exists():
        return _load_clean_table(path)
    stem = Path(name).stem
    for candidate in (
        CLEAN_ROOT / f"{stem}.parquet",
        CLEAN_ROOT / f"{stem}.csv",
        CLEAN_ROOT / "corn_wheat_log_prices_clean.parquet",
        CLEAN_ROOT / "corn_wheat_log_prices_clean.csv",
    ):
        if candidate.exists():
            return _load_clean_table(candidate)
    raise FileNotFoundError(f"Missing clean log panel at {to_rel(path)}")


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


# ---------------------------------------------------------------------------
# Day 3 — Feature engineering
# ---------------------------------------------------------------------------

FEATURE_MANIFEST_ENTRIES: list[dict] = [
    {
        "name": "ratio_zscore_50",
        "category": "Internal / Endogenous",
        "type": "Statistical mean-reversion feature",
        "risk_model_bucket": "Statistical",
        "source": "van Unen (2023)",
        "construction": (
            "ratio = Corn/Wheat; "
            "ratio_zscore_50 = (ratio - rolling_mean_50) / rolling_std_50; "
            "trailing window, never center=True"
        ),
        "paper_derived": True,
    },
    {
        "name": "ratio",
        "category": "Internal / Endogenous",
        "type": "Price-ratio level",
        "risk_model_bucket": "Fundamental-analog",
        "source": "project (van Unen §4.2 ratio definition)",
        "construction": "Corn_Price / Wheat_Price",
        "paper_derived": False,
    },
    {
        "name": "log_spread",
        "category": "Internal / Endogenous",
        "type": "Cointegration residual proxy",
        "risk_model_bucket": "Statistical",
        "source": "project (rolling hedge; no full-sample beta)",
        "construction": (
            "log(Corn) - beta_roll_60 * log(Wheat); "
            "beta from trailing OLS of log Corn on log Wheat"
        ),
        "paper_derived": False,
    },
    {
        "name": "spread_zscore_60",
        "category": "Internal / Endogenous",
        "type": "Statistical mean-reversion feature",
        "risk_model_bucket": "Statistical",
        "source": "project",
        "construction": "(log_spread - rolling_mean_60) / rolling_std_60",
        "paper_derived": False,
    },
    {
        "name": "rv_corn_20",
        "category": "Internal / Endogenous",
        "type": "Realized volatility",
        "risk_model_bucket": "Statistical",
        "source": "project (Day 3 ag-futures table analog)",
        "construction": "std of Corn daily returns over trailing 20 trading days",
        "paper_derived": False,
    },
    {
        "name": "rv_wheat_20",
        "category": "Internal / Endogenous",
        "type": "Realized volatility",
        "risk_model_bucket": "Statistical",
        "source": "project (Day 3 ag-futures table analog)",
        "construction": "std of Wheat daily returns over trailing 20 trading days",
        "paper_derived": False,
    },
    {
        "name": "ret_ratio_1d",
        "category": "Internal / Endogenous",
        "type": "Short-horizon momentum of ratio",
        "risk_model_bucket": "Statistical",
        "source": "project",
        "construction": "ratio.pct_change(1) — known at close of t (no shift needed for feature)",
        "paper_derived": False,
    },
]


def _rolling_ols_beta(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
    """Trailing OLS slope of y on x (with intercept), no look-ahead."""
    yv = y.to_numpy(dtype=float)
    xv = x.to_numpy(dtype=float)
    n = len(yv)
    out = np.full(n, np.nan, dtype=float)
    for i in range(window - 1, n):
        yy = yv[i - window + 1 : i + 1]
        xx = xv[i - window + 1 : i + 1]
        if not (np.isfinite(yy).all() and np.isfinite(xx).all()):
            continue
        x_mean = xx.mean()
        y_mean = yy.mean()
        var_x = ((xx - x_mean) ** 2).sum()
        if var_x <= 0:
            continue
        out[i] = ((xx - x_mean) * (yy - y_mean)).sum() / var_x
    return pd.Series(out, index=y.index, name="beta_roll")


def build_ratio_zscore_50(ratio: pd.Series, window: int = RATIO_ZSCORE_WINDOW) -> pd.Series:
    """Rolling z-score of the Corn/Wheat price ratio.

    Reproduced from van Unen (2023), *Pairs Trading in Agricultural Commodity
    Futures Markets*, §4.2: 50-trading-day rolling mean and standard deviation
    of the price ratio. Stored as a continuous numeric feature (not the paper's
    ±1.5σ binary entry signal).

    Uses a trailing window only (never center=True) so each date uses only
    information available through that date.
    """
    rolling_mean = ratio.rolling(window, min_periods=window).mean()
    rolling_std = ratio.rolling(window, min_periods=window).std()
    z = (ratio - rolling_mean) / rolling_std
    z = z.replace([np.inf, -np.inf], np.nan)
    return z.rename("ratio_zscore_50")


def build_corn_wheat_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Build the Day-3 feature matrix from a cleaned Corn/Wheat price panel.

    Deterministic: same clean input always yields identical feature values.
    No forward-fill; NaNs from warm-up windows are left as NaN.
    """
    missing = [c for c in PAIR_COLS if c not in panel.columns]
    if missing:
        raise KeyError(f"panel missing columns {missing}")

    prices = panel.loc[:, list(PAIR_COLS)].copy()
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    if prices.index.has_duplicates:
        raise ValueError("feature panel index has duplicate dates")

    corn = prices["Corn"].astype(float)
    wheat = prices["Wheat"].astype(float)

    # Price ratio (van Unen §4.2 definition applied to Corn/Wheat)
    ratio = (corn / wheat).rename("ratio")

    # Paper-derived continuous feature
    ratio_zscore_50 = build_ratio_zscore_50(ratio, window=RATIO_ZSCORE_WINDOW)

    log_corn = np.log(corn)
    log_wheat = np.log(wheat)
    beta_roll = _rolling_ols_beta(log_corn, log_wheat, window=ROLLING_HEDGE_WINDOW).rename(
        "rolling_beta_60"
    )
    log_spread = (log_corn - beta_roll * log_wheat).rename("log_spread")
    spread_zscore_60 = (
        (log_spread - log_spread.rolling(SPREAD_ZSCORE_WINDOW, min_periods=SPREAD_ZSCORE_WINDOW).mean())
        / log_spread.rolling(SPREAD_ZSCORE_WINDOW, min_periods=SPREAD_ZSCORE_WINDOW).std()
    ).rename("spread_zscore_60")
    spread_zscore_60 = spread_zscore_60.replace([np.inf, -np.inf], np.nan)

    corn_ret = corn.pct_change(fill_method=None)
    wheat_ret = wheat.pct_change(fill_method=None)
    rv_corn_20 = corn_ret.rolling(REALIZED_VOL_WINDOW, min_periods=REALIZED_VOL_WINDOW).std().rename(
        "rv_corn_20"
    )
    rv_wheat_20 = wheat_ret.rolling(REALIZED_VOL_WINDOW, min_periods=REALIZED_VOL_WINDOW).std().rename(
        "rv_wheat_20"
    )

    # 1-day ratio return known at close of t (uses ratio_t and ratio_{t-1} only)
    ret_ratio_1d = ratio.pct_change(fill_method=None).rename("ret_ratio_1d")

    feats = pd.concat(
        [
            prices,
            ratio,
            ratio_zscore_50,
            beta_roll,  # portfolio hedge ratio — NOT a model feature (excluded from FEATURE_COLUMNS)
            log_spread,
            spread_zscore_60,
            rv_corn_20,
            rv_wheat_20,
            ret_ratio_1d,
        ],
        axis=1,
    )
    # Next-day target lives beside features for modeling convenience (not a feature)
    feats["target_ret_ratio_1d_fwd"] = ratio.pct_change(fill_method=None).shift(-1)
    return feats


def write_features(
    features: pd.DataFrame,
    *,
    source_panel: str,
    data_source: str = "bloomberg",
    stem: str = "features",
) -> dict:
    """Write features.parquet/.csv and feature_manifest.json under data/features/."""
    ensure_dirs()
    parquet_path = FEATURES_ROOT / f"{stem}.parquet"
    csv_path = FEATURES_ROOT / f"{stem}.csv"
    manifest_path = FEATURES_ROOT / "feature_manifest.json"

    out = features.copy()
    out.index = pd.to_datetime(out.index)
    out.to_parquet(parquet_path)
    out.to_csv(csv_path, index_label="date")

    n_valid_paper = int(out["ratio_zscore_50"].notna().sum()) if "ratio_zscore_50" in out.columns else 0
    manifest = {
        "created_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_source": data_source,
        "source_panel": source_panel,
        "calendar_convention": CALENDAR_CONVENTION,
        "feature_columns": list(FEATURE_COLUMNS),
        "price_columns": list(PAIR_COLS),
        "aux_columns": ["rolling_beta_60"],
        "target_column": "target_ret_ratio_1d_fwd",
        "target_definition": "ratio_{t+1}/ratio_t - 1 (shifted; features use only info through t)",
        "paper_feature": {
            "name": "ratio_zscore_50",
            "category": "Internal / Endogenous",
            "type": "Statistical mean-reversion feature",
            "source": "van Unen (2023)",
            "citation": (
                "van Unen, Q. (2023). Pairs Trading in Agricultural Commodity Futures Markets. "
                "BSc thesis, Erasmus University Rotterdam, §4.2."
            ),
            "window": RATIO_ZSCORE_WINDOW,
            "n_non_null": n_valid_paper,
        },
        "features": FEATURE_MANIFEST_ENTRIES,
        "outputs": {
            "parquet": to_rel(parquet_path),
            "csv": to_rel(csv_path),
            "manifest": to_rel(manifest_path),
        },
        "rows": int(len(out)),
        "date_start": str(out.index.min().date()) if len(out) else None,
        "date_end": str(out.index.max().date()) if len(out) else None,
        "no_lookahead_notes": [
            "All rolling windows are trailing (center=False).",
            "target_ret_ratio_1d_fwd is the only forward-looking column; excluded from X at fit time.",
            "Rolling hedge beta uses only the trailing ROLLING_HEDGE_WINDOW observations.",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
