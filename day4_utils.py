"""Day 4 helpers: time-series CV, signals, hedge-ratio-neutral portfolio, metrics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from scipy import stats
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipeline_utils import FEATURE_COLUMNS

ANNUALIZATION = 252
COST_BPS_SCENARIOS = (0, 2, 5)
LGBM_PARAMS = dict(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=15,
    min_child_samples=40,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=-1,
)
EN_L1_RATIOS = [0.1, 0.5, 0.9]
OUTER_N_SPLITS = 5
INNER_N_SPLITS = 3
HOLDOUT_FRAC = 0.20  # last 20% untouched
DEV_FRAC = 1.0 - HOLDOUT_FRAC


# ---------------------------------------------------------------------------
# Splits / models
# ---------------------------------------------------------------------------

def modeling_frame(features: pd.DataFrame) -> pd.DataFrame:
    """Rows with all model features, target, and rolling beta present."""
    need = list(FEATURE_COLUMNS) + [
        "target_ret_ratio_1d_fwd",
        "rolling_beta_60",
        "Corn",
        "Wheat",
    ]
    # ratio_zscore_50 is already in FEATURE_COLUMNS; keep column list unique
    need = list(dict.fromkeys(need))
    missing = [c for c in need if c not in features.columns]
    if missing:
        raise KeyError(f"features missing columns: {missing}")
    out = features.loc[:, need].dropna().sort_index()
    assert_strict_dates(out.index)
    return out


def split_dev_holdout(df: pd.DataFrame, dev_frac: float = DEV_FRAC) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    n = len(df)
    split = int(n * dev_frac)
    if split < 100 or n - split < 50:
        raise ValueError(f"dev/holdout split too small: n={n}, split={split}")
    return df.iloc[:split].copy(), df.iloc[split:].copy(), split


def outer_ts_splits(n_samples: int, n_splits: int = OUTER_N_SPLITS):
    return TimeSeriesSplit(n_splits=n_splits).split(np.arange(n_samples))


def fit_elastic_net(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                ElasticNetCV(
                    l1_ratio=EN_L1_RATIOS,
                    cv=TimeSeriesSplit(n_splits=INNER_N_SPLITS),
                    max_iter=20000,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipe.fit(X, y)
    return pipe


def fit_lightgbm(X: pd.DataFrame, y: pd.Series) -> LGBMRegressor:
    model = LGBMRegressor(**LGBM_PARAMS)
    model.fit(X, y)
    return model


def predict_model(model, X: pd.DataFrame) -> np.ndarray:
    return np.asarray(model.predict(X), dtype=float)


def ensemble_predict(
    pred_en: np.ndarray,
    pred_lgbm: np.ndarray,
    mu_en: float,
    sd_en: float,
    mu_lgbm: float,
    sd_lgbm: float,
) -> np.ndarray:
    """Equal-weight average of training-standardized predictions."""
    sd_en = sd_en if sd_en > 1e-12 else 1.0
    sd_lgbm = sd_lgbm if sd_lgbm > 1e-12 else 1.0
    z_en = (pred_en - mu_en) / sd_en
    z_lg = (pred_lgbm - mu_lgbm) / sd_lgbm
    return 0.5 * (z_en + z_lg)


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float((np.sign(y_true[mask]) == np.sign(y_pred[mask])).mean())


def spearman_ic(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 3:
        return float("nan")
    corr = stats.spearmanr(y_pred, y_true).correlation
    return float(corr) if corr == corr else float("nan")


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

def signal_sign(pred: np.ndarray | pd.Series) -> np.ndarray:
    p = np.asarray(pred, dtype=float)
    s = np.sign(p)
    s[p == 0] = 0.0
    return s


def signal_threshold(pred: np.ndarray | pd.Series, threshold: float) -> np.ndarray:
    p = np.asarray(pred, dtype=float)
    s = np.sign(p)
    s[np.abs(p) < threshold] = 0.0
    return s


def signal_capped(pred: np.ndarray | pd.Series, mu: float, sd: float, cap: float = 1.0) -> np.ndarray:
    p = np.asarray(pred, dtype=float)
    sd = sd if sd > 1e-12 else 1.0
    z = (p - mu) / sd
    return np.clip(z, -cap, cap)


def threshold_candidates_from_train(pred_train: np.ndarray) -> dict[str, float]:
    abs_p = np.abs(np.asarray(pred_train, dtype=float))
    abs_p = abs_p[np.isfinite(abs_p)]
    return {
        "p25": float(np.percentile(abs_p, 25)),
        "p50": float(np.percentile(abs_p, 50)),
        "p75": float(np.percentile(abs_p, 75)),
    }


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

def weights_from_signal(signal: np.ndarray | pd.Series, beta: np.ndarray | pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Hedge-ratio-neutral weights with |w_c| + |w_w| = 1 when active."""
    s = np.asarray(signal, dtype=float)
    b = np.asarray(beta, dtype=float)
    w_c_raw = s
    w_w_raw = -s * b
    scale = np.abs(w_c_raw) + np.abs(w_w_raw)
    w_c = np.zeros_like(w_c_raw)
    w_w = np.zeros_like(w_w_raw)
    active = scale > 1e-12
    w_c[active] = w_c_raw[active] / scale[active]
    w_w[active] = w_w_raw[active] / scale[active]
    return w_c, w_w


def lag_weights(w_c: np.ndarray, w_w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shift positions so signal at t earns returns at t+1 (implemented as lag)."""
    w_c_lag = np.empty_like(w_c)
    w_w_lag = np.empty_like(w_w)
    w_c_lag[0] = 0.0
    w_w_lag[0] = 0.0
    w_c_lag[1:] = w_c[:-1]
    w_w_lag[1:] = w_w[:-1]
    return w_c_lag, w_w_lag


def asset_returns(corn: pd.Series, wheat: pd.Series) -> tuple[pd.Series, pd.Series]:
    r_c = corn.pct_change(fill_method=None)
    r_w = wheat.pct_change(fill_method=None)
    return r_c, r_w


def portfolio_gross_return(
    w_c_lag: np.ndarray,
    w_w_lag: np.ndarray,
    r_c: np.ndarray,
    r_w: np.ndarray,
) -> np.ndarray:
    return w_c_lag * r_c + w_w_lag * r_w


def turnover_series(w_c: np.ndarray, w_w: np.ndarray) -> np.ndarray:
    """turnover_t = 0.5 * (|Δw_c| + |Δw_w|) using decision-time weights."""
    dw_c = np.empty_like(w_c)
    dw_w = np.empty_like(w_w)
    dw_c[0] = np.abs(w_c[0])
    dw_w[0] = np.abs(w_w[0])
    dw_c[1:] = np.abs(w_c[1:] - w_c[:-1])
    dw_w[1:] = np.abs(w_w[1:] - w_w[:-1])
    return 0.5 * (dw_c + dw_w)


def net_returns(gross: np.ndarray, turnover: np.ndarray, cost_bps: float) -> np.ndarray:
    cost_rate = cost_bps / 10000.0
    return gross - turnover * cost_rate


def run_backtest(
    signal: np.ndarray | pd.Series,
    beta: np.ndarray | pd.Series,
    corn: pd.Series,
    wheat: pd.Series,
    cost_bps: float = 0.0,
) -> pd.DataFrame:
    """Point-in-time pairs backtest; returns aligned to corn/wheat index."""
    idx = corn.index
    s = np.asarray(signal, dtype=float)
    b = np.asarray(beta, dtype=float)
    w_c, w_w = weights_from_signal(s, b)
    w_c_lag, w_w_lag = lag_weights(w_c, w_w)
    r_c = corn.pct_change(fill_method=None).to_numpy(dtype=float)
    r_w = wheat.pct_change(fill_method=None).to_numpy(dtype=float)
    gross = portfolio_gross_return(w_c_lag, w_w_lag, r_c, r_w)
    turn = turnover_series(w_c, w_w)
    # At return date t we hold w_{t-1} (lagged). Cost of trading into w_{t-1} is turn[t-1].
    turn_for_return = np.zeros_like(turn)
    turn_for_return[1:] = turn[:-1]

    net = net_returns(gross, turn_for_return, cost_bps)
    out = pd.DataFrame(
        {
            "signal": s,
            "w_corn": w_c,
            "w_wheat": w_w,
            "w_corn_lag": w_c_lag,
            "w_wheat_lag": w_w_lag,
            "r_corn": r_c,
            "r_wheat": r_w,
            "turnover": turn_for_return,
            "turnover_decision": turn,
            "gross_return": gross,
            "net_return": net,
        },
        index=idx,
    )
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def max_drawdown(cum: pd.Series) -> float:
    peak = cum.cummax()
    dd = cum / peak - 1.0
    return float(dd.min()) if len(dd) else float("nan")


def portfolio_metrics(
    returns: pd.Series,
    turnover: pd.Series,
    signal: pd.Series | None = None,
    ann: int = ANNUALIZATION,
) -> dict[str, Any]:
    r = returns.replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) == 0:
        return {"n": 0}
    t = turnover.reindex(r.index).fillna(0.0)
    cum = (1.0 + r).cumprod()
    total = float(cum.iloc[-1] - 1.0)
    n = len(r)
    years = n / ann
    arith = float(r.mean() * ann)
    geo = float(cum.iloc[-1] ** (1 / years) - 1) if years > 0 and cum.iloc[-1] > 0 else float("nan")
    vol = float(r.std(ddof=1) * np.sqrt(ann)) if n > 1 else float("nan")
    sharpe = float(r.mean() / r.std(ddof=1) * np.sqrt(ann)) if n > 1 and r.std(ddof=1) > 0 else float("nan")
    downside = r[r < 0]
    sortino = (
        float(r.mean() / downside.std(ddof=1) * np.sqrt(ann))
        if len(downside) > 1 and downside.std(ddof=1) > 0
        else float("nan")
    )
    mdd = max_drawdown(cum)
    calmar = float(arith / abs(mdd)) if mdd < 0 else float("nan")
    avg_turn = float(t.mean())
    ann_turn = float(t.mean() * ann)

    if signal is None:
        sig = pd.Series(0.0, index=r.index)
    else:
        sig = signal.reindex(r.index).fillna(0.0)
    active = sig != 0
    n_active = int(active.sum())
    pos_changes = int((sig.diff().fillna(sig.iloc[0] if len(sig) else 0) != 0).sum())
    pct_flat = float((sig == 0).mean())

    # Hit rate when active: sign of return matches sign of lagged signal intent
    # Approximate: when |signal|>0 on decision day, next return day hit — use same-index active mask on returns
    hit = float("nan")
    if n_active > 0:
        # Using lagged signal alignment: active positions are where w_lag != 0 ≈ signal.shift(1)
        sig_lag = sig.shift(1).fillna(0.0)
        mask = sig_lag != 0
        if mask.sum() > 0:
            hit = float((np.sign(r[mask]) == np.sign(sig_lag[mask])).mean())

    gains = r[r > 0]
    losses = r[r < 0]
    avg_gain = float(gains.mean()) if len(gains) else float("nan")
    avg_loss = float(losses.mean()) if len(losses) else float("nan")
    gl_ratio = float(avg_gain / abs(avg_loss)) if avg_loss == avg_loss and avg_loss != 0 else float("nan")

    return {
        "n": n,
        "cumulative_return": total,
        "ann_arithmetic_return": arith,
        "ann_geometric_return": geo,
        "ann_volatility": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "calmar": calmar,
        "avg_daily_turnover": avg_turn,
        "ann_turnover": ann_turn,
        "n_active_days": n_active,
        "n_position_changes": pos_changes,
        "pct_days_flat": pct_flat,
        "hit_rate_active": hit,
        "avg_gain": avg_gain,
        "avg_loss": avg_loss,
        "gain_loss_ratio": gl_ratio,
        "skewness": float(r.skew()),
        "excess_kurtosis": float(r.kurtosis()),
        "information_ratio_vs_zero": sharpe,  # market-neutral: IR ≈ Sharpe vs 0 benchmark
    }


def directional_ic(hit_rate: float) -> float:
    return 2.0 * hit_rate - 1.0


def effective_breadth(
    signal: pd.Series,
    returns: pd.Series,
    ann: int = ANNUALIZATION,
) -> dict[str, float]:
    """Estimate effective breadth via position changes/year and AR(1) n_eff."""
    sig = signal.fillna(0.0)
    changes = (sig.diff().fillna(0.0) != 0).astype(float)
    n = len(sig)
    years = n / ann if n else float("nan")
    changes_per_year = float(changes.sum() / years) if years and years > 0 else float("nan")

    r = returns.dropna()
    if len(r) > 3:
        r0 = r.iloc[:-1].to_numpy()
        r1 = r.iloc[1:].to_numpy()
        if np.std(r0) > 0 and np.std(r1) > 0:
            rho = float(np.corrcoef(r0, r1)[0, 1])
        else:
            rho = 0.0
        rho = max(min(rho, 0.99), -0.99)
        n_eff = float(len(r) * (1 - rho) / (1 + rho)) if (1 + rho) != 0 else float(len(r))
        breadth_ar = float(n_eff / years) if years and years > 0 else float("nan")
    else:
        rho = float("nan")
        n_eff = float("nan")
        breadth_ar = float("nan")

    return {
        "position_changes_per_year": changes_per_year,
        "return_ar1_rho": rho,
        "n_eff_ar1": n_eff,
        "effective_breadth_ar1_per_year": breadth_ar,
        "effective_breadth_used": changes_per_year,
    }


def ir_approx(hit_rate: float, eff_breadth: float) -> float:
    return directional_ic(hit_rate) * np.sqrt(max(eff_breadth, 0.0))


def deflated_sharpe_ratio(
    sharpe: float,
    n_obs: int,
    n_trials: int,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float | None:
    """Bailey & López de Prado Deflated Sharpe Ratio (approx). Returns Prob(SR*>0)-style DSR if feasible."""
    if n_obs < 10 or not np.isfinite(sharpe) or n_trials < 1:
        return None
    # Expected max Sharpe under N(0,1) trials approximation
    e_max = (1 - np.euler_gamma) * stats.norm.ppf(1 - 1 / n_trials) + np.euler_gamma * stats.norm.ppf(
        1 - 1 / (n_trials * np.e)
    )
    # SR variance non-normal adjustment
    sr_var = (1 - skew * sharpe + (kurt - 1) / 4 * sharpe**2) / (n_obs - 1)
    if sr_var <= 0:
        return None
    dsr = float(stats.norm.cdf((sharpe - e_max) / np.sqrt(sr_var)))
    return dsr


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

def assert_strict_dates(index: pd.DatetimeIndex) -> None:
    if not isinstance(index, pd.DatetimeIndex):
        raise AssertionError("index is not DatetimeIndex")
    if not index.is_monotonic_increasing:
        raise AssertionError("dates not strictly increasing")
    if index.has_duplicates:
        raise AssertionError("duplicate dates present")


def assert_target_not_in_X(feature_cols: Sequence[str], target: str = "target_ret_ratio_1d_fwd") -> None:
    if target in feature_cols:
        raise AssertionError(f"target {target} leaked into feature columns")


def assert_train_before_val(train_idx: pd.DatetimeIndex, val_idx: pd.DatetimeIndex) -> None:
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise AssertionError("empty train or val")
    if train_idx.max() >= val_idx.min():
        raise AssertionError("training dates do not strictly precede validation dates")


def assert_normalized_weights(w_c: np.ndarray, w_w: np.ndarray, atol: float = 1e-8) -> None:
    exposure = np.abs(w_c) + np.abs(w_w)
    active = exposure > atol
    if active.any():
        if not np.allclose(exposure[active], 1.0, atol=1e-6):
            raise AssertionError("active gross exposure not normalized to 1")
    if not np.isfinite(w_c).all() or not np.isfinite(w_w).all():
        raise AssertionError("non-finite weights")


def assert_net_equals_gross_minus_cost(
    gross: np.ndarray, net: np.ndarray, turnover: np.ndarray, cost_bps: float, atol: float = 1e-10
) -> None:
    if cost_bps < 0:
        raise AssertionError("transaction costs must be non-negative")
    expected = gross - turnover * (cost_bps / 10000.0)
    mask = np.isfinite(gross) & np.isfinite(net) & np.isfinite(turnover)
    if not np.allclose(net[mask], expected[mask], atol=atol, equal_nan=True):
        raise AssertionError("net != gross - turnover * cost")


@dataclass
class FoldMetrics:
    fold: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    n_train: int
    n_val: int
    en_hit: float
    en_ic: float
    lgbm_hit: float
    lgbm_ic: float
