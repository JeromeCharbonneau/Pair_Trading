#!/usr/bin/env python3
"""Residual diagnostics on out-of-sample Elastic Net (and ensemble) residuals."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan, het_white
from statsmodels.stats.stattools import durbin_watson

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline_utils import PROJECT_ROOT, to_rel  # noqa: E402

RESULTS = PROJECT_ROOT / "data" / "results"
FIGS = RESULTS / "figures"


def _safe(x):
    if isinstance(x, (float, np.floating)):
        x = float(x)
        return None if not np.isfinite(x) else x
    return x


def diagnose(resid: pd.Series, fitted: pd.Series, label: str) -> dict:
    r = resid.dropna()
    f = fitted.reindex(r.index)
    # regressor matrix for BP/White: constant + fitted
    exog = np.column_stack([np.ones(len(r)), f.to_numpy(dtype=float)])
    dw = float(durbin_watson(r))
    lb = acorr_ljungbox(r, lags=[1, 5, 10, 20], return_df=True)
    try:
        bp_lm, bp_lp, bp_f, bp_fp = het_breuschpagan(r, exog)
    except Exception as e:  # noqa: BLE001
        bp_lm = bp_lp = bp_f = bp_fp = float("nan")
        bp_err = str(e)
    else:
        bp_err = None
    try:
        wh_lm, wh_lp, wh_f, wh_fp = het_white(r, exog)
    except Exception as e:  # noqa: BLE001
        wh_lm = wh_lp = wh_f = wh_fp = float("nan")
        wh_err = str(e)
    else:
        wh_err = None
    jb_stat, jb_p = stats.jarque_bera(r)
    acf = {str(k): float(r.autocorr(lag=k)) for k in (1, 5, 10, 20)}

    interpretations = []
    # DW ~2 no AC; <1.5 positive AC
    if dw < 1.5:
        interpretations.append(
            {
                "test": "Durbin-Watson",
                "result": "reject independence (low DW)",
                "likely": "remaining mean reversion or momentum (serial correlation)",
            }
        )
    elif dw > 2.5:
        interpretations.append(
            {
                "test": "Durbin-Watson",
                "result": "suggests negative serial correlation",
                "likely": "remaining mean reversion",
            }
        )
    else:
        interpretations.append(
            {
                "test": "Durbin-Watson",
                "result": "no strong evidence of AR(1)",
                "likely": "not primarily serial-correlation driven",
            }
        )

    for lag in (1, 5, 10, 20):
        p = float(lb.loc[lag, "lb_pvalue"])
        if p < 0.05:
            interpretations.append(
                {
                    "test": f"Ljung-Box lag {lag}",
                    "result": "reject white-noise residuals",
                    "likely": "model misspecification and/or remaining serial structure (tradable only as a future hypothesis)",
                }
            )

    if bp_lp == bp_lp and bp_lp < 0.05:
        interpretations.append(
            {
                "test": "Breusch-Pagan",
                "result": "reject homoskedasticity",
                "likely": "changing volatility",
            }
        )
    if wh_lp == wh_lp and wh_lp < 0.05:
        interpretations.append(
            {
                "test": "White",
                "result": "reject homoskedasticity",
                "likely": "changing volatility / heteroskedasticity",
            }
        )
    if jb_p < 0.05:
        interpretations.append(
            {
                "test": "Jarque-Bera",
                "result": "reject normality",
                "likely": "fat tails",
            }
        )

    return {
        "label": label,
        "n": int(len(r)),
        "mean": _safe(r.mean()),
        "std": _safe(r.std(ddof=1)),
        "skewness": _safe(r.skew()),
        "excess_kurtosis": _safe(r.kurtosis()),
        "durbin_watson": _safe(dw),
        "ljung_box": {
            str(int(lag)): {
                "stat": _safe(float(lb.loc[lag, "lb_stat"])),
                "pvalue": _safe(float(lb.loc[lag, "lb_pvalue"])),
            }
            for lag in (1, 5, 10, 20)
        },
        "breusch_pagan": {
            "lm_stat": _safe(bp_lm),
            "lm_pvalue": _safe(bp_lp),
            "f_stat": _safe(bp_f),
            "f_pvalue": _safe(bp_fp),
            "error": bp_err,
        },
        "white": {
            "lm_stat": _safe(wh_lm),
            "lm_pvalue": _safe(wh_lp),
            "f_stat": _safe(wh_f),
            "f_pvalue": _safe(wh_fp),
            "error": wh_err,
        },
        "jarque_bera": {"stat": _safe(jb_stat), "pvalue": _safe(jb_p)},
        "residual_acf": acf,
        "interpretations": interpretations,
        "policy": (
            "Do not automatically add a new feature after seeing holdout residuals. "
            "Treat any discovered structure as a future hypothesis requiring a new untouched test period."
        ),
    }


def plot_diagnostics(resid: pd.Series, fitted: pd.Series, stem: str) -> dict:
    FIGS.mkdir(parents=True, exist_ok=True)
    r = resid.dropna()
    f = fitted.reindex(r.index)
    paths = {}

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(r.index, r.values, lw=0.7)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_title(f"{stem}: residuals over time")
    p = FIGS / f"{stem}_resid_time.png"
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["resid_time"] = to_rel(p)

    fig, ax = plt.subplots(figsize=(6, 3))
    pd.plotting.autocorrelation_plot(r, ax=ax)
    ax.set_title(f"{stem}: residual ACF")
    p = FIGS / f"{stem}_resid_acf.png"
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["resid_acf"] = to_rel(p)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(f, r, s=8, alpha=0.4)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("fitted")
    ax.set_ylabel("residual")
    ax.set_title(f"{stem}: residual vs fitted")
    p = FIGS / f"{stem}_resid_vs_fitted.png"
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["resid_vs_fitted"] = to_rel(p)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].hist(r, bins=40, density=True, alpha=0.8)
    axes[0].set_title("histogram")
    stats.probplot(r, dist="norm", plot=axes[1])
    axes[1].set_title("QQ")
    fig.suptitle(f"{stem}: residual distribution")
    p = FIGS / f"{stem}_resid_hist_qq.png"
    fig.tight_layout()
    fig.savefig(p, dpi=120)
    plt.close(fig)
    paths["resid_hist_qq"] = to_rel(p)
    return paths


def main() -> None:
    preds = pd.read_parquet(RESULTS / "day4_predictions.parquet")
    preds["date"] = pd.to_datetime(preds["date"])
    preds = preds.set_index("date").sort_index()

    en_resid = preds["resid_elastic_net"]
    en_fitted = preds["pred_elastic_net"]
    ens_resid = preds["resid_ensemble"]
    ens_fitted = preds["pred_ensemble"]

    en_diag = diagnose(en_resid, en_fitted, "elastic_net_holdout")
    ens_diag = diagnose(ens_resid, ens_fitted, "ensemble_holdout")
    en_diag["figures"] = plot_diagnostics(en_resid, en_fitted, "elastic_net")
    ens_diag["figures"] = plot_diagnostics(ens_resid, ens_fitted, "ensemble")

    payload = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_predictions": "data/results/day4_predictions.parquet",
        "primary": en_diag,
        "ensemble_optional": ens_diag,
        "policy": en_diag["policy"],
    }
    out = RESULTS / "day4_residual_diagnostics.json"
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {to_rel(out)}")
    print(f"EN DW={en_diag['durbin_watson']:.3f} JB_p={en_diag['jarque_bera']['pvalue']}")


if __name__ == "__main__":
    main()
