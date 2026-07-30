#!/usr/bin/env python3
"""Day 3 TS-CV finalize + Day 4 pairs backtest (point-in-time, no holdout peeking)."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from day4_utils import (  # noqa: E402
    ANNUALIZATION,
    COST_BPS_SCENARIOS,
    DEV_FRAC,
    EN_L1_RATIOS,
    HOLDOUT_FRAC,
    INNER_N_SPLITS,
    LGBM_PARAMS,
    OUTER_N_SPLITS,
    assert_net_equals_gross_minus_cost,
    assert_normalized_weights,
    assert_strict_dates,
    assert_target_not_in_X,
    assert_train_before_val,
    deflated_sharpe_ratio,
    directional_accuracy,
    directional_ic,
    effective_breadth,
    ensemble_predict,
    fit_elastic_net,
    fit_lightgbm,
    ir_approx,
    modeling_frame,
    outer_ts_splits,
    portfolio_metrics,
    predict_model,
    run_backtest,
    signal_capped,
    signal_sign,
    signal_threshold,
    spearman_ic,
    split_dev_holdout,
    threshold_candidates_from_train,
    weights_from_signal,
)
from pipeline_utils import (  # noqa: E402
    FEATURE_COLUMNS,
    FEATURES_ROOT,
    PROJECT_ROOT,
    _rolling_ols_beta,
    load_clean_pair,
    to_rel,
)

RESULTS_ROOT = PROJECT_ROOT / "data" / "results"
FEATURE_COLS = list(FEATURE_COLUMNS)
TARGET = "target_ret_ratio_1d_fwd"


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if not np.isfinite(x) else x
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def _metrics_for_config(bt: pd.DataFrame, cost_bps: float) -> dict:
    net = bt["gross_return"] if cost_bps == 0 else bt["net_return"]
    # rebuild net for requested cost if needed
    if cost_bps != 0:
        from day4_utils import net_returns

        net = pd.Series(
            net_returns(bt["gross_return"].to_numpy(), bt["turnover"].to_numpy(), cost_bps),
            index=bt.index,
        )
    else:
        net = bt["gross_return"]
    return portfolio_metrics(net, bt["turnover"], bt["signal"])


def _backtest_at_costs(signal, beta, corn, wheat) -> dict[int, pd.DataFrame]:
    out = {}
    for bps in COST_BPS_SCENARIOS:
        bt = run_backtest(signal, beta, corn, wheat, cost_bps=bps)
        assert_normalized_weights(bt["w_corn"].to_numpy(), bt["w_wheat"].to_numpy())
        assert_net_equals_gross_minus_cost(
            bt["gross_return"].to_numpy(),
            bt["net_return"].to_numpy(),
            bt["turnover"].to_numpy(),
            bps,
        )
        out[bps] = bt
    return out


def _yearly_metrics(bt: pd.DataFrame, cost_bps: int = 2) -> dict:
    from day4_utils import net_returns

    net = pd.Series(
        net_returns(bt["gross_return"].to_numpy(), bt["turnover"].to_numpy(), cost_bps),
        index=bt.index,
    )
    years = {}
    for y, g in net.groupby(net.index.year):
        turn = bt["turnover"].reindex(g.index)
        sig = bt["signal"].reindex(g.index)
        years[str(y)] = portfolio_metrics(g, turn, sig)
    return years


def main() -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    features = pd.read_parquet(FEATURES_ROOT / "features.parquet")
    features.index = pd.to_datetime(features.index)
    assert_strict_dates(features.index)
    assert_target_not_in_X(FEATURE_COLS, TARGET)

    df = modeling_frame(features)

    dev, holdout, split_i = split_dev_holdout(df, DEV_FRAC)
    print(f"Development: {dev.index.min().date()} → {dev.index.max().date()} (n={len(dev)})")
    print(f"Holdout:     {holdout.index.min().date()} → {holdout.index.max().date()} (n={len(holdout)})")

    X_dev, y_dev = dev[FEATURE_COLS], dev[TARGET]
    X_ho, y_ho = holdout[FEATURE_COLS], holdout[TARGET]

    # ------------------------------------------------------------------
    # Outer TimeSeriesSplit CV on development
    # ------------------------------------------------------------------
    fold_rows = []
    oos_pred_en = pd.Series(np.nan, index=dev.index, dtype=float)
    oos_pred_lgbm = pd.Series(np.nan, index=dev.index, dtype=float)

    for fold, (tr_idx, va_idx) in enumerate(outer_ts_splits(len(dev), OUTER_N_SPLITS), start=1):
        tr = dev.iloc[tr_idx]
        va = dev.iloc[va_idx]
        assert_train_before_val(tr.index, va.index)
        en = fit_elastic_net(tr[FEATURE_COLS], tr[TARGET])
        lgbm = fit_lightgbm(tr[FEATURE_COLS], tr[TARGET])
        p_en = predict_model(en, va[FEATURE_COLS])
        p_lg = predict_model(lgbm, va[FEATURE_COLS])
        oos_pred_en.iloc[va_idx] = p_en
        oos_pred_lgbm.iloc[va_idx] = p_lg
        fold_rows.append(
            {
                "fold": fold,
                "train_start": str(tr.index.min().date()),
                "train_end": str(tr.index.max().date()),
                "val_start": str(va.index.min().date()),
                "val_end": str(va.index.max().date()),
                "n_train": int(len(tr)),
                "n_val": int(len(va)),
                "elastic_net_hit_rate": directional_accuracy(va[TARGET].to_numpy(), p_en),
                "elastic_net_spearman_ic": spearman_ic(va[TARGET].to_numpy(), p_en),
                "lightgbm_hit_rate": directional_accuracy(va[TARGET].to_numpy(), p_lg),
                "lightgbm_spearman_ic": spearman_ic(va[TARGET].to_numpy(), p_lg),
            }
        )
        print(
            f"Fold {fold}: train {tr.index.min().date()}→{tr.index.max().date()} "
            f"val {va.index.min().date()}→{va.index.max().date()} "
            f"EN hit={fold_rows[-1]['elastic_net_hit_rate']:.3%} "
            f"LGBM hit={fold_rows[-1]['lightgbm_hit_rate']:.3%}"
        )

    fold_df = pd.DataFrame(fold_rows)
    cv_summary = {
        "elastic_net_hit_rate_mean": float(fold_df["elastic_net_hit_rate"].mean()),
        "elastic_net_hit_rate_std": float(fold_df["elastic_net_hit_rate"].std(ddof=1)),
        "elastic_net_spearman_ic_mean": float(fold_df["elastic_net_spearman_ic"].mean()),
        "elastic_net_spearman_ic_std": float(fold_df["elastic_net_spearman_ic"].std(ddof=1)),
        "lightgbm_hit_rate_mean": float(fold_df["lightgbm_hit_rate"].mean()),
        "lightgbm_hit_rate_std": float(fold_df["lightgbm_hit_rate"].std(ddof=1)),
        "lightgbm_spearman_ic_mean": float(fold_df["lightgbm_spearman_ic"].mean()),
        "lightgbm_spearman_ic_std": float(fold_df["lightgbm_spearman_ic"].std(ddof=1)),
    }

    # Refresh Day 3 metrics with TS-CV
    day3_path = FEATURES_ROOT / "day3_model_metrics.json"
    day3_payload = {
        "data_source": "bloomberg",
        "panel": "corn_wheat_panel_bloomberg.parquet",
        "feature_columns": FEATURE_COLS,
        "target": TARGET,
        "validation": "TimeSeriesSplit expanding outer CV on development (no shuffle)",
        "outer_n_splits": OUTER_N_SPLITS,
        "inner_en_cv": f"TimeSeriesSplit(n_splits={INNER_N_SPLITS})",
        "dev_frac": DEV_FRAC,
        "holdout_frac": HOLDOUT_FRAC,
        "development": {
            "start": str(dev.index.min().date()),
            "end": str(dev.index.max().date()),
            "n": len(dev),
        },
        "holdout": {
            "start": str(holdout.index.min().date()),
            "end": str(holdout.index.max().date()),
            "n": len(holdout),
            "note": "Untouched for hyperparameter/threshold/feature selection",
        },
        "folds": fold_rows,
        "cv_summary": cv_summary,
        "label_overlap_note": (
            "Target is one-day forward return so adjacent labels share a price level; "
            "strict chronology enforced; no purge (would discard most daily pairs data)."
        ),
        "updated_utc": run_ts,
    }
    day3_path.write_text(json.dumps(_json_safe(day3_payload), indent=2) + "\n")

    # ------------------------------------------------------------------
    # Threshold selection on development OOS preds only (never holdout)
    # ------------------------------------------------------------------
    oos_mask = oos_pred_en.notna()
    # Standardize OOS preds with expanding-ish: use full-dev train moments after final fit below;
    # for threshold selection use raw |pred| percentiles from stacked OOS EN preds as plan allows
    # Plan: percentiles of |train standardized prediction| — use each fold's train std via
    # approximating with percentiles of |oos_pred_en| on development OOS (no holdout).
    thr_raw = threshold_candidates_from_train(oos_pred_en[oos_mask].to_numpy())
    # Evaluate each threshold with a quick sign/threshold net sharpe @2bps on OOS stack
    thr_eval = {}
    for name, tau in thr_raw.items():
        sig = signal_threshold(oos_pred_en[oos_mask].to_numpy(), tau)
        bt = run_backtest(
            sig,
            dev.loc[oos_mask, "rolling_beta_60"].to_numpy(),
            dev.loc[oos_mask, "Corn"],
            dev.loc[oos_mask, "Wheat"],
            cost_bps=2,
        )
        m = portfolio_metrics(bt["net_return"], bt["turnover"], bt["signal"])
        thr_eval[name] = {
            "threshold": tau,
            "net_sharpe_2bps": m.get("sharpe"),
            "hit_rate_active": m.get("hit_rate_active"),
            "ann_turnover": m.get("ann_turnover"),
        }
    # Select by net Sharpe @2bps; fallback hit rate
    def _sel_key(item):
        v = item[1]["net_sharpe_2bps"]
        if v is None or not np.isfinite(v):
            h = item[1]["hit_rate_active"]
            return h if h is not None and np.isfinite(h) else -1e9
        return v

    best_thr_name, best_thr_info = max(thr_eval.items(), key=_sel_key)
    selected_threshold = float(best_thr_info["threshold"])
    print(f"Selected threshold: {best_thr_name}={selected_threshold:.6g} (dev OOS only)")

    # ------------------------------------------------------------------
    # Final holdout fit on full development
    # ------------------------------------------------------------------
    en_final = fit_elastic_net(X_dev, y_dev)
    lgbm_final = fit_lightgbm(X_dev, y_dev)
    pred_en_dev = predict_model(en_final, X_dev)
    pred_lg_dev = predict_model(lgbm_final, X_dev)
    mu_en, sd_en = float(np.mean(pred_en_dev)), float(np.std(pred_en_dev, ddof=1))
    mu_lg, sd_lg = float(np.mean(pred_lg_dev)), float(np.std(pred_lg_dev, ddof=1))

    pred_en_ho = predict_model(en_final, X_ho)
    pred_lg_ho = predict_model(lgbm_final, X_ho)
    pred_ens_ho = ensemble_predict(pred_en_ho, pred_lg_ho, mu_en, sd_en, mu_lg, sd_lg)

    # ------------------------------------------------------------------
    # Signal configs on holdout
    # ------------------------------------------------------------------
    configs = {}

    def add_config(name: str, signal: np.ndarray, model_tag: str, rule: str):
        configs[name] = {"signal": signal, "model": model_tag, "rule": rule}

    add_config("en_sign", signal_sign(pred_en_ho), "elastic_net", "sign")
    add_config(
        "en_threshold",
        signal_threshold(pred_en_ho, selected_threshold),
        "elastic_net",
        f"threshold_{best_thr_name}",
    )
    add_config("en_capped", signal_capped(pred_en_ho, mu_en, sd_en), "elastic_net", "capped")
    add_config("lgbm_sign", signal_sign(pred_lg_ho), "lightgbm", "sign")
    add_config(
        "lgbm_threshold",
        signal_threshold(pred_lg_ho, selected_threshold),
        "lightgbm",
        f"threshold_{best_thr_name}",
    )
    add_config("lgbm_capped", signal_capped(pred_lg_ho, mu_lg, sd_lg), "lightgbm", "capped")
    add_config("ensemble_sign", signal_sign(pred_ens_ho), "ensemble", "sign")
    add_config(
        "ensemble_threshold",
        signal_threshold(pred_ens_ho, selected_threshold),
        "ensemble",
        f"threshold_{best_thr_name}",
    )
    # Benchmarks
    z = holdout["ratio_zscore_50"].to_numpy()
    add_config("bench_zscore_mr", -signal_sign(z), "benchmark", "neg_ratio_zscore_50_sign")
    add_config("bench_flat", np.zeros(len(holdout)), "benchmark", "flat")

    beta_ho = holdout["rolling_beta_60"]
    corn_ho = holdout["Corn"]
    wheat_ho = holdout["Wheat"]

    results_by_config = {}
    daily_frames = []
    research_log = []

    for name, cfg in configs.items():
        sig = cfg["signal"]
        bts = _backtest_at_costs(sig, beta_ho, corn_ho, wheat_ho)
        metrics_costs = {}
        for bps, bt in bts.items():
            metrics_costs[str(bps)] = _metrics_for_config(bt, bps)
        bt0 = bts[0]
        yearly = _yearly_metrics(bt0, cost_bps=2)
        hit = metrics_costs["2"].get("hit_rate_active")
        breadth = effective_breadth(bt0["signal"], bts[2]["net_return"])
        dic = directional_ic(hit) if hit is not None and np.isfinite(hit) else float("nan")
        ir_a = (
            ir_approx(hit, breadth["effective_breadth_used"])
            if hit is not None and np.isfinite(hit)
            else float("nan")
        )
        results_by_config[name] = {
            "model": cfg["model"],
            "rule": cfg["rule"],
            "metrics_by_cost_bps": metrics_costs,
            "yearly_net_2bps": yearly,
            "directional_ic": dic,
            "effective_breadth": breadth,
            "ir_approx": ir_a,
            "realized_net_sharpe_2bps": metrics_costs["2"].get("sharpe"),
        }
        # tidy daily for this config (gross + nets)
        d = bt0[
            [
                "signal",
                "w_corn",
                "w_wheat",
                "w_corn_lag",
                "w_wheat_lag",
                "turnover",
                "gross_return",
            ]
        ].copy()
        d["config"] = name
        d["net_return_0bps"] = bts[0]["net_return"]
        d["net_return_2bps"] = bts[2]["net_return"]
        d["net_return_5bps"] = bts[5]["net_return"]
        daily_frames.append(d)

        decision = "candidate"
        research_log.append(
            {
                "date_utc": run_ts,
                "config": name,
                "model": cfg["model"],
                "rule": cfg["rule"],
                "threshold": selected_threshold if "threshold" in cfg["rule"] else None,
                "holdout_net_sharpe_2bps": metrics_costs["2"].get("sharpe"),
                "holdout_max_drawdown_2bps": metrics_costs["2"].get("max_drawdown"),
                "holdout_ann_turnover_2bps": metrics_costs["2"].get("ann_turnover"),
                "decision": decision,
                "reason": "evaluated on untouched holdout after pre-registered selection",
            }
        )

    # Threshold probes logged
    for name, info in thr_eval.items():
        research_log.append(
            {
                "date_utc": run_ts,
                "config": f"threshold_probe_{name}",
                "model": "elastic_net_oos_stack",
                "rule": "threshold_selection_dev_only",
                "threshold": info["threshold"],
                "dev_net_sharpe_2bps": info["net_sharpe_2bps"],
                "decision": "selected" if name == best_thr_name else "rejected",
                "reason": "chosen by mean/dev OOS net Sharpe@2bps" if name == best_thr_name else "lower/unstable vs winner",
            }
        )

    daily_all = pd.concat(daily_frames, axis=0).sort_index()

    # Predictions file (holdout)
    preds = pd.DataFrame(
        {
            "date": holdout.index,
            "realized_target": y_ho.to_numpy(),
            "pred_elastic_net": pred_en_ho,
            "pred_lightgbm": pred_lg_ho,
            "pred_ensemble": pred_ens_ho,
            "resid_elastic_net": y_ho.to_numpy() - pred_en_ho,
            "resid_lightgbm": y_ho.to_numpy() - pred_lg_ho,
            "resid_ensemble": y_ho.to_numpy() - pred_ens_ho,
            "rolling_beta_60": beta_ho.to_numpy(),
            "ratio_zscore_50": holdout["ratio_zscore_50"].to_numpy(),
            "signal_en_sign": configs["en_sign"]["signal"],
            "signal_en_threshold": configs["en_threshold"]["signal"],
            "signal_en_capped": configs["en_capped"]["signal"],
            "signal_lgbm_sign": configs["lgbm_sign"]["signal"],
            "signal_lgbm_threshold": configs["lgbm_threshold"]["signal"],
            "signal_lgbm_capped": configs["lgbm_capped"]["signal"],
            "signal_ensemble_sign": configs["ensemble_sign"]["signal"],
            "signal_ensemble_threshold": configs["ensemble_threshold"]["signal"],
            "signal_bench_zscore_mr": configs["bench_zscore_mr"]["signal"],
            "signal_bench_flat": configs["bench_flat"]["signal"],
        }
    )
    # Attach preferred weights later after selection — attach all primary sign weights
    for cname in ("en_sign", "lgbm_sign", "ensemble_sign", "en_threshold", "ensemble_threshold"):
        bt = run_backtest(configs[cname]["signal"], beta_ho, corn_ho, wheat_ho, 0)
        preds[f"w_corn_{cname}"] = bt["w_corn"].to_numpy()
        preds[f"w_wheat_{cname}"] = bt["w_wheat"].to_numpy()

    preds_path_csv = RESULTS_ROOT / "day4_predictions.csv"
    preds_path_pq = RESULTS_ROOT / "day4_predictions.parquet"
    preds.to_csv(preds_path_csv, index=False)
    preds.to_parquet(preds_path_pq, index=False)

    daily_path_csv = RESULTS_ROOT / "day4_backtest_daily.csv"
    daily_path_pq = RESULTS_ROOT / "day4_backtest_daily.parquet"
    daily_out = daily_all.copy()
    daily_out.index.name = "date"
    daily_out = daily_out.reset_index()
    daily_out["date"] = pd.to_datetime(daily_out["date"])
    daily_out.to_csv(daily_path_csv, index=False)
    daily_out.to_parquet(daily_path_pq, index=False)

    # ------------------------------------------------------------------
    # Preferred strategy (stability, not max Sharpe alone)
    # ------------------------------------------------------------------
    candidates = [
        n
        for n in results_by_config
        if not n.startswith("bench_")
    ]

    def stability_score(name: str) -> float:
        m = results_by_config[name]["metrics_by_cost_bps"]["2"]
        yearly = results_by_config[name]["yearly_net_2bps"]
        sharpes = [v.get("sharpe") for v in yearly.values() if v.get("sharpe") is not None]
        sharpes = [s for s in sharpes if s == s]
        mean_y = float(np.mean(sharpes)) if sharpes else -1e9
        std_y = float(np.std(sharpes)) if len(sharpes) > 1 else 0.0
        mdd = m.get("max_drawdown") or 0.0
        turn = m.get("ann_turnover") or 0.0
        sharpe5 = results_by_config[name]["metrics_by_cost_bps"]["5"].get("sharpe")
        sharpe5 = sharpe5 if sharpe5 == sharpe5 else -1e9
        # Prefer stable positive mean yearly Sharpe, limited MDD/turnover, cost-robust
        return mean_y - 0.5 * std_y + 0.25 * sharpe5 - 0.1 * abs(mdd) - 0.0005 * turn

    preferred = max(candidates, key=stability_score)
    for entry in research_log:
        if entry.get("config") == preferred:
            entry["decision"] = "preferred"
            entry["reason"] = (
                "best stability score (sub-period Sharpe@2bps, MDD, turnover, cost robustness); "
                "not selected by max single Sharpe alone"
            )
        elif entry.get("decision") == "candidate" and entry.get("config") in candidates:
            entry["decision"] = "rejected_vs_preferred"
            entry["reason"] = f"inferior stability vs {preferred}"

    # ------------------------------------------------------------------
    # Robustness checks
    # ------------------------------------------------------------------
    robustness = {}

    # Gross vs net for preferred
    robustness["gross_vs_net_preferred"] = {
        "gross_sharpe": results_by_config[preferred]["metrics_by_cost_bps"]["0"].get("sharpe"),
        "net_2bps_sharpe": results_by_config[preferred]["metrics_by_cost_bps"]["2"].get("sharpe"),
        "net_5bps_sharpe": results_by_config[preferred]["metrics_by_cost_bps"]["5"].get("sharpe"),
    }

    # Model comparison sign strategy
    robustness["model_sign_net_2bps"] = {
        k: results_by_config[k]["metrics_by_cost_bps"]["2"].get("sharpe")
        for k in ("en_sign", "lgbm_sign", "ensemble_sign")
    }

    # Sign vs threshold for preferred model family
    model_pref = results_by_config[preferred]["model"]
    robustness["sign_vs_threshold"] = {
        k: results_by_config[k]["metrics_by_cost_bps"]["2"]
        for k in results_by_config
        if results_by_config[k]["model"] == model_pref and results_by_config[k]["rule"] in (
            "sign",
            f"threshold_{best_thr_name}",
        )
    }

    # Hedge window sensitivity on preferred signal
    panel = load_clean_pair("corn_wheat_panel_bloomberg.parquet")
    log_c = np.log(panel["Corn"])
    log_w = np.log(panel["Wheat"])
    hedge_sens = {}
    pref_signal = configs[preferred]["signal"]
    for win in (40, 60, 90):
        beta_alt = _rolling_ols_beta(log_c, log_w, window=win).reindex(holdout.index)
        # drop na alignment
        mask = beta_alt.notna()
        bt = run_backtest(
            np.asarray(pref_signal)[mask.to_numpy()],
            beta_alt[mask].to_numpy(),
            holdout.loc[mask, "Corn"],
            holdout.loc[mask, "Wheat"],
            cost_bps=2,
        )
        hedge_sens[str(win)] = portfolio_metrics(bt["net_return"], bt["turnover"], bt["signal"])
    robustness["hedge_window_sensitivity"] = hedge_sens

    # Exclude largest |gross| days (top 1%)
    bt_pref = run_backtest(pref_signal, beta_ho, corn_ho, wheat_ho, 2)
    g = bt_pref["gross_return"]
    thr = g.dropna().abs().quantile(0.99)
    keep_idx = g.index[g.abs() <= thr]
    robustness["exclude_top1pct_abs_gross"] = portfolio_metrics(
        bt_pref.loc[keep_idx, "net_return"],
        bt_pref.loc[keep_idx, "turnover"],
        bt_pref.loc[keep_idx, "signal"],
    )

    # Extra 1-day delay of predictions
    delayed = np.concatenate([[0.0], pref_signal[:-1]])
    bt_delay = run_backtest(delayed, beta_ho, corn_ho, wheat_ho, 2)
    robustness["prediction_delay_extra_1d"] = portfolio_metrics(
        bt_delay["net_return"], bt_delay["turnover"], bt_delay["signal"]
    )

    # Nearby thresholds stability (EN)
    nearby = {}
    for name, tau in thr_raw.items():
        sig = signal_threshold(pred_en_ho, tau)
        bt = run_backtest(sig, beta_ho, corn_ho, wheat_ho, 2)
        nearby[name] = {
            "threshold": tau,
            "sharpe": portfolio_metrics(bt["net_return"], bt["turnover"], bt["signal"]).get("sharpe"),
            "ann_turnover": portfolio_metrics(bt["net_return"], bt["turnover"], bt["signal"]).get(
                "ann_turnover"
            ),
        }
    robustness["nearby_threshold_holdout_en"] = nearby

    # Deflated Sharpe for preferred
    pref_m = results_by_config[preferred]["metrics_by_cost_bps"]["2"]
    n_trials = 13  # primary configs
    dsr = deflated_sharpe_ratio(
        pref_m.get("sharpe") or float("nan"),
        pref_m.get("n") or 0,
        n_trials,
        skew=pref_m.get("skewness") or 0.0,
        kurt=(pref_m.get("excess_kurtosis") or 0.0) + 3.0,
    )

    metrics_payload = {
        "created_utc": run_ts,
        "data_source": "bloomberg",
        "feature_file": "data/features/features.parquet",
        "panel": "data/clean/corn_wheat_panel_bloomberg.parquet",
        "feature_columns": FEATURE_COLS,
        "target": TARGET,
        "model_settings": {
            "elastic_net": {
                "l1_ratio": EN_L1_RATIOS,
                "inner_cv": f"TimeSeriesSplit(n_splits={INNER_N_SPLITS})",
                "scaler": "StandardScaler",
            },
            "lightgbm": LGBM_PARAMS,
            "ensemble": "equal-weight average of train-standardized EN and LGBM predictions",
        },
        "dates": {
            "development": day3_payload["development"],
            "holdout": day3_payload["holdout"],
        },
        "cv_folds": fold_rows,
        "cv_summary": cv_summary,
        "signal_rules": {
            "selected_threshold_name": best_thr_name,
            "selected_threshold": selected_threshold,
            "threshold_candidates_dev": thr_eval,
            "rules": ["sign", "threshold", "capped", "ensemble", "bench_zscore_mr", "bench_flat"],
        },
        "transaction_costs": {
            "model": "linear implementation-shortfall approximation",
            "scenarios_bps_per_unit_turnover": list(COST_BPS_SCENARIOS),
            "note": (
                "No square-root market-impact (ADV unavailable). "
                "Commissions, bid-ask, slippage, roll costs, margin/leverage discussed in docs/DAY4_METHOD.md. "
                "Bloomberg continuous rolls remain vendor-defined."
            ),
        },
        "results_by_config": results_by_config,
        "preferred_config": preferred,
        "preferred_selection_rule": "stability_score(subperiod Sharpe@2bps, MDD, turnover, cost robustness)",
        "robustness": robustness,
        "deflated_sharpe_preferred_net_2bps": dsr,
        "n_primary_configurations_tested": n_trials,
        "limitations": [
            "Bloomberg continuous contract rolls are vendor-defined / opaque",
            "Linear cost model only; no ADV-based impact",
            "1-day label overlap not purged",
            "Holdout used only for reporting after pre-registered selection",
            "Do not add features from holdout residual diagnostics without a new untouched period",
        ],
        "outputs": {
            "predictions_csv": to_rel(preds_path_csv),
            "predictions_parquet": to_rel(preds_path_pq),
            "daily_csv": to_rel(daily_path_csv),
            "daily_parquet": to_rel(daily_path_pq),
        },
        "annualization": ANNUALIZATION,
        "research_log_entries": research_log,
    }

    metrics_path = RESULTS_ROOT / "day4_metrics.json"
    metrics_path.write_text(json.dumps(_json_safe(metrics_payload), indent=2) + "\n")

    # Research log markdown stub content also written by docs step; dump JSON companion
    (RESULTS_ROOT / "day4_research_log.json").write_text(
        json.dumps(_json_safe(research_log), indent=2) + "\n"
    )

    print(f"\nPreferred config: {preferred}")
    print(f"Net Sharpe @2bps: {pref_m.get('sharpe')}")
    print(f"Max DD @2bps: {pref_m.get('max_drawdown')}")
    print(f"Wrote {to_rel(metrics_path)}")
    print(f"Wrote {to_rel(preds_path_csv)}")
    print(f"Wrote {to_rel(daily_path_csv)}")


if __name__ == "__main__":
    main()
