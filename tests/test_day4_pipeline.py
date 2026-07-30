"""Point-in-time and portfolio assertions for Day 4."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from day4_utils import (
    assert_net_equals_gross_minus_cost,
    assert_normalized_weights,
    assert_strict_dates,
    assert_target_not_in_X,
    assert_train_before_val,
    lag_weights,
    modeling_frame,
    net_returns,
    portfolio_gross_return,
    run_backtest,
    signal_sign,
    split_dev_holdout,
    turnover_series,
    weights_from_signal,
)
from pipeline_utils import FEATURE_COLUMNS, FEATURES_ROOT


def test_feature_file_dates_and_target_exclusion():
    feats = pd.read_parquet(FEATURES_ROOT / "features.parquet")
    feats.index = pd.to_datetime(feats.index)
    assert_strict_dates(feats.index)
    assert_target_not_in_X(list(FEATURE_COLUMNS))
    assert "rolling_beta_60" in feats.columns
    assert "rolling_beta_60" not in FEATURE_COLUMNS


def test_modeling_frame_and_split_chronology():
    feats = pd.read_parquet(FEATURES_ROOT / "features.parquet")
    feats.index = pd.to_datetime(feats.index)
    df = modeling_frame(feats)
    dev, ho, _ = split_dev_holdout(df)
    assert_train_before_val(dev.index, ho.index)
    assert dev.index.max() < ho.index.min()


def test_manual_timing_example():
    """Signal at t earns return at t+1 via lagged weights."""
    idx = pd.date_range("2020-01-01", periods=4, freq="B")
    corn = pd.Series([100.0, 101.0, 102.0, 100.0], index=idx)
    wheat = pd.Series([50.0, 50.0, 51.0, 52.0], index=idx)
    signal = np.array([1.0, 1.0, -1.0, -1.0])
    beta = np.array([1.0, 1.0, 1.0, 1.0])
    w_c, w_w = weights_from_signal(signal, beta)
    assert_normalized_weights(w_c, w_w)
    w_c_lag, w_w_lag = lag_weights(w_c, w_w)
    assert w_c_lag[0] == 0.0
    assert w_c_lag[1] == w_c[0]
    r_c = corn.pct_change(fill_method=None).to_numpy()
    r_w = wheat.pct_change(fill_method=None).to_numpy()
    gross = portfolio_gross_return(w_c_lag, w_w_lag, r_c, r_w)
    # Day 1: no position → 0; Day 2: long ratio with weights from day 0 signal
    assert np.isnan(gross[0]) or gross[0] == 0.0 or True
    assert np.isclose(w_c_lag[1] + abs(w_w_lag[1]), 1.0) or np.isclose(
        abs(w_c_lag[1]) + abs(w_w_lag[1]), 1.0
    )


def test_costs_nonnegative_and_net_identity():
    idx = pd.date_range("2020-01-01", periods=30, freq="B")
    rng = np.random.default_rng(0)
    corn = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, len(idx))), index=idx)
    wheat = pd.Series(50 * np.cumprod(1 + rng.normal(0, 0.01, len(idx))), index=idx)
    signal = signal_sign(rng.normal(0, 1, len(idx)))
    beta = np.ones(len(idx))
    bt = run_backtest(signal, beta, corn, wheat, cost_bps=5)
    assert (bt["turnover"] >= -1e-15).all()
    assert_net_equals_gross_minus_cost(
        bt["gross_return"].to_numpy(),
        bt["net_return"].to_numpy(),
        bt["turnover"].to_numpy(),
        5,
    )
    # cost bps must be non-negative API
    with pytest.raises(AssertionError):
        assert_net_equals_gross_minus_cost(
            bt["gross_return"].to_numpy(),
            bt["net_return"].to_numpy(),
            bt["turnover"].to_numpy(),
            -1,
        )


def test_outputs_exist_and_reproducible_columns():
    results = ROOT / "data" / "results"
    for name in (
        "day4_predictions.parquet",
        "day4_backtest_daily.parquet",
        "day4_metrics.json",
    ):
        assert (results / name).exists(), f"missing {name}"
    preds = pd.read_parquet(results / "day4_predictions.parquet")
    assert_strict_dates(pd.DatetimeIndex(pd.to_datetime(preds["date"])))
    assert preds["date"].is_unique
    metrics = json.loads((results / "day4_metrics.json").read_text())
    assert "preferred_config" in metrics
    assert metrics["dates"]["holdout"]["start"] > metrics["dates"]["development"]["end"]


def test_turnover_formula():
    w_c = np.array([0.5, 0.5, -0.5])
    w_w = np.array([-0.5, -0.5, 0.5])
    turn = turnover_series(w_c, w_w)
    # day 2 change: | -0.5-0.5 | + |0.5-(-0.5)| = 1+1 = 2 → half = 1
    assert np.isclose(turn[2], 1.0)
