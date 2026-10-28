import numpy as np
import pandas as pd
import pytest

from t2.improvement.feedback import diagnose_experiment, research_guardrails
from t2.improvement.gpu_ml import build_causal_features, make_direction_labels


def sample_candles(n=160):
    timestamps = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")
    close = 2000 + np.cumsum(np.sin(np.arange(n) / 4) * 0.2 + 0.03)
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": close - 0.1,
        "high": close + 0.3,
        "low": close - 0.3,
        "close": close,
    })


def test_features_are_causal_and_have_expected_columns():
    candles = sample_candles()
    features = build_causal_features(candles)
    assert len(features) == len(candles)
    assert "ema_gap_atr" in features
    changed = candles.copy()
    changed.loc[len(changed)-1, "close"] += 100
    changed.loc[len(changed)-1, "high"] += 100
    changed.loc[len(changed)-1, "open"] += 100
    original = build_causal_features(candles)
    revised = build_causal_features(changed)
    pd.testing.assert_series_equal(original.iloc[:-1]["ret_1"], revised.iloc[:-1]["ret_1"])


def test_direction_labels_leave_future_horizon_unknown():
    labels = make_direction_labels(sample_candles(), horizon=3)
    assert labels.iloc[:-3].notna().all()
    assert labels.iloc[-3:].isna().all()


def test_feedback_flags_failed_validation_and_cost_sensitive_edge():
    advice = diagnose_experiment({
        "validation_profit_factor": 0.9,
        "validation_expectancy_per_trade": -1,
        "test_profit_factor": 1.1,
        "test_expectancy_per_trade": 1,
        "test_pnl_by_cost": {"0.0": 10, "0.02": -30},
    })
    findings = {item["finding"] for item in advice}
    assert "validation_edge_not_established" in findings
    assert "edge_erased_by_costs" in findings


def test_guardrails_never_enable_live_trading_or_auto_promotion():
    rules = research_guardrails()
    assert rules["live_trading_enabled"] is False
    assert rules["allow_order_submission"] is False
    assert rules["automatic_promotion"] is False
    assert rules["test_set_may_select_candidates"] is False
