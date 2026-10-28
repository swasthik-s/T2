import numpy as np
import pandas as pd
import pytest

from t2.improvement.target_labels import make_target_before_stop_labels
from t2.strategies.research_families import ema_crossover, orb_breakout, vwap_ema


def sample_candles(n=240, volume=False):
    ts = pd.date_range("2025-01-01", periods=n, freq="5min", tz="UTC")
    close = 2000 + np.cumsum(np.sin(np.arange(n) / 7) * 0.8 + 0.03)
    data = {
        "timestamp": ts,
        "open": close - 0.1,
        "high": close + 0.35,
        "low": close - 0.35,
        "close": close,
    }
    if volume:
        data["volume"] = np.arange(n) % 100 + 1
    return pd.DataFrame(data)


def test_target_labels_are_binary_and_last_horizon_is_unknown():
    candles = sample_candles()
    labels = make_target_before_stop_labels(candles, pd.Series([0.5] * len(candles)), horizon=12)
    assert labels.iloc[:-12].isin([0.0, 1.0]).all()
    assert labels.iloc[-12:].isna().all()


def test_target_label_requires_aligned_stop_distance():
    candles = sample_candles()
    with pytest.raises(ValueError, match="length"):
        make_target_before_stop_labels(candles, pd.Series([1.0]), horizon=3)


def test_ema_family_returns_valid_boolean_signals():
    data = ema_crossover(sample_candles(), fast=5, slow=15)
    assert len(data) == 240
    assert data["long_signal"].dtype == bool
    assert data["short_signal"].dtype == bool
    assert (data["stop_distance"] > 0).all()


def test_orb_has_no_signals_before_range_is_complete():
    candles = sample_candles(400)
    signals = orb_breakout(candles, session_start_utc=0, range_minutes=30)
    assert not signals.loc[:5, "long_signal"].any()
    assert not signals.loc[:5, "short_signal"].any()


def test_vwap_refuses_missing_volume():
    with pytest.raises(ValueError, match="genuine source volume"):
        vwap_ema(sample_candles())


def test_vwap_works_with_source_volume():
    data = vwap_ema(sample_candles(volume=True))
    assert len(data) == 240
    assert (data["stop_distance"] > 0).all()
