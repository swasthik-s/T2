import numpy as np
import pandas as pd

from t2.strategies.ema_atr_baseline import build_ema_atr_signals


def test_zero_atr_candles_have_positive_placeholder_and_no_signals():
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=80, freq="5min", tz="UTC"),
        "open": [100.0] * 80,
        "high": [100.0] * 80,
        "low": [100.0] * 80,
        "close": [100.0] * 80,
    })

    result = build_ema_atr_signals(frame, fast_ema=3, slow_ema=5, atr_period=3)

    assert np.isfinite(result["stop_distance"]).all()
    assert (result["stop_distance"] > 0).all()
    assert not result["long_signal"].any()
    assert not result["short_signal"].any()
    assert (result.loc[result["atr"].eq(0), "stop_distance"] > 0).all()


def test_signals_require_positive_atr():
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=12, freq="5min", tz="UTC"),
        "open": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
        "high": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112],
        "low": [99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        "close": [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
    })

    result = build_ema_atr_signals(frame, fast_ema=2, slow_ema=4, atr_period=3)

    invalid_atr = result["atr"].isna() | (result["atr"] <= 0)
    assert not result.loc[invalid_atr, "long_signal"].any()
    assert not result.loc[invalid_atr, "short_signal"].any()
    assert (result.loc[~invalid_atr, "stop_distance"] > 0).all()
