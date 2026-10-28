"""First fixed-parameter XAUUSD M5 baseline: EMA crossover with ATR stops.

This is a research benchmark, not a claim of profitability. Indicators are
causal (each row uses only that candle and earlier data). No volume or VWAP
is used. Long and short sides are evaluated separately as directional
diagnostics; their results must not be added together as if they were one
portfolio because the separate runs can overlap in time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from t2.data.ohlcv import validate_ohlcv


def build_ema_atr_signals(
    candles: pd.DataFrame,
    fast_ema: int = 20,
    slow_ema: int = 50,
    atr_period: int = 14,
    atr_stop_multiple: float = 1.5,
) -> pd.DataFrame:
    """Return causal crossover signals and ATR stop distances.

    A signal is true only on the candle where fast EMA crosses slow EMA.
    The engine evaluates the signal at candle close and enters next candle open.
    """
    if not (isinstance(fast_ema, int) and isinstance(slow_ema, int)
            and isinstance(atr_period, int) and 1 <= fast_ema < slow_ema
            and atr_period >= 2):
        raise ValueError("Require 1 <= fast_ema < slow_ema and atr_period >= 2")
    if not np.isfinite(atr_stop_multiple) or atr_stop_multiple <= 0:
        raise ValueError("atr_stop_multiple must be positive and finite")

    data = validate_ohlcv(candles)
    close = data["close"].astype(float)
    high = data["high"].astype(float)
    low = data["low"].astype(float)

    fast = close.ewm(span=fast_ema, adjust=False, min_periods=fast_ema).mean()
    slow = close.ewm(span=slow_ema, adjust=False, min_periods=slow_ema).mean()

    previous_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(atr_period, min_periods=atr_period).mean()

    long_signal = (fast > slow) & (fast.shift(1) <= slow.shift(1))
    short_signal = (fast < slow) & (fast.shift(1) >= slow.shift(1))
    stop_distance = atr * atr_stop_multiple
    # The engine validates the full stop-distance vector, including warm-up rows
    # where no signal can fire. Use a positive range-based placeholder only for
    # those non-signal warm-up rows; valid signals still require a real ATR.
    fallback_distance = (high - low).rolling(atr_period, min_periods=1).mean()
    fallback_distance = fallback_distance.clip(lower=np.finfo(float).eps) * atr_stop_multiple
    stop_distance = stop_distance.fillna(fallback_distance)

    valid = atr.notna() & np.isfinite(stop_distance) & (stop_distance > 0)
    return pd.DataFrame({
        "long_signal": (long_signal & valid).fillna(False).astype(bool),
        "short_signal": (short_signal & valid).fillna(False).astype(bool),
        "stop_distance": stop_distance,
        "fast_ema": fast,
        "slow_ema": slow,
        "atr": atr,
    })
