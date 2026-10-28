"""Trading-outcome labels for supervised research; no execution or promotion.

A label is positive only when the profit target is reached before the stop
within the forward horizon. Same-candle stop/target ambiguity is labeled 0
conservatively. Entries are approximated at the next candle open, matching the
backtest engine's signal timing. The entry candle is not inspected, matching the OHLC engine; final rows without a complete horizon are NaN.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from t2.data.ohlcv import validate_ohlcv


def make_target_before_stop_labels(
    candles: pd.DataFrame,
    stop_distance: pd.Series,
    *,
    horizon: int = 12,
    reward_risk_ratio: float = 2.0,
) -> pd.Series:
    """Label whether target beats stop after next-open entry within horizon.

    Labels: 1=target first; 0=stop first, both hit in same candle, or timeout;
    NaN=features/entry/barrier/horizon unavailable. The target is a trade
    outcome, not a guarantee of profitability after costs.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if not np.isfinite(reward_risk_ratio) or reward_risk_ratio <= 0:
        raise ValueError("reward_risk_ratio must be positive and finite")
    data = validate_ohlcv(candles)
    distances = pd.to_numeric(pd.Series(stop_distance).reset_index(drop=True), errors="coerce")
    if len(distances) != len(data):
        raise ValueError("stop_distance length must match candles")
    labels = np.full(len(data), np.nan, dtype=np.float64)
    opens = data["open"].to_numpy(dtype=float)
    highs = data["high"].to_numpy(dtype=float)
    lows = data["low"].to_numpy(dtype=float)
    for i in range(max(0, len(data) - horizon - 1)):
        distance = float(distances.iloc[i])
        entry = float(opens[i + 1])
        if not np.isfinite(distance) or distance <= 0 or not np.isfinite(entry) or entry <= distance:
            continue
        stop = entry - distance
        target = entry + distance * reward_risk_ratio
        label = 0.0  # conservative default for timeout or ambiguous outcome
        for j in range(i + 2, i + horizon + 2):
            hit_stop = lows[j] <= stop
            hit_target = highs[j] >= target
            if hit_stop or hit_target:
                label = 1.0 if hit_target and not hit_stop else 0.0
                break
        labels[i] = label
    return pd.Series(labels, index=candles.index, name="target_before_stop")
