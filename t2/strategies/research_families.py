"""Independent causal strategy families for shared T2 backtests.

All functions return close-confirmed signals and positive ATR-based stop
distances. The common engine enters at the next candle open. ORB uses a
configurable UTC opening range. VWAP refuses data without a source volume column.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from t2.data.ohlcv import validate_ohlcv


def _atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    previous = data["close"].shift(1)
    tr = pd.concat([
        data["high"] - data["low"],
        (data["high"] - previous).abs(),
        (data["low"] - previous).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def _stops(data: pd.DataFrame, atr_period: int, multiple: float) -> tuple[pd.Series, pd.Series]:
    if not np.isfinite(multiple) or multiple <= 0:
        raise ValueError("atr_stop_multiple must be positive and finite")
    atr = _atr(data, atr_period)
    fallback = (data["high"] - data["low"]).rolling(atr_period, min_periods=1).mean()
    distance = (atr * multiple).where(atr > 0, fallback).clip(lower=1e-8)
    valid = atr.notna() & np.isfinite(atr) & (atr > 0)
    return distance, valid


def ema_crossover(
    candles: pd.DataFrame, *, fast: int = 20, slow: int = 50,
    atr_period: int = 14, atr_stop_multiple: float = 1.5,
) -> pd.DataFrame:
    """EMA crossover, with long and short signals evaluated at candle close."""
    if not (isinstance(fast, int) and isinstance(slow, int) and 1 <= fast < slow):
        raise ValueError("Require integer periods with 1 <= fast < slow")
    data = validate_ohlcv(candles)
    distance, valid = _stops(data, atr_period, atr_stop_multiple)
    close = data["close"]
    fast_line = close.ewm(span=fast, adjust=False, min_periods=fast).mean()
    slow_line = close.ewm(span=slow, adjust=False, min_periods=slow).mean()
    long = (fast_line > slow_line) & (fast_line.shift(1) <= slow_line.shift(1)) & valid
    short = (fast_line < slow_line) & (fast_line.shift(1) >= slow_line.shift(1)) & valid
    return pd.DataFrame({
        "long_signal": long.fillna(False).astype(bool),
        "short_signal": short.fillna(False).astype(bool),
        "stop_distance": distance,
    })


def orb_breakout(
    candles: pd.DataFrame, *, session_start_utc: int = 8,
    range_minutes: int = 30, atr_period: int = 14,
    atr_stop_multiple: float = 1.5,
) -> pd.DataFrame:
    """One breakout signal per UTC date after a configurable opening range.

    This is a generic UTC session hypothesis, not an exchange-specific session
    calendar. Configure the hour to match the market session being researched.
    """
    if not (isinstance(session_start_utc, int) and 0 <= session_start_utc <= 23):
        raise ValueError("session_start_utc must be an hour from 0 to 23")
    if isinstance(range_minutes, bool) or not isinstance(range_minutes, int) or range_minutes < 5:
        raise ValueError("range_minutes must be an integer >= 5")
    data = validate_ohlcv(candles)
    distance, valid = _stops(data, atr_period, atr_stop_multiple)
    ts = data["timestamp"]
    day = ts.dt.strftime("%Y-%m-%d")
    minute = ts.dt.hour * 60 + ts.dt.minute
    start = session_start_utc * 60
    in_range = (minute >= start) & (minute < start + range_minutes)
    range_frame = data.assign(_day=day, _in_range=in_range)
    highs = range_frame.loc[range_frame["_in_range"]].groupby("_day")["high"].max()
    lows = range_frame.loc[range_frame["_in_range"]].groupby("_day")["low"].min()
    range_high = day.map(highs)
    range_low = day.map(lows)
    after_range = minute >= start + range_minutes
    long_raw = (data["close"] > range_high) & after_range & valid
    short_raw = (data["close"] < range_low) & after_range & valid
    long_signal = long_raw & ~long_raw.groupby(day).cummax().shift(fill_value=False)
    short_signal = short_raw & ~short_raw.groupby(day).cummax().shift(fill_value=False)
    return pd.DataFrame({
        "long_signal": long_signal.fillna(False).astype(bool),
        "short_signal": short_signal.fillna(False).astype(bool),
        "stop_distance": distance,
    })


def vwap_ema(
    candles: pd.DataFrame, *, ema_period: int = 20,
    atr_period: int = 14, atr_stop_multiple: float = 1.5,
) -> pd.DataFrame:
    """Session VWAP + EMA trend filter; requires genuine source volume."""
    data = validate_ohlcv(candles)
    if "volume" not in data.columns:
        raise ValueError("VWAP requires genuine source volume; volume will not be fabricated")
    if (data["volume"] <= 0).all():
        raise ValueError("VWAP requires positive source volume")
    if not isinstance(ema_period, int) or ema_period < 2:
        raise ValueError("ema_period must be an integer >= 2")
    distance, valid = _stops(data, atr_period, atr_stop_multiple)
    typical = (data["high"] + data["low"] + data["close"]) / 3.0
    day = data["timestamp"].dt.strftime("%Y-%m-%d")
    pv = typical * data["volume"]
    vwap = pv.groupby(day).cumsum() / data["volume"].groupby(day).cumsum().replace(0, np.nan)
    ema = data["close"].ewm(span=ema_period, adjust=False, min_periods=ema_period).mean()
    long = (data["close"] > vwap) & (data["close"] > ema) & valid
    short = (data["close"] < vwap) & (data["close"] < ema) & valid
    return pd.DataFrame({
        "long_signal": long.fillna(False).astype(bool),
        "short_signal": short.fillna(False).astype(bool),
        "stop_distance": distance,
    })
