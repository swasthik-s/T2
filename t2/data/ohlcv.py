"""Deterministic OHLC/OHLCV loading and validation helpers.

Supports price-only OHLC feeds and OHLCV feeds. It never fabricates volume
when the source does not provide it.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ("timestamp", "open", "high", "low", "close")
_TIMESTAMP_ALIASES = ("timestamp", "datetime", "date_time", "date", "time")


class MarketDataError(ValueError):
    """Raised when market data is missing, malformed, or unsafe to use."""


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with normalized, canonical column names."""
    result = frame.copy()
    result.columns = [str(column).strip().lower().replace(" ", "_") for column in result.columns]
    if len(set(result.columns)) != len(result.columns):
        raise MarketDataError("Column names are duplicated after normalization.")

    timestamp_column = next((name for name in _TIMESTAMP_ALIASES if name in result.columns), None)
    if timestamp_column is None:
        raise MarketDataError(
            "Missing timestamp column. Expected one of: " + ", ".join(_TIMESTAMP_ALIASES)
        )
    if timestamp_column != "timestamp":
        if "timestamp" in result.columns:
            raise MarketDataError("Both timestamp and a timestamp alias are present.")
        result = result.rename(columns={timestamp_column: "timestamp"})

    missing = [name for name in REQUIRED_COLUMNS if name not in result.columns]
    if missing:
        raise MarketDataError("Missing required OHLC columns: " + ", ".join(missing))
    return result


def validate_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate OHLC or OHLCV candles and return a chronologically sorted copy.

    OHLC columns are mandatory. Volume is validated when supplied, but is not
    required: a missing volume column stays missing rather than being filled
    with synthetic values. Timestamps are parsed as UTC. Duplicate or invalid
    timestamps, non-finite values, non-positive prices, negative volume, and
    impossible OHLC relationships raise MarketDataError.

    Missing time intervals are not inferred because the expected trading
    calendar/session depends on the instrument and data source.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if frame.empty:
        raise MarketDataError("Market data contains no rows.")

    result = _normalize_columns(frame)
    try:
        result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="raise")
    except (ValueError, TypeError, OverflowError) as exc:
        raise MarketDataError("Timestamp column contains invalid values.") from exc
    if result["timestamp"].isna().any():
        raise MarketDataError("Timestamp column contains missing values.")
    if result["timestamp"].duplicated().any():
        duplicates = result.loc[result["timestamp"].duplicated(keep=False), "timestamp"]
        example = duplicates.iloc[0].isoformat()
        raise MarketDataError(f"Duplicate candle timestamp detected: {example}")

    for column in ("open", "high", "low", "close"):
        try:
            result[column] = pd.to_numeric(result[column], errors="raise")
        except (ValueError, TypeError) as exc:
            raise MarketDataError(f"Column '{column}' contains non-numeric values.") from exc
        values = result[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise MarketDataError(f"Column '{column}' contains NaN or infinite values.")

    if "volume" in result.columns:
        try:
            result["volume"] = pd.to_numeric(result["volume"], errors="raise")
        except (ValueError, TypeError) as exc:
            raise MarketDataError("Column 'volume' contains non-numeric values.") from exc
        volume = result["volume"].to_numpy(dtype=float)
        if not np.isfinite(volume).all():
            raise MarketDataError("Column 'volume' contains NaN or infinite values.")
        if (result["volume"] < 0).any():
            raise MarketDataError("Volume cannot be negative.")

    prices = result[["open", "high", "low", "close"]]
    if (prices <= 0).any().any():
        raise MarketDataError("OHLC prices must be greater than zero.")
    if (result["high"] < result[["open", "low", "close"]].max(axis=1)).any():
        raise MarketDataError("Invalid candle: high must be >= open, low, and close.")
    if (result["low"] > result[["open", "high", "close"]].min(axis=1)).any():
        raise MarketDataError("Invalid candle: low must be <= open, high, and close.")

    return result.sort_values("timestamp", kind="stable").reset_index(drop=True)


def load_ohlcv_csv(path: str | Path) -> pd.DataFrame:
    """Read an OHLC or OHLCV CSV file and validate it."""
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Market data CSV not found: {csv_path}")
    try:
        frame = pd.read_csv(csv_path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise MarketDataError(f"Could not read market data CSV: {csv_path}") from exc
    return validate_ohlcv(frame)


def describe_ohlcv(frame: pd.DataFrame) -> dict[str, object]:
    """Return a compact, JSON-friendly summary of validated market data."""
    data = validate_ohlcv(frame)
    has_volume = "volume" in data.columns
    return {
        "rows": int(len(data)),
        "start_utc": data["timestamp"].iloc[0].isoformat(),
        "end_utc": data["timestamp"].iloc[-1].isoformat(),
        "columns": list(data.columns),
        "volume_available": has_volume,
        "volume_note": "Source volume column present." if has_volume else "Volume not supplied by source; no synthetic volume added.",
        "duplicate_timestamps": 0,
        "missing_intervals_checked": False,
        "note": "Missing intervals require an instrument-specific session calendar and timeframe.",
    }
