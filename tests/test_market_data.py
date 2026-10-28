from pathlib import Path

import pandas as pd
import pytest

from t2.data.ohlcv import MarketDataError, describe_ohlcv, load_ohlcv_csv, validate_ohlcv


@pytest.fixture
def valid_candles():
    return pd.DataFrame(
        {
            "timestamp": ["2026-01-02T00:01:00Z", "2026-01-02T00:00:00Z"],
            "open": [101, 100],
            "high": [103, 102],
            "low": [99, 98],
            "close": [102.0, 101.0],
            "volume": [10, 12],
        }
    )


def test_valid_data_is_sorted_and_timestamp_is_utc(valid_candles):
    result = validate_ohlcv(valid_candles)
    assert result["timestamp"].is_monotonic_increasing
    assert str(result["timestamp"].dt.tz) == "UTC"
    assert result["close"].tolist() == [101, 102]


def test_timestamp_alias_and_column_case_are_normalized(valid_candles):
    frame = valid_candles.rename(columns={"timestamp": "DateTime", "open": "Open"})
    result = validate_ohlcv(frame)
    assert "timestamp" in result.columns
    assert "open" in result.columns


def test_duplicate_timestamps_are_rejected(valid_candles):
    valid_candles.loc[1, "timestamp"] = valid_candles.loc[0, "timestamp"]
    with pytest.raises(MarketDataError, match="Duplicate candle timestamp"):
        validate_ohlcv(valid_candles)


def test_invalid_high_is_rejected(valid_candles):
    valid_candles.loc[0, "high"] = 100
    with pytest.raises(MarketDataError, match="high must be"):
        validate_ohlcv(valid_candles)


def test_negative_volume_is_rejected(valid_candles):
    valid_candles.loc[0, "volume"] = -1
    with pytest.raises(MarketDataError, match="Volume cannot be negative"):
        validate_ohlcv(valid_candles)


def test_non_finite_prices_are_rejected(valid_candles):
    valid_candles.loc[0, "close"] = float("inf")
    with pytest.raises(MarketDataError, match="NaN or infinite"):
        validate_ohlcv(valid_candles)


def test_empty_frame_is_rejected():
    with pytest.raises(MarketDataError, match="no rows"):
        validate_ohlcv(pd.DataFrame())


def test_csv_loader_and_summary(tmp_path: Path, valid_candles):
    csv_path = tmp_path / "candles.csv"
    valid_candles.to_csv(csv_path, index=False)
    loaded = load_ohlcv_csv(csv_path)
    summary = describe_ohlcv(loaded)
    assert summary["rows"] == 2
    assert summary["duplicate_timestamps"] == 0
    assert summary["missing_intervals_checked"] is False


def test_missing_csv_raises_file_not_found(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_ohlcv_csv(tmp_path / "missing.csv")
