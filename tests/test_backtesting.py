import pandas as pd
import pytest

from t2.backtesting.engine import BacktestConfig, run_backtest


def candles(rows):
    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def test_long_target_hit_has_expected_net_profit():
    data = candles([
        ("2026-01-01T00:00:00Z", 100, 101, 99, 100, 10),
        ("2026-01-01T00:01:00Z", 100, 101, 99, 100, 10),
        ("2026-01-01T00:02:00Z", 100, 103, 100, 102, 10),
    ])
    result = run_backtest(data, pd.Series([True, False, False]), 1.0,
                          BacktestConfig(initial_equity=1000, risk_per_trade_pct=1, reward_risk_ratio=2))
    assert result["trades"] == 1
    assert result["wins"] == 1
    assert result["trade_log"][0]["entry_price"] == 100
    assert result["trade_log"][0]["exit_price"] == 102
    assert result["net_pnl"] == pytest.approx(20)


def test_stop_wins_when_stop_and_target_hit_same_candle():
    data = candles([
        ("2026-01-01T00:00:00Z", 100, 101, 99, 100, 10),
        ("2026-01-01T00:01:00Z", 100, 101, 99, 100, 10),
        ("2026-01-01T00:02:00Z", 100, 103, 98, 100, 10),
    ])
    result = run_backtest(data, pd.Series([True, False, False]), 1.0,
                          BacktestConfig(initial_equity=1000, risk_per_trade_pct=1, reward_risk_ratio=2))
    assert result["trades"] == 1
    assert result["losses"] == 1
    assert result["trade_log"][0]["exit_price"] == 99
    assert result["net_pnl"] == pytest.approx(-10)


def test_entry_occurs_on_next_candle_open():
    data = candles([
        ("2026-01-01T00:00:00Z", 100, 101, 99, 100, 10),
        ("2026-01-01T00:01:00Z", 105, 106, 104, 105, 10),
        ("2026-01-01T00:02:00Z", 105, 106, 104, 105, 10),
    ])
    result = run_backtest(data, pd.Series([True, False, False]), 1.0,
                          BacktestConfig(initial_equity=1000, risk_per_trade_pct=1))
    assert result["trade_log"][0]["entry_price"] == 105


def test_commission_reduces_net_profit():
    data = candles([
        ("2026-01-01T00:00:00Z", 100, 101, 99, 100, 10),
        ("2026-01-01T00:01:00Z", 100, 101, 99, 100, 10),
        ("2026-01-01T00:02:00Z", 100, 103, 100, 102, 10),
    ])
    free = run_backtest(data, [True, False, False], 1.0,
                        BacktestConfig(initial_equity=1000, risk_per_trade_pct=1))
    charged = run_backtest(data, [True, False, False], 1.0,
                           BacktestConfig(initial_equity=1000, risk_per_trade_pct=1, commission_pct_per_side=0.1))
    assert charged["net_pnl"] < free["net_pnl"]


def test_invalid_stop_distance_is_rejected():
    data = candles([
        ("2026-01-01T00:00:00Z", 100, 101, 99, 100, 10),
        ("2026-01-01T00:01:00Z", 100, 101, 99, 100, 10),
    ])
    with pytest.raises(ValueError, match="stop_distance"):
        run_backtest(data, [True, False], 0)


def test_short_target_hit_has_expected_net_profit():
    data = candles([
        ("2026-01-01T00:00:00Z", 100, 101, 99, 100, 10),
        ("2026-01-01T00:01:00Z", 100, 101, 99, 100, 10),
        ("2026-01-01T00:02:00Z", 100, 100, 97, 98, 10),
    ])
    result = run_backtest(data, [True, False, False], 1.0,
                          BacktestConfig(initial_equity=1000, risk_per_trade_pct=1,
                                         reward_risk_ratio=2), side="short")
    assert result["trades"] == 1
    assert result["wins"] == 1
    assert result["trade_log"][0]["exit_price"] == 98
    assert result["net_pnl"] == pytest.approx(20)


def test_short_stop_hit_has_expected_loss():
    data = candles([
        ("2026-01-01T00:00:00Z", 100, 101, 99, 100, 10),
        ("2026-01-01T00:01:00Z", 100, 101, 99, 100, 10),
        ("2026-01-01T00:02:00Z", 102, 103, 101, 102, 10),
    ])
    result = run_backtest(data, [True, False, False], 1.0,
                          BacktestConfig(initial_equity=1000, risk_per_trade_pct=1), side="short")
    assert result["trades"] == 1
    assert result["losses"] == 1
    assert result["trade_log"][0]["exit_price"] == 101
    assert result["net_pnl"] == pytest.approx(-10)


def test_open_position_at_end_is_reported_not_closed():
    # Keep all candles strictly between the stop (99) and target (102).
    data = candles([
        ("2026-01-01T00:00:00Z", 100, 100.5, 99.5, 100, 10),
        ("2026-01-01T00:01:00Z", 100, 100.5, 99.5, 100, 10),
        ("2026-01-01T00:02:00Z", 100, 100.5, 99.5, 100, 10),
    ])
    result = run_backtest(data, [True, False, False], 1.0)
    assert result["trades"] == 0
    assert result["open_position_at_end_ignored"] is True
    assert result["final_equity"] == result["initial_equity"]


@pytest.mark.parametrize("kwargs", [
    {"initial_equity": float("nan")},
    {"initial_equity": float("inf")},
    {"risk_per_trade_pct": float("nan")},
    {"reward_risk_ratio": float("inf")},
    {"commission_pct_per_side": float("nan")},
    {"slippage_pct_per_side": float("inf")},
    {"max_trades": True},
])
def test_config_rejects_non_finite_or_invalid_values(kwargs):
    with pytest.raises(ValueError):
        BacktestConfig(**kwargs)


def test_stop_distance_series_with_wrong_length_is_rejected():
    data = candles([
        ("2026-01-01T00:00:00Z", 100, 101, 99, 100, 10),
        ("2026-01-01T00:01:00Z", 100, 101, 99, 100, 10),
    ])
    with pytest.raises(ValueError, match="stop_distance"):
        run_backtest(data, [True, False], pd.Series([1.0]))


def test_non_numeric_stop_distance_is_rejected_cleanly():
    data = candles([
        ("2026-01-01T00:00:00Z", 100, 101, 99, 100, 10),
        ("2026-01-01T00:01:00Z", 100, 101, 99, 100, 10),
    ])
    with pytest.raises(ValueError, match="stop_distance"):
        run_backtest(data, [True, False], "not-a-number")
