import pandas as pd
import pytest

from t2.backtesting.evaluation import (
    build_equity_curve,
    chronological_split,
    monthly_performance,
    summarize_backtest,
)


def sample_backtest():
    return {
        "initial_equity": 1000.0,
        "open_position_at_end_ignored": False,
        "trade_log": [
            {"side": "long", "entry_time": "2026-01-02T00:00:00Z", "exit_time": "2026-01-02T00:02:00Z",
             "gross_pnl": 20.0, "costs": 1.0, "net_pnl": 19.0},
            {"side": "short", "entry_time": "2026-01-03T00:00:00Z", "exit_time": "2026-01-03T00:02:00Z",
             "gross_pnl": -10.0, "costs": 0.5, "net_pnl": -10.5},
            {"side": "long", "entry_time": "2026-02-03T00:00:00Z", "exit_time": "2026-02-03T00:02:00Z",
             "gross_pnl": 40.0, "costs": 1.0, "net_pnl": 39.0},
        ],
    }


def test_summary_calculates_headline_metrics_and_costs():
    result = summarize_backtest(sample_backtest())
    assert result["trades"] == 3
    assert result["wins"] == 2
    assert result["losses"] == 1
    assert result["net_pnl"] == pytest.approx(47.5)
    assert result["final_equity_closed_trades"] == pytest.approx(1047.5)
    assert result["total_costs"] == pytest.approx(2.5)
    assert result["win_rate_pct"] == pytest.approx(200 / 3)
    assert result["profit_factor"] == pytest.approx(58 / 10.5)
    assert result["expectancy_per_trade"] == pytest.approx(47.5 / 3)


def test_equity_curve_tracks_peak_and_drawdown():
    curve = build_equity_curve(sample_backtest())
    assert curve["equity"].tolist() == pytest.approx([1019.0, 1008.5, 1047.5])
    assert curve["peak_equity"].tolist() == pytest.approx([1019.0, 1019.0, 1047.5])
    assert curve["drawdown"].tolist() == pytest.approx([0.0, 10.5, 0.0])
    assert curve["drawdown_pct"].iloc[1] == pytest.approx(10.5 / 1019 * 100)


def test_summary_splits_long_and_short_performance():
    result = summarize_backtest(sample_backtest())
    assert result["long_short"]["long"]["trades"] == 2
    assert result["long_short"]["long"]["net_pnl"] == pytest.approx(58)
    assert result["long_short"]["short"]["trades"] == 1
    assert result["long_short"]["short"]["net_pnl"] == pytest.approx(-10.5)


def test_monthly_performance_uses_exit_month_in_utc():
    monthly = monthly_performance(sample_backtest())
    assert monthly["month"].tolist() == ["2026-01", "2026-02"]
    assert monthly["trades"].tolist() == [2, 1]
    assert monthly["net_pnl"].tolist() == pytest.approx([8.5, 39.0])
    assert monthly["total_costs"].tolist() == pytest.approx([1.5, 1.0])


def test_empty_trade_log_returns_zero_summary_and_empty_curve():
    data = {"initial_equity": 1000.0, "trade_log": []}
    summary = summarize_backtest(data)
    curve = build_equity_curve(data)
    monthly = monthly_performance(data)
    assert summary["trades"] == 0
    assert summary["net_pnl"] == 0
    assert summary["profit_factor"] == 0
    assert curve.empty
    assert monthly.empty


def test_chronological_split_preserves_order_and_no_overlap():
    candles = pd.DataFrame([
        ("2026-01-01T00:00:00Z", 100, 101, 99, 100, 10),
        ("2026-01-01T00:01:00Z", 101, 102, 100, 101, 10),
        ("2026-01-01T00:02:00Z", 102, 103, 101, 102, 10),
        ("2026-01-01T00:03:00Z", 103, 104, 102, 103, 10),
        ("2026-01-01T00:04:00Z", 104, 105, 103, 104, 10),
    ], columns=["timestamp", "open", "high", "low", "close", "volume"])
    train, test = chronological_split(candles, 0.6)
    assert len(train) == 3
    assert len(test) == 2
    assert train["timestamp"].max() < test["timestamp"].min()
    assert train["close"].tolist() == [100, 101, 102]
    assert test["close"].tolist() == [103, 104]


@pytest.mark.parametrize("fraction", [0.0, 0.49, 1.0, float("nan"), float("inf")])
def test_chronological_split_rejects_invalid_fraction(fraction):
    candles = pd.DataFrame([
        ("2026-01-01T00:00:00Z", 100, 101, 99, 100, 10),
        ("2026-01-01T00:01:00Z", 101, 102, 100, 101, 10),
    ], columns=["timestamp", "open", "high", "low", "close", "volume"])
    with pytest.raises(ValueError, match="train_fraction"):
        chronological_split(candles, fraction)


def test_summary_rejects_missing_trade_fields():
    with pytest.raises(ValueError, match="missing required fields"):
        summarize_backtest({"initial_equity": 1000, "trade_log": [{"net_pnl": 1}]})


def test_summary_rejects_non_finite_pnl():
    trade = sample_backtest()
    trade["trade_log"][0]["net_pnl"] = float("nan")
    with pytest.raises(ValueError, match="finite values"):
        summarize_backtest(trade)


def test_save_evaluation_report_writes_metrics_and_csv_files(tmp_path):
    import json

    from t2.backtesting.evaluation import save_evaluation_report

    paths = save_evaluation_report(sample_backtest(), str(tmp_path))
    assert set(paths) == {"metrics_json", "trades_csv", "monthly_csv", "equity_curve_csv"}
    assert all(__import__("pathlib").Path(path).is_file() for path in paths.values())

    metrics = json.loads(__import__("pathlib").Path(paths["metrics_json"]).read_text(encoding="utf-8"))
    assert metrics["trades"] == 3
    assert metrics["net_pnl"] == pytest.approx(47.5)

    trades = pd.read_csv(paths["trades_csv"])
    monthly = pd.read_csv(paths["monthly_csv"])
    equity = pd.read_csv(paths["equity_curve_csv"])
    assert len(trades) == 3
    assert monthly["month"].tolist() == ["2026-01", "2026-02"]
    assert equity["equity"].iloc[-1] == pytest.approx(1047.5)


def test_save_evaluation_report_converts_infinite_profit_factor_to_json_null(tmp_path):
    import json

    from t2.backtesting.evaluation import save_evaluation_report

    backtest = {
        "initial_equity": 1000.0,
        "trade_log": [
            {"side": "long", "entry_time": "2026-01-01T00:00:00Z", "exit_time": "2026-01-01T00:01:00Z",
             "gross_pnl": 10.0, "costs": 0.0, "net_pnl": 10.0}
        ],
    }
    paths = save_evaluation_report(backtest, str(tmp_path))
    metrics = json.loads(__import__("pathlib").Path(paths["metrics_json"]).read_text(encoding="utf-8"))
    assert metrics["profit_factor"] is None
