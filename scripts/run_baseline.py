"""Run T2's fixed EMA/ATR baseline on historical OHLC data.

Usage:
  python scripts/run_baseline.py data/historical/xauusd_m5_2024-10-28_to_2026-10-28.csv

Writes train/test directional reports under reports/baseline_ema_atr/.
This is a research-only script; it does not connect to a broker or submit orders.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from t2.backtesting.engine import BacktestConfig, run_backtest
from t2.backtesting.evaluation import save_evaluation_report, summarize_backtest
from t2.data.ohlcv import load_ohlcv_csv
from t2.strategies.ema_atr_baseline import build_ema_atr_signals


def main() -> int:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "data/historical/xauusd_m5_2024-10-28_to_2026-10-28.csv"
    )
    if not source.is_file():
        raise FileNotFoundError(
            f"Historical CSV not found: {source}. Pass its path as the first argument."
        )

    candles = load_ohlcv_csv(source)
    if "volume" in candles.columns:
        candles = candles.drop(columns=["volume"])

    signals = build_ema_atr_signals(candles)
    split_at = int(len(candles) * 0.70)
    if split_at < 1 or split_at >= len(candles):
        raise ValueError("Need at least two candles for chronological split")

    segments = {
        "train": (0, split_at),
        "out_of_sample": (split_at, len(candles)),
    }
    root = Path("reports/baseline_ema_atr")
    root.mkdir(parents=True, exist_ok=True)

    # Provisional cost assumption only: 0.02% slippage per side, no explicit
    # commission. This is not broker-calibrated and must be sensitivity-tested.
    config = BacktestConfig(
        initial_equity=10_000.0,
        risk_per_trade_pct=0.25,
        reward_risk_ratio=2.0,
        commission_pct_per_side=0.0,
        slippage_pct_per_side=0.02,
        max_trades=10_000,
    )

    summary = {
        "strategy": "EMA(20/50) crossover, ATR(14) stop x1.5, target 2R",
        "instrument": "XAUUSD",
        "timeframe": "M5",
        "source": source.name,
        "rows": len(candles),
        "train_rows": split_at,
        "out_of_sample_rows": len(candles) - split_at,
        "split_timestamp_utc": candles["timestamp"].iloc[split_at].isoformat(),
        "parameters_fixed_before_evaluation": True,
        "vwap_used": False,
        "volume_used": False,
        "paper_trading_only": True,
        "live_orders_submitted": False,
        "cost_assumptions": {
            "commission_pct_per_side": 0.0,
            "slippage_pct_per_side": 0.02,
            "warning": "Provisional sensitivity assumption, not calibrated to a broker. Bid-only data lacks ask/spread.",
        },
        "directional_runs_are_separate": True,
        "warning": "Long and short reports are independent directional diagnostics; do not add their PnL as one portfolio.",
        "results": {},
    }

    for segment, (start, end) in segments.items():
        part = candles.iloc[start:end].reset_index(drop=True)
        sig = signals.iloc[start:end].reset_index(drop=True)
        summary["results"][segment] = {}
        for side, signal_column in (
            ("long", "long_signal"),
            ("short", "short_signal"),
        ):
            result = run_backtest(
                part,
                sig[signal_column].astype(bool),
                sig["stop_distance"].reset_index(drop=True),
                config=config,
                side=side,
            )
            folder = root / segment / side
            paths = save_evaluation_report(result, str(folder))
            summary["results"][segment][side] = {
                "metrics": summarize_backtest(result),
                "report_files": paths,
            }

    summary_path = root / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False, default=str))
    print(f"\nSaved summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
