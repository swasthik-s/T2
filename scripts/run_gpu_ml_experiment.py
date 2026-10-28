"""Run a GPU-capable ML direction experiment with validation-only threshold selection.

Usage:
  python scripts/run_gpu_ml_experiment.py [path-to-normalized-csv]
PyTorch is optional until this script is run. No live execution is possible here.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from t2.backtesting.engine import BacktestConfig, run_backtest
from t2.backtesting.evaluation import summarize_backtest
from t2.data.ohlcv import load_ohlcv_csv
from t2.improvement.gpu_ml import (
    build_causal_features, fit_direction_mlp, make_direction_labels,
    predict_probabilities, select_device,
)


def atr_stop_distance(candles: pd.DataFrame, period: int = 14, multiple: float = 1.5) -> pd.Series:
    high, low, close = candles["high"], candles["low"], candles["close"]
    previous = close.shift(1)
    tr = pd.concat([high-low, (high-previous).abs(), (low-previous).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=period).mean()
    fallback = (high-low).rolling(period, min_periods=1).mean().clip(lower=1e-8)
    return (atr.where(atr > 0, fallback) * multiple).clip(lower=1e-8)


def evaluate(candles: pd.DataFrame, signals: pd.Series, stops: pd.Series, slippage: float) -> dict:
    cfg = BacktestConfig(
        initial_equity=10_000.0, risk_per_trade_pct=0.25,
        reward_risk_ratio=2.0, commission_pct_per_side=0.0,
        slippage_pct_per_side=slippage, max_trades=10_000,
    )
    raw = run_backtest(
        candles.reset_index(drop=True),
        signals.astype(bool).reset_index(drop=True),
        stops.reset_index(drop=True), config=cfg, side="long",
    )
    return summarize_backtest(raw)


def main() -> int:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        ROOT / "data/historical/xauusd_m5_2024-10-28_to_2026-10-28.csv"
    )
    candles = load_ohlcv_csv(source).reset_index(drop=True)
    # This dataset is bid OHLC only. Do not fabricate volume or claim VWAP support.
    n = len(candles)
    train_end, val_end = int(n * 0.60), int(n * 0.80)
    features = build_causal_features(candles)
    horizon = 3
    labels = make_direction_labels(candles, horizon=horizon)
    # Purge labels whose forward horizon crosses the training boundary.
    labels.iloc[max(0, train_end-horizon):train_end] = np.nan
    fit = fit_direction_mlp(features, labels, train_end, epochs=25, prefer_gpu=True)
    probabilities = predict_probabilities(fit, features)
    stops = atr_stop_distance(candles)
    validation_candles = candles.iloc[train_end:val_end].reset_index(drop=True)
    test_candles = candles.iloc[val_end:].reset_index(drop=True)
    validation_stops = stops.iloc[train_end:val_end].reset_index(drop=True)
    test_stops = stops.iloc[val_end:].reset_index(drop=True)

    thresholds = (0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65)
    rows = []
    for threshold in thresholds:
        validation_signal = (probabilities.iloc[train_end:val_end] >= threshold).fillna(False)
        metrics = evaluate(validation_candles, validation_signal, validation_stops, 0.02)
        eligible = (
            metrics["trades"] >= 30
            and np.isfinite(metrics["profit_factor"])
            and metrics["profit_factor"] > 1.0
            and metrics["expectancy_per_trade"] > 0
        )
        rows.append({
            "threshold": threshold, "eligible": bool(eligible),
            "trades": metrics["trades"], "net_pnl": metrics["net_pnl"],
            "profit_factor": metrics["profit_factor"],
            "expectancy_per_trade": metrics["expectancy_per_trade"],
            "max_closed_trade_drawdown_pct": metrics["max_closed_trade_drawdown_pct"],
        })
    rows.sort(key=lambda r: (
        int(r["eligible"]), r["expectancy_per_trade"],
        -r["max_closed_trade_drawdown_pct"], r["trades"]
    ), reverse=True)
    chosen = rows[0]
    test_signal = (probabilities.iloc[val_end:] >= chosen["threshold"]).fillna(False)
    test_costs = {}
    for cost in (0.0, 0.005, 0.01, 0.02):
        test_costs[str(cost)] = evaluate(test_candles, test_signal, test_stops, cost)

    out_dir = ROOT / "reports/gpu_ml_research"
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    result = {
        "run_id": run_id,
        "source_file": source.name,
        "rows": n,
        "device": fit.device,
        "epochs": fit.epochs_trained,
        "training_rows": train_end,
        "validation_rows": val_end-train_end,
        "test_rows": n-val_end,
        "target": f"future close direction over {horizon} candles (binary; not target-hit probability)",
        "features": list(features.columns),
        "selection": "threshold chosen on validation only; test not used for threshold selection",
        "validation_thresholds": rows,
        "selected_threshold": chosen["threshold"],
        "validation_gate_passed": chosen["eligible"],
        "test_cost_sensitivity": test_costs,
        "limitations": [
            "This is an experimental ML direction classifier, not a validated profitable strategy.",
            "The model is trained once on the chronological training segment; it does not self-modify its code.",
            "The bid-only source lacks actual spread/ask and source volume; costs are provisional.",
            "No live or paper broker orders are submitted. Open trades at segment boundaries are not carried.",
            "The label predicts future close direction, not whether a trade hits target before stop.",
        ],
    }
    (out_dir / f"summary_{run_id}.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    pd.DataFrame(rows).to_csv(out_dir / f"validation_thresholds_{run_id}.csv", index=False)
    print(f"Device: {fit.device}")
    print(f"Rows: {n:,} | Train: {train_end:,} | Validation: {val_end-train_end:,} | Test: {n-val_end:,}")
    print(f"Selected threshold: {chosen['threshold']} | validation gate passed={chosen['eligible']}")
    print("Held-out test results:")
    for cost, metrics in test_costs.items():
        print(
            f"  slippage/side={float(cost):.3f}%: trades={metrics['trades']}, "
            f"PnL={metrics['net_pnl']:.2f}, PF={metrics['profit_factor']:.3f}, "
            f"expectancy={metrics['expectancy_per_trade']:.3f}, "
            f"DD={metrics['max_closed_trade_drawdown_pct']:.2f}%"
        )
    print(f"Summary: {out_dir / f'summary_{run_id}.json'}")
    print("No orders were submitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
