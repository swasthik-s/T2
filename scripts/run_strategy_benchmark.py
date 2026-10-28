"""Compare independent T2 strategy families under one deterministic backtest engine.

Usage: python scripts/run_strategy_benchmark.py [normalized-market-data.csv]
Random Forest is an optional CPU baseline. VWAP runs only when real source
volume exists. This script never sends orders and never promotes a strategy.
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
from t2.improvement.gpu_ml import FEATURE_COLUMNS, build_causal_features
from t2.improvement.target_labels import make_target_before_stop_labels
from t2.strategies.research_families import ema_crossover, orb_breakout, vwap_ema


def atr_stops(candles: pd.DataFrame, period: int = 14, multiple: float = 1.5) -> pd.Series:
    previous = candles["close"].shift(1)
    tr = pd.concat([
        candles["high"] - candles["low"],
        (candles["high"] - previous).abs(),
        (candles["low"] - previous).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=period).mean()
    fallback = (candles["high"] - candles["low"]).rolling(period, min_periods=1).mean()
    return (atr.where(atr > 0, fallback).clip(lower=1e-8) * multiple).clip(lower=1e-8)


def evaluate(candles: pd.DataFrame, signals: pd.Series, stops: pd.Series, slippage: float = 0.02) -> dict:
    config = BacktestConfig(
        initial_equity=10_000.0,
        risk_per_trade_pct=0.25,
        reward_risk_ratio=2.0,
        commission_pct_per_side=0.0,
        slippage_pct_per_side=slippage,
        max_trades=10_000,
    )
    result = run_backtest(
        candles.reset_index(drop=True),
        signals.fillna(False).astype(bool).reset_index(drop=True),
        stops.reset_index(drop=True),
        config=config,
        side="long",
    )
    return summarize_backtest(result)


def choose_threshold(candles, probabilities, stops, start, end):
    rows = []
    for threshold in (0.50, 0.55, 0.60, 0.65, 0.70):
        metrics = evaluate(
            candles.iloc[start:end],
            probabilities.iloc[start:end] >= threshold,
            stops.iloc[start:end],
        )
        eligible = (
            metrics["trades"] >= 30
            and np.isfinite(metrics["profit_factor"])
            and metrics["profit_factor"] > 1.0
            and metrics["expectancy_per_trade"] > 0
        )
        rows.append({
            "threshold": threshold,
            "eligible": bool(eligible),
            "trades": metrics["trades"],
            "net_pnl": metrics["net_pnl"],
            "profit_factor": metrics["profit_factor"],
            "expectancy_per_trade": metrics["expectancy_per_trade"],
            "max_closed_trade_drawdown_pct": metrics["max_closed_trade_drawdown_pct"],
        })
    rows.sort(key=lambda row: (
        int(row["eligible"]), row["expectancy_per_trade"],
        -row["max_closed_trade_drawdown_pct"], row["trades"],
    ), reverse=True)
    return rows[0], rows


def fit_random_forest(features, labels, train_end):
    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError as exc:
        raise RuntimeError("Random Forest needs scikit-learn. Install with: pip install scikit-learn") from exc
    train_x = features.iloc[:train_end]
    train_y = labels.iloc[:train_end]
    valid = train_x.notna().all(axis=1) & train_y.notna()
    x = train_x.loc[valid].to_numpy(dtype=np.float32)
    y = train_y.loc[valid].to_numpy(dtype=np.int32)
    if len(x) < 100 or len(np.unique(y)) < 2:
        raise ValueError("Insufficient target-aware labels for Random Forest training")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=50,
        class_weight="balanced_subsample",
        random_state=17,
        n_jobs=-1,
    )
    model.fit(x, y)
    all_x = features.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    valid_all = np.isfinite(all_x).all(axis=1)
    probability = np.full(len(features), np.nan, dtype=float)
    positive_class = list(model.classes_).index(1)
    probability[valid_all] = model.predict_proba(all_x[valid_all])[:, positive_class]
    return model, pd.Series(probability, index=features.index, name="target_probability")


def safe_metrics(metrics):
    return {
        key: (None if isinstance(value, (float, np.floating)) and not np.isfinite(value) else value)
        for key, value in metrics.items()
        if key not in ("long_short",)
    }


def json_safe(value):
    """Convert NumPy values and non-finite metrics to strict JSON-safe values."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def main() -> int:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        ROOT / "data/historical/xauusd_m5_2024-10-28_to_2026-10-28.csv"
    )
    candles = load_ohlcv_csv(source).reset_index(drop=True)
    n = len(candles)
    train_end, val_end = int(n * 0.60), int(n * 0.80)
    stops = atr_stops(candles)
    candidates: dict[str, dict] = {}
    skips: dict[str, str] = {}

    ema = ema_crossover(candles)
    orb = orb_breakout(candles)
    candidates["ema_crossover_long"] = {"signals": ema["long_signal"], "stops": ema["stop_distance"]}
    candidates["ema_crossover_short"] = {"signals": ema["short_signal"], "stops": ema["stop_distance"]}
    candidates["orb_long"] = {"signals": orb["long_signal"], "stops": orb["stop_distance"]}
    candidates["orb_short"] = {"signals": orb["short_signal"], "stops": orb["stop_distance"]}

    if "volume" in candles.columns and (candles["volume"] > 0).any():
        vwap = vwap_ema(candles)
        candidates["vwap_ema_long"] = {"signals": vwap["long_signal"], "stops": vwap["stop_distance"]}
        candidates["vwap_ema_short"] = {"signals": vwap["short_signal"], "stops": vwap["stop_distance"]}
    else:
        skips["vwap_ema"] = "Skipped: source data has no genuine volume column."

    results = {}
    for name, candidate in candidates.items():
        validation = evaluate(
            candles.iloc[train_end:val_end],
            candidate["signals"].iloc[train_end:val_end],
            candidate["stops"].iloc[train_end:val_end],
        )
        test = evaluate(
            candles.iloc[val_end:],
            candidate["signals"].iloc[val_end:],
            candidate["stops"].iloc[val_end:],
        )
        eligible = (
            validation["trades"] >= 30
            and np.isfinite(validation["profit_factor"])
            and validation["profit_factor"] > 1.0
            and validation["expectancy_per_trade"] > 0
        )
        results[name] = {
            "family": "rule_based",
            "validation": safe_metrics(validation),
            "validation_gate_passed": bool(eligible),
            "test_at_0.02pct_slippage_per_side": safe_metrics(test),
            "test_used_for_selection": False,
        }

    # Random Forest uses target-before-stop labels and a threshold chosen on validation only.
    features = build_causal_features(candles).loc[:, FEATURE_COLUMNS]
    labels = make_target_before_stop_labels(
        candles, stops, horizon=12, reward_risk_ratio=2.0,
    )
    # Purge labels whose forward outcomes can cross the train/validation boundary.
    labels.iloc[max(0, train_end - 13):train_end] = np.nan
    labels.iloc[max(0, val_end - 13):val_end] = np.nan
    try:
        model, probabilities = fit_random_forest(features, labels, train_end)
        selected, threshold_rows = choose_threshold(
            candles, probabilities, stops, train_end, val_end,
        )
        test_metrics = evaluate(
            candles.iloc[val_end:],
            probabilities.iloc[val_end:] >= selected["threshold"],
            stops.iloc[val_end:],
        )
        results["random_forest_target_before_stop"] = {
            "family": "supervised_ml",
            "label": "target-before-stop within 12 bars; ambiguous bar is negative",
            "model": "RandomForestClassifier(n_estimators=300,max_depth=8,min_samples_leaf=50)",
            "selected_threshold": selected["threshold"],
            "validation_thresholds": threshold_rows,
            "validation_gate_passed": selected["eligible"],
            "test_at_0.02pct_slippage_per_side": safe_metrics(test_metrics),
            "test_used_for_selection": False,
        }
    except (RuntimeError, ValueError) as exc:
        skips["random_forest_target_before_stop"] = str(exc)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ROOT / "reports/strategy_benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "run_id": run_id,
        "source_file": source.name,
        "rows": n,
        "train_rows": train_end,
        "validation_rows": val_end - train_end,
        "test_rows": n - val_end,
        "split": "chronological 60/20/20; threshold selection uses validation only",
        "cost_model": "0.02% slippage per side, 0% commission; provisional and not broker-calibrated",
        "initial_equity": 10000.0,
        "risk_per_trade_pct": 0.25,
        "reward_risk_ratio": 2.0,
        "results": results,
        "skipped": skips,
        "guardrails": {
            "live_trading_enabled": False,
            "allow_order_submission": False,
            "automatic_promotion": False,
            "test_set_may_select_candidates": False,
        },
        "limitations": [
            "Rule-based families are evaluated long/short separately, not as a combined portfolio.",
            "Test split is for one final diagnostic only; do not tune strategies against it.",
            "Costs are provisional and data may be bid-only; no broker-specific spread/commission is modeled.",
            "ORB uses a configurable generic UTC session, not a verified venue-specific calendar.",
            "VWAP requires genuine source volume and is skipped when volume is absent.",
            "Random Forest is CPU-based; the existing neural network remains the GPU training experiment.",
            "Backtest uses a simplified OHLC stop/target fill model and closed-trade drawdown.",
        ],
    }
    json_path = out_dir / f"summary_{run_id}.json"
    json_path.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False), encoding="utf-8")
    rows = []
    for name, result in results.items():
        validation = result["validation"] if "validation" in result else next(
            (r for r in result.get("validation_thresholds", []) if r["threshold"] == result.get("selected_threshold")), {}
        )
        test = result["test_at_0.02pct_slippage_per_side"]
        rows.append({
            "strategy": name,
            "validation_gate_passed": result["validation_gate_passed"],
            "validation_trades": validation.get("trades"),
            "validation_pnl": validation.get("net_pnl"),
            "validation_profit_factor": validation.get("profit_factor"),
            "test_trades": test.get("trades"),
            "test_pnl": test.get("net_pnl"),
            "test_profit_factor": test.get("profit_factor"),
            "test_expectancy": test.get("expectancy_per_trade"),
        })
    pd.DataFrame(rows).to_csv(out_dir / f"leaderboard_{run_id}.csv", index=False)
    print(f"Device: CPU for rule-based backtests and Random Forest; rows={n:,}")
    print("Strategy | Validation gate | Test trades | Test PnL | Test PF")
    for row in rows:
        print(
            f"{row['strategy']} | {row['validation_gate_passed']} | "
            f"{row['test_trades']} | {row['test_pnl']} | {row['test_profit_factor']}"
        )
    if skips:
        print("Skipped:", json.dumps(skips))
    print(f"Report: {json_path}")
    print("No orders were submitted; no strategy was promoted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
