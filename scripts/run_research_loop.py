"""Bounded autonomous T2 research loop (research only; no broker or order access).

Ranks a fixed candidate grid on train/validation only, then evaluates only the
validation-selected candidate for each independent direction on the held-out
test segment. This is not a combined long/short portfolio.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

# Make "python scripts/run_research_loop.py" work from any current directory.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from t2.backtesting.engine import BacktestConfig, run_backtest
from t2.backtesting.evaluation import save_evaluation_report, summarize_backtest
from t2.data.ohlcv import load_ohlcv_csv
from t2.improvement.research_loop import (
    candidate_score,
    chronological_three_way_split,
    generate_candidates,
)
from t2.strategies.ema_atr_baseline import build_ema_atr_signals


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _config(slippage_pct_per_side: float) -> BacktestConfig:
    return BacktestConfig(
        initial_equity=10_000.0,
        risk_per_trade_pct=0.25,
        reward_risk_ratio=2.0,
        commission_pct_per_side=0.0,
        slippage_pct_per_side=slippage_pct_per_side,
        max_trades=10_000,
    )


def _evaluate(
    candles: pd.DataFrame,
    signals: pd.DataFrame,
    signal_column: str,
    side: str,
    slippage_pct_per_side: float,
) -> dict[str, Any]:
    raw = run_backtest(
        candles.reset_index(drop=True),
        signals[signal_column].astype(bool).reset_index(drop=True),
        signals["stop_distance"].reset_index(drop=True),
        config=_config(slippage_pct_per_side),
        side=side,
    )
    return summarize_backtest(raw)


def main() -> int:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else (
        REPO_ROOT / "data/historical/xauusd_m5_2024-10-28_to_2026-10-28.csv"
    )
    if not source.is_file():
        raise FileNotFoundError(
            f"Historical CSV not found: {source}. Pass its path as the first argument."
        )

    candles = load_ohlcv_csv(source)
    if "volume" in candles.columns:
        candles = candles.drop(columns=["volume"])
    candles = candles.reset_index(drop=True)
    segments = chronological_three_way_split(candles)
    signals_cache: dict[tuple[int, int, float], pd.DataFrame] = {}

    output_root = REPO_ROOT / "reports/autonomous_research"
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()

    segment_frames = {
        name: candles.iloc[start:end].reset_index(drop=True)
        for name, (start, end) in segments.items()
    }

    leaderboard: list[dict[str, Any]] = []
    selected: dict[str, dict[str, Any]] = {}
    candidates = generate_candidates()
    validation_cost = 0.02  # provisional, deliberately unchanged across candidates

    print(f"Research run: {run_id}")
    print(f"Candles: {len(candles):,} | Candidates: {len(candidates)}")
    print("Stage 1/3: evaluating candidates on train and validation only.")

    for candidate_index, candidate in enumerate(candidates, start=1):
        key = (
            int(candidate["fast_ema"]),
            int(candidate["slow_ema"]),
            float(candidate["atr_stop_multiple"]),
        )
        strategy_signals = build_ema_atr_signals(
            candles,
            fast_ema=key[0],
            slow_ema=key[1],
            atr_period=int(candidate["atr_period"]),
            atr_stop_multiple=key[2],
        )
        signals_cache[key] = strategy_signals

        for side, signal_column in (("long", "long_signal"), ("short", "short_signal")):
            row: dict[str, Any] = {
                "candidate_id": f"ema{key[0]}_{key[1]}_atr{key[2]:g}_{side}",
                **candidate,
                "side": side,
                "validation_slippage_pct_per_side": validation_cost,
            }
            for segment_name in ("train", "validation"):
                start, end = segments[segment_name]
                segment_signals = strategy_signals.iloc[start:end].reset_index(drop=True)
                metrics = _evaluate(
                    segment_frames[segment_name],
                    segment_signals,
                    signal_column,
                    side,
                    validation_cost,
                )
                for metric in (
                    "trades", "net_pnl", "return_pct_closed_trades",
                    "win_rate_pct", "profit_factor", "expectancy_per_trade",
                    "max_closed_trade_drawdown_pct", "total_costs",
                ):
                    row[f"{segment_name}_{metric}"] = metrics[metric]
            row["validation_score"] = list(candidate_score({
                "trades": row["validation_trades"],
                "profit_factor": row["validation_profit_factor"],
                "expectancy_per_trade": row["validation_expectancy_per_trade"],
                "max_closed_trade_drawdown_pct": row["validation_max_closed_trade_drawdown_pct"],
            }))
            row["validation_gate_passed"] = bool(row["validation_score"][0] == 1.0)
            leaderboard.append(row)

        if candidate_index % 3 == 0 or candidate_index == len(candidates):
            print(f"  Completed {candidate_index}/{len(candidates)} candidates")

    leaderboard.sort(
        key=lambda row: tuple(row["validation_score"]),
        reverse=True,
    )
    leaderboard_path = output_root / f"leaderboard_{run_id}.csv"
    pd.DataFrame([
        {key: value for key, value in row.items() if key != "validation_score"}
        | {"validation_score": str(row["validation_score"])}
        for row in leaderboard
    ]).to_csv(leaderboard_path, index=False)

    # Choose the highest validation-ranked candidate separately per direction.
    # No test-segment result is used for selection.
    for side in ("long", "short"):
        side_rows = [row for row in leaderboard if row["side"] == side]
        selected[side] = side_rows[0]

    print("Stage 2/3: evaluating only the validation-selected candidates on held-out test data.")
    test_costs = (0.0, 0.005, 0.01, 0.02)
    test_results: dict[str, Any] = {}
    for side, row in selected.items():
        candidate = {
            "fast_ema": int(row["fast_ema"]),
            "slow_ema": int(row["slow_ema"]),
            "atr_period": int(row["atr_period"]),
            "atr_stop_multiple": float(row["atr_stop_multiple"]),
            "reward_risk_ratio": float(row["reward_risk_ratio"]),
        }
        key = (
            candidate["fast_ema"],
            candidate["slow_ema"],
            candidate["atr_stop_multiple"],
        )
        strategy_signals = signals_cache[key]
        start, end = segments["test"]
        test_signals = strategy_signals.iloc[start:end].reset_index(drop=True)
        test_candles = segment_frames["test"]
        signal_column = "long_signal" if side == "long" else "short_signal"
        test_results[side] = {
            "selected_candidate": row["candidate_id"],
            "parameters": candidate,
            "validation_gate_passed": row["validation_gate_passed"],
            "validation_metrics": {
                metric: row[f"validation_{metric}"]
                for metric in (
                    "trades", "net_pnl", "return_pct_closed_trades", "win_rate_pct",
                    "profit_factor", "expectancy_per_trade",
                    "max_closed_trade_drawdown_pct", "total_costs",
                )
            },
            "test_cost_sensitivity": {},
        }
        for cost in test_costs:
            metrics = _evaluate(
                test_candles, test_signals, signal_column, side, cost
            )
            test_results[side]["test_cost_sensitivity"][str(cost)] = metrics

        # Save the highest-cost test report's full trade/equity files.
        raw = run_backtest(
            test_candles,
            test_signals[signal_column].astype(bool),
            test_signals["stop_distance"],
            config=_config(0.02),
            side=side,
        )
        save_evaluation_report(
            raw, str(output_root / run_id / side / "test_slippage_0.02")
        )

    print("Stage 3/3: saving durable experiment memory.")
    summary = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": source.name,
        "source_sha256": source_hash,
        "rows": len(candles),
        "split_indices": segments,
        "split_timestamps_utc": {
            name: {
                "start": segment_frames[name]["timestamp"].iloc[0].isoformat(),
                "end": segment_frames[name]["timestamp"].iloc[-1].isoformat(),
                "rows": len(segment_frames[name]),
            }
            for name in ("train", "validation", "test")
        },
        "candidate_count": len(candidates),
        "selection_rule": (
            "Validation only: at least 30 trades, finite profit factor > 1, "
            "positive expectancy; then rank by expectancy, lower closed-trade "
            "drawdown, and trade count. If none qualify, best failed candidate "
            "is still tested for diagnosis and explicitly marked as ineligible."
        ),
        "validation_slippage_pct_per_side": validation_cost,
        "test_slippage_sensitivity_pct_per_side": list(test_costs),
        "commission_pct_per_side": 0.0,
        "vwap_used": False,
        "volume_used": False,
        "paper_trading_only": True,
        "live_orders_submitted": False,
        "auto_promotion_enabled": False,
        "directional_runs_are_separate": True,
        "test_results": test_results,
        "leaderboard_csv": str(leaderboard_path.relative_to(REPO_ROOT)),
        "limitations": [
            "Bid OHLC only; ask/spread and volume are unavailable.",
            "The percentage cost model is provisional and not broker-calibrated.",
            "Long and short results are independent diagnostics, not a combined portfolio.",
            "The test segment is evaluated only after validation-based selection, but future repeated research cycles can still overfit if this same test is reused repeatedly.",
            "No strategy is automatically promoted or connected to order submission.",
            "This pandas/event-driven backtest is CPU-bound; it does not currently use a GPU.",
        ],
    }
    safe_summary = _json_safe(summary)
    summary_path = output_root / f"summary_{run_id}.json"
    summary_path.write_text(
        json.dumps(safe_summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    memory_path = output_root / "experiment_memory.jsonl"
    with memory_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(safe_summary, allow_nan=False) + "\n")

    print("\nSelected candidates (chosen using validation only):")
    for side, result in test_results.items():
        validation = result["validation_metrics"]
        print(
            f"  {side}: {result['selected_candidate']} | "
            f"validation PF={validation['profit_factor']:.3f}, "
            f"validation expectancy={validation['expectancy_per_trade']:.3f}, "
            f"validation gate passed={result['validation_gate_passed']}"
        )
        for cost, metrics in result["test_cost_sensitivity"].items():
            print(
                f"    test slippage/side={cost}%: trades={metrics['trades']}, "
                f"PnL={metrics['net_pnl']:.2f}, PF={metrics['profit_factor']}, "
                f"expectancy={metrics['expectancy_per_trade']:.3f}, "
                f"DD={metrics['max_closed_trade_drawdown_pct']:.2f}%"
            )
    print(f"\nLeaderboard: {leaderboard_path}")
    print(f"Run summary: {summary_path}")
    print(f"Persistent experiment memory: {memory_path}")
    print("Research loop complete. No live orders were submitted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
