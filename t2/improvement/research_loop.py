"""Deterministic, bounded candidate generation and selection for T2 research."""
from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
import pandas as pd


def generate_candidates() -> list[dict[str, float | int]]:
    """Small, fixed search space; never generated from the test-set results."""
    candidates: list[dict[str, float | int]] = []
    for fast, slow, stop_multiple in product(
        (10, 20, 30),
        (50, 100),
        (1.0, 1.5, 2.0),
    ):
        if fast >= slow:
            continue
        candidates.append({
            "fast_ema": fast,
            "slow_ema": slow,
            "atr_period": 14,
            "atr_stop_multiple": stop_multiple,
            "reward_risk_ratio": 2.0,
        })
    return candidates


def chronological_three_way_split(
    candles: pd.DataFrame,
    train_fraction: float = 0.60,
    validation_fraction: float = 0.20,
) -> dict[str, tuple[int, int]]:
    """Return ordered index ranges; remaining rows are the untouched test set."""
    if not np.isfinite(train_fraction) or not np.isfinite(validation_fraction):
        raise ValueError("Split fractions must be finite")
    if train_fraction <= 0 or validation_fraction <= 0 or train_fraction + validation_fraction >= 1:
        raise ValueError("Train and validation fractions must be positive and sum to less than 1")
    n = len(candles)
    train_end = int(n * train_fraction)
    validation_end = int(n * (train_fraction + validation_fraction))
    if train_end < 1 or validation_end <= train_end or validation_end >= n:
        raise ValueError("Not enough candles for three chronological segments")
    return {
        "train": (0, train_end),
        "validation": (train_end, validation_end),
        "test": (validation_end, n),
    }


def candidate_score(metrics: dict[str, Any], min_trades: int = 30) -> tuple[float, ...]:
    """Rank only by validation metrics; candidates below basic evidence gates rank last."""
    trades = int(metrics.get("trades", 0))
    pf = float(metrics.get("profit_factor", 0.0))
    expectancy = float(metrics.get("expectancy_per_trade", 0.0))
    drawdown = float(metrics.get("max_closed_trade_drawdown_pct", 100.0))
    eligible = trades >= min_trades and np.isfinite(pf) and pf > 1.0 and expectancy > 0
    if not eligible:
        return (0.0, expectancy if np.isfinite(expectancy) else -np.inf,
                -drawdown if np.isfinite(drawdown) else -np.inf, float(trades))
    return (1.0, expectancy, -drawdown, float(trades))
