"""Deterministic feedback rules for the self-correcting research cycle.

Feedback may schedule bounded experiments and flag data/model problems. It never
edits source code, changes risk limits, or promotes a strategy automatically.
"""
from __future__ import annotations

from typing import Any


def diagnose_experiment(metrics: dict[str, Any]) -> list[dict[str, str]]:
    """Return grounded diagnoses from validation/test metrics."""
    advice: list[dict[str, str]] = []
    validation_pf = float(metrics.get("validation_profit_factor", 0.0))
    validation_exp = float(metrics.get("validation_expectancy_per_trade", 0.0))
    test_pf = float(metrics.get("test_profit_factor", 0.0))
    test_exp = float(metrics.get("test_expectancy_per_trade", 0.0))
    test_cost_pnl = metrics.get("test_pnl_by_cost", {})
    if validation_pf <= 1 or validation_exp <= 0:
        advice.append({
            "finding": "validation_edge_not_established",
            "next_experiment": "Do not tune this candidate further; test a different entry hypothesis or market-regime filter.",
            "priority": "high",
        })
    if test_pf <= 1 or test_exp <= 0:
        advice.append({
            "finding": "held_out_test_not_profitable",
            "next_experiment": "Keep candidate ineligible; record the failure and evaluate an independent strategy family.",
            "priority": "high",
        })
    if isinstance(test_cost_pnl, dict) and test_cost_pnl:
        values = []
        for key, value in test_cost_pnl.items():
            try:
                values.append((float(key), float(value)))
            except (TypeError, ValueError):
                continue
        if values:
            values.sort()
            if values[0][1] > 0 and values[-1][1] <= 0:
                advice.append({
                    "finding": "edge_erased_by_costs",
                    "next_experiment": "Obtain broker-specific spread/commission data and test lower-turnover or higher-selectivity entries.",
                    "priority": "high",
                })
    if not advice:
        advice.append({
            "finding": "no_failure_rule_triggered",
            "next_experiment": "Continue walk-forward and parameter-sensitivity tests; do not promote based on one split.",
            "priority": "normal",
        })
    return advice


def research_guardrails() -> dict[str, Any]:
    """Machine-readable invariants for automated orchestration."""
    return {
        "live_trading_enabled": False,
        "allow_order_submission": False,
        "automatic_promotion": False,
        "test_set_may_select_candidates": False,
        "source_code_self_modification": False,
        "require_validation_and_walk_forward": True,
        "record_failed_experiments": True,
    }
