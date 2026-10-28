import pandas as pd
import pytest

from t2.improvement.research_loop import (
    candidate_score,
    chronological_three_way_split,
    generate_candidates,
)


def test_candidate_grid_is_bounded_and_excludes_invalid_ema_pairs():
    candidates = generate_candidates()
    assert len(candidates) == 18
    assert all(candidate["fast_ema"] < candidate["slow_ema"] for candidate in candidates)
    assert {candidate["atr_stop_multiple"] for candidate in candidates} == {1.0, 1.5, 2.0}


def test_three_way_split_is_chronological_and_covers_all_rows():
    candles = pd.DataFrame({"x": range(100)})
    ranges = chronological_three_way_split(candles)
    assert ranges == {
        "train": (0, 60),
        "validation": (60, 80),
        "test": (80, 100),
    }


@pytest.mark.parametrize("train,validation", [(0, 0.2), (0.8, 0.2), (0.6, -0.1), (float("nan"), 0.2)])
def test_three_way_split_rejects_invalid_fractions(train, validation):
    with pytest.raises(ValueError):
        chronological_three_way_split(
            pd.DataFrame({"x": range(100)}), train, validation
        )


def test_candidate_score_gate_requires_positive_expectancy_and_pf_above_one():
    good = {
        "trades": 40,
        "profit_factor": 1.2,
        "expectancy_per_trade": 2.0,
        "max_closed_trade_drawdown_pct": 8.0,
    }
    bad = {**good, "profit_factor": 0.99}
    assert candidate_score(good)[0] == 1.0
    assert candidate_score(bad)[0] == 0.0
