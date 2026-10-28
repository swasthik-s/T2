# T2 bounded autonomous research loop

## What it does

Run `python scripts/run_research_loop.py [path-to-normalized-csv]` to execute a reproducible, bounded experiment cycle:

1. Generates a fixed grid of 18 EMA/ATR candidates.
2. Calculates causal signals on the historical series.
3. Evaluates all candidates on chronological 60% train and 20% validation segments using one unchanged provisional cost assumption.
4. Selects the highest validation-ranked candidate separately for long and short diagnostics. The eligibility gate requires at least 30 trades, finite profit factor above 1, and positive expectancy.
5. Evaluates only those selected candidates on the final 20% test segment, with a slippage-cost sensitivity table (0%, 0.005%, 0.01%, 0.02% per side).
6. Saves a leaderboard, per-side test reports, a run summary, and append-only JSONL experiment memory.

If no candidate passes the validation gate, the highest-ranked failed candidate is still tested for diagnosis, but is explicitly marked ineligible. This is not promotion. There is no broker connection, order submission, or automatic strategy promotion.

## Colab setup

```python
%cd /content/T2
!git pull origin main
!python -m pytest -v
!python scripts/run_research_loop.py
```

If the data is in a different location:

```python
!python scripts/run_research_loop.py /content/T2/data/historical/xauusd_m5_2024-10-28_to_2026-10-28.csv
```

Outputs are written under `reports/autonomous_research/`. Download or sync this directory before the Colab runtime is deleted. The JSONL memory is persistent only if the report directory is saved to GitHub, Drive, or another persistent store; local Colab storage alone is temporary.

## Important limitations

- This is autonomous execution of a small, predefined search space, not an AI agent inventing arbitrary new strategies yet.
- Long and short candidates are separate directional diagnostics, not a combined portfolio.
- The same test period must not be repeatedly used to steer later iterations. For recurring experiments, reserve fresh data or use nested walk-forward testing.
- Bid-only data lacks ask prices/spread and source volume. The percentage cost model is provisional, not broker-calibrated.
- A strategy must not be promoted solely because a metric crosses a threshold.
- No live trading is enabled. No orders are submitted.
- This event-driven pandas backtest is CPU-bound; a Colab GPU does not materially accelerate it. GPU use should be added for suitable ML workloads (e.g. training/evaluating a model after a reliable feature pipeline exists), not forced into a classical backtest.

## Current autonomy boundary

Automated: candidate grid generation, train/validation evaluation, validation-only ranking, held-out test for selected candidates, cost sensitivity, report writing, and experiment-history append.

Still manual / future work: hypothesis generation by an AI research agent, code generation in a sandbox, stronger walk-forward/robustness diagnostics, broker-calibrated costs, durable storage across Colab sessions, and any paper-execution integration.
