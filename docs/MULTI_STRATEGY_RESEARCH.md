# Multi-Strategy Research Benchmark

This iteration adds a common evaluation path for independent strategy families. It is research-only: no broker connection, order submission, automatic promotion, or code self-modification is enabled.

## Run in Google Colab

From the existing clone:

```python
%cd /content/T2
!git pull
!python -m pip install -r requirements.txt
!python -m pytest -q
```

Confirm that the normalized XAUUSD M5 CSV exists at:

`/content/T2/data/historical/xauusd_m5_2024-10-28_to_2026-10-28.csv`

Then run:

```python
%cd /content/T2
!python scripts/run_strategy_benchmark.py
```

Or pass a different normalized CSV path:

```python
!python scripts/run_strategy_benchmark.py /path/to/market_data.csv
```

Reports are written to `reports/strategy_benchmark/` as a JSON summary and CSV leaderboard.

## Families included

- **EMA crossover:** long and short diagnostics using causal EMA crossovers and ATR-based stops.
- **Opening-range breakout (ORB):** long and short diagnostics using a configurable UTC session start (default 08:00 UTC) and 30-minute range. This is a generic hypothesis, not a verified broker/exchange calendar; change the session to the session being studied before interpreting results.
- **VWAP + EMA:** only included when the source CSV has a genuine `volume` column. The loader never fabricates volume. Current Dukascopy data may be bid OHLC only, so VWAP can be skipped.
- **Random Forest:** target-aware classifier using the probability that a long trade reaches a 2R target before a 1R stop within 12 bars. It is a CPU-based baseline, not a GPU model. The neural network remains a separate CUDA experiment.

All rule-based candidates use the shared backtest engine. The RF threshold is chosen on validation data only. The final chronological test segment is diagnostic only and must not be used to choose parameters.

## Target labels

`t2/improvement/target_labels.py` labels a long trade using next-candle-open entry and an ATR-derived stop distance. The entry candle is deliberately not inspected by the labeler, matching the backtest engine's conservative OHLC rule. Within the following horizon:
- 1 = target touched before stop;
- 0 = stop first, both levels touched in one candle (conservative), or timeout;
- NaN = insufficient complete future horizon or invalid barrier.

This is still a simplified OHLC label. It does not include broker-specific spread, commission, financing, gap-through-stop execution, or tick-level path ordering.

## Common comparison rules

- Chronological 60% train / 20% validation / 20% test.
- Starting equity 10,000 arbitrary account units, 0.25% risk per trade, 2R target, and provisional 0.02% slippage per side; commission is set to zero because broker terms are not yet configured.
- Each direction is evaluated separately, not as one combined portfolio.
- Candidates pass the provisional validation gate only with at least 30 trades, profit factor > 1, and positive expectancy. Passing is not proof of robustness or readiness to trade.
- A strategy that fails validation remains ineligible even if its test segment happens to be profitable.
- Test performance is not used for selection; do not iterate against this same test segment.

## Known limitations / next work

1. Source data quality, spread/ask, and genuine volume need verification. Do not interpret provisional cost sensitivity as a broker-accurate forecast.
2. ORB session boundaries should be configured to the intended trading session and checked against market hours.
3. The current engine does not mark open positions to market, calculate intratrade drawdown, or model gap-through-stop fills.
4. The long-only RF classifier is a first baseline. Short-side labeling and a true joint portfolio simulator are future work.
5. Add walk-forward evaluation and a paper-trading shadow monitor only after the data and execution model are validated.
6. No strategy is automatically promoted, and no live orders can be submitted by these scripts.
