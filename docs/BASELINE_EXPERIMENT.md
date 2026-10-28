# T2 EMA/ATR baseline experiment

Run from the repository root after placing the normalized historical CSV at
`data/historical/xauusd_m5_2024-10-28_to_2026-10-28.csv`:

```bash
python scripts/run_baseline.py
```

Or pass a different CSV path as the first argument.

## Fixed baseline rules

- Instrument/timeframe: XAUUSD M5
- Long: EMA(20) crosses above EMA(50) on candle close
- Short: EMA(20) crosses below EMA(50) on candle close
- Entry: next candle open
- Stop distance: 1.5 × ATR(14), using a simple rolling mean of true range
- Target: 2R
- Risk sizing: 0.25% of current closed-trade equity
- Initial equity: 10,000 arbitrary account-currency units
- Cost placeholder: 0.02% slippage per side, no explicit commission
- VWAP and volume are not used

The cost placeholder is not broker-calibrated. It is deliberately visible and
must be tested at several higher/lower values. Bid-only OHLC does not contain
the ask/spread, so these results are not an execution-accurate simulation.

## Evaluation protocol

The script computes indicators causally over the chronological data, then
splits rows 70%/30%. The final 30% is out-of-sample for this fixed baseline.
Do not tune parameters based on those results and continue calling them
out-of-sample; if parameters change, reserve a new untouched period or use
walk-forward evaluation.

Long and short sides run separately through the single-side engine. They are
directional diagnostics, not a combined account equity curve, because separate
runs may overlap in time. No broker connection or order submission exists in
this script. Paper trading only; live trading remains disabled.
