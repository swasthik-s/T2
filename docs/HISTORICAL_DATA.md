# T2 Stage 1 — Download real XAUUSD M5 data

## Research window

Target window: **2024-10-28 through 2026-10-27 UTC** (24 months before 2026-10-28).

Data source: Dukascopy historical market data. Dukascopy publishes historical data for metals and other instruments; the third-party dukascopy-node CLI provides a convenient CSV export interface. See:
- Dukascopy Historical Data Export: https://www.dukascopy.com/swiss/english/marketwatch/historical/
- XAUUSD downloader/timeframe documentation: https://www.dukascopy-node.app/instrument/xauusd

The downloaded series represents that source's feed, not every broker's XAUUSD quote stream. Tick volume is not exchange-traded gold volume. Preserve the original downloaded file for provenance.

## Run in Google Colab

From the notebook, clone/pull T2 first and enter the repository:

~~~python
%cd /content/T2
!git pull origin main
~~~

Download 5-minute XAUUSD bars:

~~~bash
!npx --yes dukascopy-node -i xauusd -from 2024-10-28 -to 2026-10-28 -t m5 -f csv
~~~

The CLI may take time for a 24-month range. When it finishes, list the output:

~~~bash
!ls -lh xauusd*.csv
~~~

If the CLI reports a usage/argument error, run !npx --yes dukascopy-node --help and inspect its installed-version syntax instead of guessing another command.

## Validate and normalize the downloaded file

Run this cell after identifying the downloaded CSV. It selects the first matching file, validates OHLCV structure, sorts rows in UTC order, and writes a normalized copy and data-quality report. If multiple files were produced, set source_file to the exact intended file instead of accepting the first match.

~~~python
from pathlib import Path
from t2.data.ohlcv import load_ohlcv_csv, describe_ohlcv
import json

candidates = sorted(Path("/content/T2").glob("xauusd*.csv"))
if not candidates:
    raise FileNotFoundError("No xauusd*.csv file found. Inspect the downloader output.")

source_file = candidates[0]
raw = load_ohlcv_csv(source_file)

out_dir = Path("/content/T2/data/historical")
out_dir.mkdir(parents=True, exist_ok=True)
canonical_file = out_dir / "xauusd_m5_2024-10-28_to_2026-10-28.csv"
report_file = out_dir / "xauusd_m5_data_quality.json"

raw.to_csv(canonical_file, index=False)
report = describe_ohlcv(raw)
report.update({
    "source": "Dukascopy via dukascopy-node",
    "instrument": "XAUUSD",
    "requested_timeframe": "M5",
    "requested_start_utc": "2024-10-28",
    "requested_end_utc": "2026-10-28",
    "source_file": source_file.name,
    "normalized_file": canonical_file.name,
    "warning": "Verify actual first/last timestamps and missing sessions before backtesting. This is source-specific XAUUSD data, not a universal broker price series."
})
report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

print(json.dumps(report, indent=2))
print(f"Normalized CSV: {canonical_file}")
print(f"Quality report: {report_file}")
~~~

## Acceptance checks before backtesting

1. Confirm the printed start_utc and end_utc actually cover the requested window. The CLI's date-boundary semantics may affect the last date.
2. Confirm row count is plausible for M5 and that the downloader did not silently return only a partial range.
3. Review missing candle intervals against normal weekend/market closures. T2 intentionally does not fill gaps automatically.
4. Preserve the raw file and data-quality JSON. Do not overwrite the original download.
5. Do not infer spread from OHLC alone. Until bid/ask data or a documented spread model is available, backtest results need an explicit spread/slippage assumption.
6. VWAP remains excluded. This workflow only downloads and validates data; it does not generate trades or enable broker orders.
