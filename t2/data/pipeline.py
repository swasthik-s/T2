"""Small orchestration layer for the T2 market-data preparation step."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from t2.data.ohlcv import describe_ohlcv, load_ohlcv_csv


def prepare_csv(input_path: str | Path, report_path: str | Path | None = None) -> dict[str, Any]:
    """Validate a CSV and return a summary, optionally writing a JSON report.

    The report is metadata only; this function does not modify the source CSV,
    infer missing bars, calculate signals, or interact with a broker.
    """
    data = load_ohlcv_csv(input_path)
    report = describe_ohlcv(data)
    report["source_file"] = Path(input_path).name
    if report_path is not None:
        target = Path(report_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
