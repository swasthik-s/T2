import json
from pathlib import Path

from t2.data.pipeline import prepare_csv


def test_pipeline_reads_sample_and_writes_json_report(tmp_path: Path):
    sample = Path("data/sample_ohlcv.csv")
    report_file = tmp_path / "reports" / "sample_report.json"
    report = prepare_csv(sample, report_file)

    saved = json.loads(report_file.read_text(encoding="utf-8"))
    assert report == saved
    assert report["rows"] == 5
    assert report["duplicate_timestamps"] == 0
    assert report["source_file"] == "sample_ohlcv.csv"
    assert report["missing_intervals_checked"] is False


def test_pipeline_can_run_without_writing_report():
    report = prepare_csv("data/sample_ohlcv.csv")
    assert report["rows"] == 5
    assert report["start_utc"].startswith("2026-01-02T00:00:00")
