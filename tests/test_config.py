from pathlib import Path

import pytest
import yaml

from t2.config import DEFAULT_CONFIG_PATH, load_config


def test_default_config_is_paper_only():
    config = load_config()
    trading = config["trading"]
    assert trading["paper_trading_only"] is True
    assert trading["live_trading_enabled"] is False
    assert trading["allow_order_submission"] is False
    assert config["research"]["auto_promote_strategy"] is False


def test_vwap_is_explicitly_excluded():
    config = load_config()
    excluded = {item.upper() for item in config["research"]["exclude_indicators"]}
    assert "VWAP" in excluded


def test_loader_rejects_live_trading(tmp_path: Path):
    config = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    config["trading"]["live_trading_enabled"] = True
    candidate = tmp_path / "unsafe.yaml"
    candidate.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="live_trading_enabled"):
        load_config(candidate)


def test_loader_rejects_order_submission_enabled(tmp_path: Path):
    config = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    config["trading"]["allow_order_submission"] = True
    candidate = tmp_path / "unsafe.yaml"
    candidate.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(ValueError, match="Order submission"):
        load_config(candidate)
