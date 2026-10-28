"""Configuration loading and safety checks for T2."""
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load YAML configuration and enforce critical safety invariants."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    trading = config.get("trading", {})
    if trading.get("live_trading_enabled") is not False:
        raise ValueError("T2 foundation requires live_trading_enabled: false")
    if trading.get("paper_trading_only") is not True:
        raise ValueError("T2 foundation requires paper_trading_only: true")
    if trading.get("allow_order_submission") is not False:
        raise ValueError("Order submission must remain disabled until explicitly implemented and tested.")

    excluded = {str(item).strip().upper() for item in config.get("research", {}).get("exclude_indicators", [])}
    if "VWAP" not in excluded:
        raise ValueError("VWAP must remain explicitly excluded as requested.")

    risk_pct = float(trading.get("risk_per_trade_pct", 0))
    daily_loss_pct = float(trading.get("max_daily_loss_pct", 0))
    if not 0 < risk_pct <= 1:
        raise ValueError("risk_per_trade_pct must be greater than 0 and at most 1 percent.")
    if not 0 < daily_loss_pct <= 2:
        raise ValueError("max_daily_loss_pct must be greater than 0 and at most 2 percent.")

    return config
