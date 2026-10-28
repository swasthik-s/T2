"""Deterministic, long/short, single-position OHLCV backtesting engine.

Signals are supplied by the caller and evaluated at candle close. Entries occur
at the next candle open to reduce look-ahead bias. Stop/target checks use each
bar's high/low; if both are touched in one candle, the conservative assumption
is that the stop is hit first. No broker orders are submitted.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd

from t2.data.ohlcv import validate_ohlcv

Side = Literal["long", "short"]


@dataclass(frozen=True)
class BacktestConfig:
    initial_equity: float = 10_000.0
    risk_per_trade_pct: float = 0.25
    reward_risk_ratio: float = 2.0
    commission_pct_per_side: float = 0.0
    slippage_pct_per_side: float = 0.0
    max_trades: int = 10_000

    def __post_init__(self) -> None:
        if self.initial_equity <= 0:
            raise ValueError("initial_equity must be positive")
        if not 0 < self.risk_per_trade_pct <= 1:
            raise ValueError("risk_per_trade_pct must be > 0 and <= 1")
        if self.reward_risk_ratio <= 0:
            raise ValueError("reward_risk_ratio must be positive")
        if self.commission_pct_per_side < 0 or self.slippage_pct_per_side < 0:
            raise ValueError("Costs cannot be negative")
        if self.max_trades < 1:
            raise ValueError("max_trades must be at least 1")


@dataclass(frozen=True)
class Trade:
    side: Side
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    quantity: float
    stop_price: float
    target_price: float
    gross_pnl: float
    costs: float
    net_pnl: float
    outcome: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _cost(notional: float, config: BacktestConfig) -> float:
    return notional * (config.commission_pct_per_side + config.slippage_pct_per_side) / 100.0


def run_backtest(
    candles: pd.DataFrame,
    entry_signal: pd.Series,
    stop_distance: pd.Series | float,
    config: BacktestConfig | None = None,
    side: Side = "long",
) -> dict[str, object]:
    """Run a basic single-position backtest using close-confirmed entry signals.

    ``entry_signal[i]`` is only known after candle i closes, so entry is at the
    next candle's open. ``stop_distance`` is a positive price distance aligned
    to the signal candle (or a positive constant). Quantity is sized from the
    current equity and stop distance. At most one position is open at a time.
    """
    cfg = config or BacktestConfig()
    if side not in ("long", "short"):
        raise ValueError("side must be 'long' or 'short'")
    data = validate_ohlcv(candles)
    signals = pd.Series(entry_signal).reset_index(drop=True)
    if len(signals) != len(data):
        raise ValueError("entry_signal length must match candles")
    if signals.isna().any():
        raise ValueError("entry_signal cannot contain missing values")
    if not signals.map(lambda x: isinstance(x, (bool, np.bool_))).all():
        raise ValueError("entry_signal must contain only booleans")
    if isinstance(stop_distance, pd.Series):
        stops = pd.to_numeric(stop_distance.reset_index(drop=True), errors="raise")
    else:
        stops = pd.Series(float(stop_distance), index=range(len(data)))
    if len(stops) != len(data) or not np.isfinite(stops.to_numpy(dtype=float)).all() or (stops <= 0).any():
        raise ValueError("stop_distance must be positive, finite, and aligned to candles")

    equity = float(cfg.initial_equity)
    peak_equity = equity
    max_drawdown_pct = 0.0
    trades: list[Trade] = []
    position: dict[str, object] | None = None
    pending: tuple[int, float] | None = None

    for i, row in data.iterrows():
        ts = row["timestamp"].isoformat()
        # A pending signal is entered at this candle's open, never at the signal candle close.
        if position is None and pending is not None and pending[0] == i:
            distance = float(pending[1])
            entry = float(row["open"])
            risk_cash = equity * cfg.risk_per_trade_pct / 100.0
            quantity = risk_cash / distance
            if side == "long":
                stop = entry - distance
                target = entry + distance * cfg.reward_risk_ratio
            else:
                stop = entry + distance
                target = entry - distance * cfg.reward_risk_ratio
            if stop > 0 and quantity > 0:
                position = {"side": side, "entry_i": i, "entry_time": ts, "entry": entry,
                            "quantity": quantity, "stop": stop, "target": target,
                            "risk_cash": risk_cash}
            pending = None

        if position is not None and i > int(position["entry_i"]):
            stop = float(position["stop"])
            target = float(position["target"])
            hit_stop = row["low"] <= stop if side == "long" else row["high"] >= stop
            hit_target = row["high"] >= target if side == "long" else row["low"] <= target
            exit_price: float | None = None
            outcome = ""
            if hit_stop:  # conservative if both levels touched in same candle
                exit_price, outcome = stop, "loss"
            elif hit_target:
                exit_price, outcome = target, "win"
            if exit_price is not None:
                entry = float(position["entry"])
                qty = float(position["quantity"])
                gross = (exit_price - entry) * qty if side == "long" else (entry - exit_price) * qty
                costs = _cost(entry * qty, cfg) + _cost(exit_price * qty, cfg)
                net = gross - costs
                equity += net
                peak_equity = max(peak_equity, equity)
                drawdown = (peak_equity - equity) / peak_equity * 100 if peak_equity else 0.0
                max_drawdown_pct = max(max_drawdown_pct, drawdown)
                trades.append(Trade(side, str(position["entry_time"]), ts, entry, exit_price,
                                    qty, stop, target, gross, costs, net, outcome))
                position = None
                if len(trades) >= cfg.max_trades:
                    break

        # A signal at this close can only enter on the next candle open.
        if position is None and pending is None and i < len(data) - 1 and bool(signals.iloc[i]):
            pending = (i + 1, float(stops.iloc[i]))

    # Open positions at the end are intentionally not force-closed or scored.
    wins = sum(t.net_pnl > 0 for t in trades)
    losses = sum(t.net_pnl <= 0 for t in trades)
    gross_profit = sum(t.net_pnl for t in trades if t.net_pnl > 0)
    gross_loss = -sum(t.net_pnl for t in trades if t.net_pnl < 0)
    net_pnl = sum(t.net_pnl for t in trades)
    count = len(trades)
    return {
        "initial_equity": cfg.initial_equity,
        "final_equity": cfg.initial_equity + net_pnl,
        "net_pnl": net_pnl,
        "trades": count,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": wins / count * 100 if count else 0.0,
        "profit_factor": gross_profit / gross_loss if gross_loss else (float("inf") if gross_profit else 0.0),
        "expectancy_per_trade": net_pnl / count if count else 0.0,
        "max_drawdown_pct": max_drawdown_pct,
        "trade_log": [trade.to_dict() for trade in trades],
        "open_position_at_end_ignored": position is not None,
        "assumptions": ["Signal evaluated at candle close; entry at next candle open.",
                        "If stop and target are both touched in a candle, stop is assumed first.",
                        "Open position at dataset end is not force-closed.",
                        "Slippage and commission are percentage estimates, not a venue-specific fill model."],
    }
