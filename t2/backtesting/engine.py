"""Deterministic, long/short, single-position OHLCV backtesting engine.

Signals are evaluated at candle close; entries occur at the next candle open.
Stops/targets are checked from subsequent candles' high/low. If both are touched
within one candle, the stop is assumed first. This engine never submits orders.
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
        if not np.isfinite(self.initial_equity) or self.initial_equity <= 0:
            raise ValueError("initial_equity must be positive and finite")
        if not np.isfinite(self.risk_per_trade_pct) or not 0 < self.risk_per_trade_pct <= 1:
            raise ValueError("risk_per_trade_pct must be > 0 and <= 1")
        if not np.isfinite(self.reward_risk_ratio) or self.reward_risk_ratio <= 0:
            raise ValueError("reward_risk_ratio must be positive and finite")
        if (not np.isfinite(self.commission_pct_per_side) or self.commission_pct_per_side < 0
                or not np.isfinite(self.slippage_pct_per_side) or self.slippage_pct_per_side < 0):
            raise ValueError("Costs must be non-negative and finite")
        if isinstance(self.max_trades, bool) or not isinstance(self.max_trades, int) or self.max_trades < 1:
            raise ValueError("max_trades must be a positive integer")


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
    """Run a single-position backtest from boolean close-confirmed signals.

    `entry_signal[i]` is known only at candle i close, so it can enter at
    candle i+1 open. Stop distance is a positive price distance aligned to
    the signal candle (or a positive constant). Only one position may be open.
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

    try:
        if isinstance(stop_distance, pd.Series):
            stops = pd.to_numeric(stop_distance.reset_index(drop=True), errors="raise")
        else:
            distance = float(stop_distance)
            stops = pd.Series(distance, index=range(len(data)))
    except (TypeError, ValueError) as exc:
        raise ValueError("stop_distance must be positive, finite, and aligned to candles") from exc
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

        # Enter at the next candle open after a signal candle has closed.
        if position is None and pending is not None and pending[0] == i:
            distance = float(pending[1])
            entry = float(row["open"])
            risk_cash = equity * cfg.risk_per_trade_pct / 100.0
            quantity = risk_cash / distance
            stop = entry - distance if side == "long" else entry + distance
            target = entry + distance * cfg.reward_risk_ratio if side == "long" else entry - distance * cfg.reward_risk_ratio
            if stop > 0 and target > 0 and np.isfinite(quantity) and quantity > 0:
                position = {
                    "side": side, "entry_i": i, "entry_time": ts, "entry": entry,
                    "quantity": quantity, "stop": stop, "target": target,
                }
            pending = None

        # Do not inspect the entry candle's high/low; OHLC alone cannot reveal
        # whether the stop/target was touched before or after the entry.
        if position is not None and i > int(position["entry_i"]):
            stop = float(position["stop"])
            target = float(position["target"])
            hit_stop = row["low"] <= stop if side == "long" else row["high"] >= stop
            hit_target = row["high"] >= target if side == "long" else row["low"] <= target
            exit_price: float | None = None
            outcome = ""
            if hit_stop:
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
                drawdown = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0.0
                max_drawdown_pct = max(max_drawdown_pct, drawdown)
                trades.append(Trade(side, str(position["entry_time"]), ts, entry, exit_price,
                                    qty, stop, target, gross, costs, net, outcome))
                position = None
                if len(trades) >= cfg.max_trades:
                    break

        if position is None and pending is None and i < len(data) - 1 and bool(signals.iloc[i]):
            pending = (i + 1, float(stops.iloc[i]))

    # Open positions at the dataset boundary are reported but not marked to
    # market; this prevents silently inventing an exit price.
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
        "assumptions": [
            "Signal evaluated at candle close; entry at next candle open.",
            "Stops/targets are checked from candles after entry; same-candle ambiguity is avoided.",
            "If stop and target are both touched in a candle, stop is assumed first.",
            "Open position at dataset end is not force-closed or marked to market.",
            "Gap-through-stop fills, spread, financing, and venue-specific fill details are not modeled.",
            "Slippage and commission are percentage estimates rather than a venue-specific fill model.",
        ],
    }
