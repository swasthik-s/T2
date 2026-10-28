"""Performance reporting and time-based evaluation for T2 backtests.

The evaluation layer consumes the existing backtest result dictionary; it does not
create signals, place orders, or modify the backtest engine.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


_REQUIRED_TRADE_FIELDS = {
    "entry_time", "exit_time", "side", "gross_pnl", "costs", "net_pnl",
}


def _trades_frame(backtest: Mapping[str, Any]) -> pd.DataFrame:
    trades = backtest.get("trade_log", [])
    if not isinstance(trades, Sequence) or isinstance(trades, (str, bytes)):
        raise ValueError("trade_log must be a sequence of trade records")
    if not trades:
        return pd.DataFrame(columns=[
            "entry_time", "exit_time", "side", "gross_pnl", "costs", "net_pnl",
        ])
    frame = pd.DataFrame(trades)
    missing = sorted(_REQUIRED_TRADE_FIELDS - set(frame.columns))
    if missing:
        raise ValueError("trade_log is missing required fields: " + ", ".join(missing))
    for column in ("gross_pnl", "costs", "net_pnl"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f"trade_log field '{column}' must contain finite values")
    for column in ("entry_time", "exit_time"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
        if frame[column].isna().any():
            raise ValueError(f"trade_log field '{column}' contains missing timestamps")
    if not frame["side"].isin(["long", "short"]).all():
        raise ValueError("trade_log side must be 'long' or 'short'")
    if (frame["costs"] < 0).any():
        raise ValueError("trade costs cannot be negative")
    if (frame["exit_time"] < frame["entry_time"]).any():
        raise ValueError("trade exit_time cannot precede entry_time")
    return frame.sort_values(["exit_time", "entry_time"], kind="stable").reset_index(drop=True)


def build_equity_curve(backtest: Mapping[str, Any]) -> pd.DataFrame:
    """Return closed-trade equity and peak-to-trough drawdown time series.

    Unrealized P&L from an open position at the dataset end is not included,
    consistent with the current engine's explicit open-position assumption.
    """
    initial = float(backtest.get("initial_equity", 0.0))
    if not np.isfinite(initial) or initial <= 0:
        raise ValueError("initial_equity must be positive and finite")
    trades = _trades_frame(backtest)
    if trades.empty:
        return pd.DataFrame({
            "timestamp": pd.Series(dtype="datetime64[ns, UTC]"),
            "trade_number": pd.Series(dtype="int64"),
            "trade_pnl": pd.Series(dtype="float64"),
            "equity": pd.Series(dtype="float64"),
            "peak_equity": pd.Series(dtype="float64"),
            "drawdown": pd.Series(dtype="float64"),
            "drawdown_pct": pd.Series(dtype="float64"),
        })

    trades = trades.copy()
    trades["trade_number"] = np.arange(1, len(trades) + 1)
    trades["trade_pnl"] = trades["net_pnl"].astype(float)
    trades["equity"] = initial + trades["trade_pnl"].cumsum()
    trades["peak_equity"] = trades["equity"].cummax().clip(lower=initial)
    trades["drawdown"] = trades["peak_equity"] - trades["equity"]
    trades["drawdown_pct"] = np.where(
        trades["peak_equity"] > 0,
        trades["drawdown"] / trades["peak_equity"] * 100.0,
        0.0,
    )
    return trades[[
        "exit_time", "trade_number", "trade_pnl", "equity", "peak_equity",
        "drawdown", "drawdown_pct",
    ]].rename(columns={"exit_time": "timestamp"})


def summarize_backtest(backtest: Mapping[str, Any]) -> dict[str, Any]:
    """Calculate headline performance metrics, costs, and long/short splits."""
    initial = float(backtest.get("initial_equity", 0.0))
    if not np.isfinite(initial) or initial <= 0:
        raise ValueError("initial_equity must be positive and finite")
    trades = _trades_frame(backtest)
    equity = build_equity_curve(backtest)
    pnl = trades["net_pnl"].astype(float) if not trades.empty else pd.Series(dtype=float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    net_pnl = float(pnl.sum())
    count = int(len(trades))
    side_summaries: dict[str, dict[str, float | int]] = {}
    for side in ("long", "short"):
        subset = trades.loc[trades["side"] == side, "net_pnl"].astype(float) if not trades.empty else pd.Series(dtype=float)
        side_win = subset[subset > 0]
        side_loss = subset[subset < 0]
        side_gp = float(side_win.sum())
        side_gl = float(-side_loss.sum())
        side_summaries[side] = {
            "trades": int(len(subset)),
            "net_pnl": float(subset.sum()),
            "win_rate_pct": float((subset > 0).mean() * 100.0) if len(subset) else 0.0,
            "profit_factor": side_gp / side_gl if side_gl else (float("inf") if side_gp else 0.0),
            "expectancy_per_trade": float(subset.mean()) if len(subset) else 0.0,
        }
    total_costs = float(trades["costs"].sum()) if count else 0.0
    return {
        "initial_equity": initial,
        "final_equity_closed_trades": initial + net_pnl,
        "net_pnl": net_pnl,
        "return_pct_closed_trades": net_pnl / initial * 100.0,
        "trades": count,
        "wins": int((pnl > 0).sum()),
        "losses": int((pnl < 0).sum()),
        "breakeven_trades": int((pnl == 0).sum()),
        "win_rate_pct": float((pnl > 0).mean() * 100.0) if count else 0.0,
        "profit_factor": gross_profit / gross_loss if gross_loss else (float("inf") if gross_profit else 0.0),
        "expectancy_per_trade": float(pnl.mean()) if count else 0.0,
        "average_win": float(wins.mean()) if len(wins) else 0.0,
        "average_loss": float(losses.mean()) if len(losses) else 0.0,
        "gross_profit_from_wins": gross_profit,
        "gross_loss_from_losses": gross_loss,
        "total_costs": total_costs,
        "max_closed_trade_drawdown": float(equity["drawdown"].max()) if not equity.empty else 0.0,
        "max_closed_trade_drawdown_pct": float(equity["drawdown_pct"].max()) if not equity.empty else 0.0,
        "long_short": side_summaries,
        "open_position_at_end_ignored": bool(backtest.get("open_position_at_end_ignored", False)),
        "drawdown_scope": "Closed-trade equity only; intratrade/unrealized drawdown is not measured.",
    }


def monthly_performance(backtest: Mapping[str, Any]) -> pd.DataFrame:
    """Group closed trades by UTC exit month."""
    trades = _trades_frame(backtest)
    columns = ["month", "trades", "wins", "losses", "net_pnl", "total_costs", "win_rate_pct"]
    if trades.empty:
        return pd.DataFrame(columns=columns)
    trades = trades.copy()
    trades["month"] = trades["exit_time"].dt.strftime("%Y-%m")
    trades["is_win"] = trades["net_pnl"] > 0
    trades["is_loss"] = trades["net_pnl"] < 0
    out = trades.groupby("month", sort=True).agg(
        trades=("net_pnl", "size"),
        wins=("is_win", "sum"),
        losses=("is_loss", "sum"),
        net_pnl=("net_pnl", "sum"),
        total_costs=("costs", "sum"),
        win_rate_pct=("is_win", "mean"),
    ).reset_index()
    out["win_rate_pct"] = out["win_rate_pct"] * 100.0
    out[["trades", "wins", "losses"]] = out[["trades", "wins", "losses"]].astype(int)
    return out[columns]


def chronological_split(
    candles: pd.DataFrame,
    train_fraction: float = 0.7,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split candles chronologically, never randomly, into train and test sets.

    This helper only splits rows. Strategy indicators/signals must still be
    computed without leaking future information; use a warm-up window when
    evaluating indicator-based strategies on the test segment.
    """
    from t2.data.ohlcv import validate_ohlcv

    if not np.isfinite(train_fraction) or not 0.5 <= train_fraction < 1.0:
        raise ValueError("train_fraction must be finite and in [0.5, 1)")
    data = validate_ohlcv(candles)
    split_at = int(len(data) * train_fraction)
    if split_at < 1 or split_at >= len(data):
        raise ValueError("Need at least two candles to split")
    return data.iloc[:split_at].reset_index(drop=True), data.iloc[split_at:].reset_index(drop=True)
