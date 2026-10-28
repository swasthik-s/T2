"""GPU-capable, leakage-aware ML research utilities for T2.

PyTorch is optional. This module never submits orders or changes strategy code.
Labels are future returns; features are causal. Thresholds must be selected on
validation data, and the final test segment must remain untouched until then.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

FEATURE_COLUMNS = (
    "ret_1", "ret_3", "ret_6", "ret_12",
    "ema_gap_atr", "atr_pct", "range_atr", "body_atr",
    "volatility_12", "hour_sin", "hour_cos",
)


def build_causal_features(candles: pd.DataFrame, atr_period: int = 14) -> pd.DataFrame:
    """Build features using only current/past OHLC and timestamp information."""
    if atr_period < 2:
        raise ValueError("atr_period must be >= 2")
    close = pd.to_numeric(candles["close"], errors="raise").astype(float)
    high = pd.to_numeric(candles["high"], errors="raise").astype(float)
    low = pd.to_numeric(candles["low"], errors="raise").astype(float)
    open_ = pd.to_numeric(candles["open"], errors="raise").astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(),
                    (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(atr_period, min_periods=atr_period).mean().replace(0, np.nan)
    ema20 = close.ewm(span=20, adjust=False, min_periods=20).mean()
    ema50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
    log_return = np.log(close / close.shift(1))
    timestamp = pd.to_datetime(candles["timestamp"], utc=True)
    hour = timestamp.dt.hour + timestamp.dt.minute / 60.0
    result = pd.DataFrame(index=candles.index)
    for lag in (1, 3, 6, 12):
        result[f"ret_{lag}"] = close.pct_change(lag)
    result["ema_gap_atr"] = (ema20 - ema50) / atr
    result["atr_pct"] = atr / close
    result["range_atr"] = (high - low) / atr
    result["body_atr"] = (close - open_).abs() / atr
    result["volatility_12"] = log_return.rolling(12, min_periods=12).std()
    result["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    result["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    return result.replace([np.inf, -np.inf], np.nan)


def make_direction_labels(candles: pd.DataFrame, horizon: int = 3) -> pd.Series:
    """Binary label: future close exceeds current close; final horizon is unknown."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    close = pd.to_numeric(candles["close"], errors="raise").astype(float)
    future_return = close.shift(-horizon) / close - 1.0
    labels = (future_return > 0).astype(float)
    labels[future_return.isna()] = np.nan
    return labels


def select_device(prefer_gpu: bool = True) -> str:
    """Return cuda when available, otherwise CPU; never assumes GPU exists."""
    try:
        import torch
    except ImportError:
        return "cpu (PyTorch not installed)"
    if prefer_gpu and torch.cuda.is_available():
        return "cuda"
    if prefer_gpu and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class MLPFit:
    model: Any
    device: str
    feature_mean: np.ndarray
    feature_std: np.ndarray
    epochs_trained: int


def fit_direction_mlp(
    features: pd.DataFrame,
    labels: pd.Series,
    train_end: int,
    *,
    epochs: int = 20,
    batch_size: int = 2048,
    learning_rate: float = 1e-3,
    seed: int = 17,
    prefer_gpu: bool = True,
) -> MLPFit:
    """Fit a small binary classifier on the training prefix only."""
    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for GPU/ML experiments. In Colab, install it "
            "using the runtime's compatible PyTorch build, then restart if needed."
        ) from exc
    if not 0 < train_end <= len(features) or epochs < 1 or batch_size < 1:
        raise ValueError("Invalid training boundary or training parameters")
    x_frame = features.iloc[:train_end].copy()
    y_series = labels.iloc[:train_end].copy()
    valid = x_frame.notna().all(axis=1) & y_series.notna()
    x = x_frame.loc[valid].to_numpy(dtype=np.float32)
    y = y_series.loc[valid].to_numpy(dtype=np.float32)
    if len(x) < 100 or len(np.unique(y)) < 2:
        raise ValueError("Insufficient valid training rows or only one label class")
    if not np.isfinite(x).all():
        raise ValueError("Training features contain non-finite values")
    torch.manual_seed(seed)
    device = select_device(prefer_gpu)
    if device == "cpu (PyTorch not installed)":
        raise RuntimeError("PyTorch is not installed")
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = x.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    x = (x - mean) / std
    device_obj = torch.device(device)
    model = nn.Sequential(
        nn.Linear(x.shape[1], 64), nn.ReLU(), nn.Dropout(0.15),
        nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.10),
        nn.Linear(32, 1),
    ).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    tx = torch.from_numpy(x).to(device_obj)
    ty = torch.from_numpy(y).to(device_obj).reshape(-1, 1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(tx), generator=generator).to(device_obj)
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(tx[idx]), ty[idx])
            loss.backward()
            optimizer.step()
    model.eval()
    return MLPFit(model, device, mean, std, epochs)


def predict_probabilities(fit: MLPFit, features: pd.DataFrame, batch_size: int = 8192) -> pd.Series:
    """Predict probabilities for rows with valid causal features; invalid rows remain NaN."""
    import torch
    x = features.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    valid = np.isfinite(x).all(axis=1)
    out = np.full(len(features), np.nan, dtype=np.float64)
    if valid.any():
        x_valid = (x[valid] - fit.feature_mean) / fit.feature_std
        device = torch.device(fit.device)
        fit.model.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, len(x_valid), batch_size):
                tx = torch.from_numpy(x_valid[start:start + batch_size]).to(device)
                chunks.append(torch.sigmoid(fit.model(tx)).squeeze(-1).cpu().numpy())
        out[valid] = np.concatenate(chunks).astype(np.float64)
    return pd.Series(out, index=features.index, name="long_probability")
