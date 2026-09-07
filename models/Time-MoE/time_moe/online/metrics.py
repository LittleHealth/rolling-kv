"""Forecast accuracy metrics."""

import numpy as np
import torch
from typing import Union

Arraylike = Union[np.ndarray, torch.Tensor]


def _to_np(x: Arraylike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().float().numpy()
    return np.asarray(x, dtype=np.float32)


def compute_mae(pred: Arraylike, target: Arraylike) -> float:
    p, t = _to_np(pred).ravel(), _to_np(target).ravel()
    return float(np.mean(np.abs(p - t)))


def compute_mse(pred: Arraylike, target: Arraylike) -> float:
    p, t = _to_np(pred).ravel(), _to_np(target).ravel()
    return float(np.mean((p - t) ** 2))


def compute_mase(
    pred: Arraylike,
    target: Arraylike,
    naive_scale: float,
) -> float:
    """Mean Absolute Scaled Error.  naive_scale = MAE of a seasonal naive baseline."""
    mae = compute_mae(pred, target)
    return float(mae / (naive_scale + 1e-8))


def compute_smape(pred: Arraylike, target: Arraylike) -> float:
    p, t = _to_np(pred).ravel(), _to_np(target).ravel()
    denom = (np.abs(p) + np.abs(t)) / 2.0 + 1e-8
    return float(np.mean(np.abs(p - t) / denom) * 100.0)


def naive_mae(series: Arraylike, period: int = 1) -> float:
    """MAE of the seasonal naive forecast (shift by `period`)."""
    s = _to_np(series).ravel()
    return float(np.mean(np.abs(s[period:] - s[:-period])))
