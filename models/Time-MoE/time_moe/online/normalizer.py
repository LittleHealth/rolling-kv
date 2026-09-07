"""
Rolling normalization that freezes mean/std between full refreshes.

TimeMoE expects per-sample mean/std normalization before input, and
inverse normalization after output. When using a rolling KV cache we
cannot re-normalize the historical context (the KV values are already
baked with the old stats), so we freeze the stats at each full refresh
and reuse them until the next refresh.
"""

import numpy as np
import torch


class FrozenNormalizer:
    def __init__(self, eps: float = 1e-8):
        self.eps = eps
        self.mean: float = 0.0
        self.std: float = 1.0

    def fit(self, window: torch.Tensor) -> None:
        """Compute and freeze stats from the current window (1-D tensor)."""
        m = window.float().mean().item()
        s = window.float().std().item()
        self.mean = m
        self.std = max(s, self.eps)

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x.float() - self.mean) / self.std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x.float() * self.std + self.mean

    @property
    def stats(self):
        return self.mean, self.std
