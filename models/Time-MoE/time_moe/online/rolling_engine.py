"""
Rolling KV-cache engine for online time-series forecasting with TimeMoE.

Architecture
------------
  raw_buffer      : circular buffer of shape [B, L] for the last `context_length` raw values
  observed_cache  : DynamicCache for the normalised context (batched: K/V are [B, H, S, D])
  norms           : list of B FrozenNormalisers – one per series, frozen at each full_refresh
  step_count      : how many online steps have been taken since init

Three update strategies (applied in priority order each step):

  full_refresh   : re-encode the whole raw buffer                  (every K steps)
  tail_recompute : keep prefix cache, re-encode the last T values   (every M steps, if not K)
  fast_update    : drop oldest cache entry, append new value        (all other steps)

After any update, `last_logits` holds the model's H-step prediction in
normalised space for the most recent position.  `forecast()` denormalises it.

Batch support
-------------
Set `EngineConfig.batch_size > 1` to process B independent series simultaneously.
Each series has its own FrozenNormaliser; the model receives [B, ...] input tensors.
`step()` accepts either a Python float (B=1 backward-compat) or a [B] tensor.
`forecast()` returns [H] for B=1, [B, H] for B>1.

Position-ID convention
----------------------
We use window-relative positions (model's default: past_kv_length + [0..S-1]).
This avoids absolute positions > L-1 which would exceed the rotary embedding
bounds (cos/sin computed for seq_len = kv_seq_len only).  A 1-step position
approximation is introduced for the last cached key in fast_update, but it is
bounded and corrected by full_refresh.
"""

import torch
from dataclasses import dataclass, field
from typing import Optional, List

from transformers import DynamicCache

from .cache_utils import slice_cache, get_cache_seq_len
from .normalizer import FrozenNormalizer


@dataclass
class EngineConfig:
    context_length: int = 512
    prediction_length: int = 64
    tail_length: int = 128
    tail_recompute_every: int = 32
    full_refresh_every: int = 256
    batch_size: int = 1
    device: str = "cpu"
    dtype: torch.dtype = torch.float32


@dataclass
class StepRecord:
    step_id: int
    method: str
    latency_ms: float
    prediction: List[float] = field(default_factory=list)


class RollingTimeMoEEngine:
    def __init__(self, model, cfg: EngineConfig):
        self.model = model
        self.cfg = cfg
        self.B = cfg.batch_size
        self.model.eval()

        self.norms: List[FrozenNormalizer] = [FrozenNormalizer() for _ in range(self.B)]
        self.raw_buffer: Optional[torch.Tensor] = None  # [B, L]
        self.observed_cache: Optional[DynamicCache] = None
        self.last_logits: Optional[torch.Tensor] = None  # [B, 1, H_model]
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _normalize_batch(self, raw: torch.Tensor) -> torch.Tensor:
        """raw: [B, T] → normalized [B, T] using per-element frozen stats."""
        return torch.stack([self.norms[b].normalize(raw[b]) for b in range(self.B)])

    def _denormalize_batch(self, normed: torch.Tensor) -> torch.Tensor:
        """normed: [B, H] → original scale [B, H]."""
        return torch.stack([self.norms[b].denormalize(normed[b]) for b in range(self.B)])

    def _to_batch_col(self, x_news) -> torch.Tensor:
        """Convert scalar / list / [B] / [B,1] to [B, 1] float32 on device."""
        if isinstance(x_news, (int, float)):
            return torch.full((self.B, 1), x_news, dtype=torch.float32, device=self.cfg.device)
        t = torch.as_tensor(x_news, dtype=torch.float32, device=self.cfg.device)
        return t.view(self.B, 1)

    # ------------------------------------------------------------------
    # Core update operations
    # ------------------------------------------------------------------

    def full_refresh(self, raw_windows: torch.Tensor) -> None:
        """Re-encode the whole window and reset normalisation stats.

        Args:
            raw_windows: [L] (batch_size=1) or [B, L].
        """
        if raw_windows.dim() == 1:
            raw_windows = raw_windows.unsqueeze(0)  # [1, L] for B=1
        raw_windows = raw_windows.to(dtype=torch.float32, device=self.cfg.device)

        self.raw_buffer = raw_windows.clone()
        for b in range(self.B):
            self.norms[b].fit(raw_windows[b])

        normed = self._normalize_batch(raw_windows)  # [B, L]

        with torch.no_grad():
            out = self.model(
                input_ids=normed.to(self.cfg.dtype),
                use_cache=True,
                return_dict=True,
                max_horizon_length=self.cfg.prediction_length,
            )

        self.observed_cache = out.past_key_values
        self.last_logits = out.logits[:, -1:, :].to(torch.float32)

    def fast_update(self, x_news) -> None:
        """Drop the oldest cache entry and append a new observation per series.

        x_news: float (B=1), or tensor [B].
        """
        x_new_t = self._to_batch_col(x_news)  # [B, 1]
        self.raw_buffer = torch.cat([self.raw_buffer[:, 1:], x_new_t], dim=1)

        x_normed = self._normalize_batch(x_new_t)  # [B, 1]
        sliced = slice_cache(self.observed_cache, start=1)  # length L-1

        # No explicit position_ids: model defaults to [past_kv_length] = [L-1],
        # staying within rotary embedding bounds [0, kv_seq_len).
        with torch.no_grad():
            out = self.model(
                input_ids=x_normed.to(self.cfg.dtype),
                past_key_values=sliced,
                use_cache=True,
                return_dict=True,
                max_horizon_length=self.cfg.prediction_length,
            )

        self.observed_cache = out.past_key_values
        self.last_logits = out.logits[:, -1:, :].to(torch.float32)

    def tail_recompute(self) -> None:
        """Re-encode the last `tail_length` steps using the prefix cache."""
        T = self.cfg.tail_length
        prefix_len = self.cfg.context_length - T

        prefix_cache = slice_cache(self.observed_cache, end=prefix_len)

        tail_raw = self.raw_buffer[:, -T:]  # [B, T]
        tail_normed = self._normalize_batch(tail_raw).to(self.cfg.dtype)

        with torch.no_grad():
            out = self.model(
                input_ids=tail_normed,
                past_key_values=prefix_cache,
                use_cache=True,
                return_dict=True,
                max_horizon_length=self.cfg.prediction_length,
            )

        self.observed_cache = out.past_key_values
        self.last_logits = out.logits[:, -1:, :].to(torch.float32)

    # ------------------------------------------------------------------
    # Forecast
    # ------------------------------------------------------------------

    def forecast(self) -> torch.Tensor:
        """Return H-step prediction in original scale.

        Returns [H] for batch_size=1, [B, H] for batch_size>1.
        """
        H = self.cfg.prediction_length
        logits = self.last_logits
        if logits.dim() == 3:
            logits = logits[:, -1, :]  # [B, H_model]
        pred_normed = logits[:, :H]  # [B, H]
        result = self._denormalize_batch(pred_normed).cpu()  # [B, H]
        if self.B == 1:
            return result.squeeze(0)  # [H] for backward compat
        return result  # [B, H]

    # ------------------------------------------------------------------
    # Main online step
    # ------------------------------------------------------------------

    def step(self, x_news) -> torch.Tensor:
        """Process one new observation per series and return H-step forecasts.

        Args:
            x_news: float (batch_size=1) or tensor [B].

        Returns:
            [H] if batch_size=1, else [B, H].
        """
        self._step_count += 1
        s = self._step_count
        fr_every = self.cfg.full_refresh_every
        tr_every = self.cfg.tail_recompute_every

        if fr_every > 0 and s % fr_every == 0:
            x_new_t = self._to_batch_col(x_news)
            self.raw_buffer = torch.cat([self.raw_buffer[:, 1:], x_new_t], dim=1)
            self.full_refresh(self.raw_buffer)
        elif tr_every > 0 and s % tr_every == 0:
            self.fast_update(x_news)
            self.tail_recompute()
        else:
            self.fast_update(x_news)

        return self.forecast()
