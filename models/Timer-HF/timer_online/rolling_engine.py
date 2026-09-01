"""Patch-aligned rolling KV cache for Timer-base-84M.

Timer maps every non-overlapping 96-point patch to one Transformer token.
Rolling therefore advances by exactly one patch: the oldest cached token is
evicted and only the new patch is encoded.  Refreshes recompute the complete
window and reset the frozen normalization statistics.
"""

from __future__ import annotations

import dataclasses

import torch
from transformers import DynamicCache

from .cache_utils import get_layer_kv, num_layers, slice_cache
from .rope_utils import rebase_rope_keys_minus_one_, rope_minus_one_factors


@dataclasses.dataclass
class TimerRollingConfig:
    context_length: int = 2880
    horizon: int = 96
    full_refresh_every: int = 0
    batch_size: int = 1
    device: str = "cuda"
    dtype: torch.dtype = torch.float32
    rope_rebase: bool = True


class RollingTimerEngine:
    """Online Timer forecaster with frozen-normalization rolling KV reuse."""

    def __init__(self, model, cfg: TimerRollingConfig):
        self.model = model
        self.cfg = cfg
        self.patch = int(model.config.input_token_len)
        self.max_output = max(model.config.output_token_lens)
        if cfg.context_length < self.patch or cfg.context_length % self.patch:
            raise ValueError(
                f"context_length must be a positive multiple of {self.patch}"
            )
        if not 1 <= cfg.horizon <= self.max_output:
            raise ValueError(f"horizon must be in [1, {self.max_output}]")
        self.n_tokens = cfg.context_length // self.patch
        self.raw_buffer: torch.Tensor | None = None
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None
        self.cache: DynamicCache | None = None
        self.last_prediction: torch.Tensor | None = None
        self.n_updates = 0
        self.model.eval()

    def _as_window(self, raw_window) -> torch.Tensor:
        value = torch.as_tensor(raw_window, device=self.cfg.device, dtype=torch.float32)
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.shape != (self.cfg.batch_size, self.cfg.context_length):
            raise ValueError(
                f"expected [{self.cfg.batch_size}, {self.cfg.context_length}], "
                f"got {list(value.shape)}"
            )
        return value

    def _as_patch(self, new_patch) -> torch.Tensor:
        value = torch.as_tensor(new_patch, device=self.cfg.device, dtype=torch.float32)
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.shape != (self.cfg.batch_size, self.patch):
            raise ValueError(
                f"expected [{self.cfg.batch_size}, {self.patch}], got {list(value.shape)}"
            )
        return value

    @torch.no_grad()
    def full_refresh(self, raw_window) -> torch.Tensor:
        window = self._as_window(raw_window)
        self.raw_buffer = window.clone()
        self.mean = window.mean(dim=-1, keepdim=True)
        self.std = window.std(dim=-1, keepdim=True).clamp_min(1e-8)
        normalized = ((window - self.mean) / self.std).to(self.cfg.dtype)
        result = self.model(
            input_ids=normalized,
            past_key_values=DynamicCache(),
            use_cache=True,
            return_dict=True,
            max_output_length=self.cfg.horizon,
            revin=False,
        )
        self.cache = result.past_key_values
        self.last_prediction = result.logits.float() * self.std + self.mean
        return self.last_prediction

    @torch.no_grad()
    def fast_update(self, new_patch) -> torch.Tensor:
        if self.cache is None:
            raise RuntimeError("call full_refresh() before fast_update()")
        patch = self._as_patch(new_patch)
        self.raw_buffer = torch.cat((self.raw_buffer[:, self.patch :], patch), dim=-1)
        normalized = ((patch - self.mean) / self.std).to(self.cfg.dtype)
        survivors = slice_cache(self.cache, start=1)
        if self.cfg.rope_rebase:
            for layer_idx in range(num_layers(survivors)):
                key, _ = get_layer_kv(survivors, layer_idx)
                cos_one, sin_one = rope_minus_one_factors(self.model, key)
                rebase_rope_keys_minus_one_(key, cos_one, sin_one)
        result = self.model(
            input_ids=normalized,
            past_key_values=survivors,
            use_cache=True,
            return_dict=True,
            max_output_length=self.cfg.horizon,
            revin=False,
        )
        self.cache = result.past_key_values
        self.last_prediction = result.logits.float() * self.std + self.mean
        return self.last_prediction

    @torch.no_grad()
    def step_patch(self, new_patch) -> torch.Tensor:
        self.n_updates += 1
        patch = self._as_patch(new_patch)
        refresh = self.cfg.full_refresh_every
        if refresh > 0 and self.n_updates % refresh == 0:
            window = torch.cat((self.raw_buffer[:, self.patch :], patch), dim=-1)
            return self.full_refresh(window)
        return self.fast_update(patch)

    def forecast(self) -> torch.Tensor:
        if self.last_prediction is None:
            raise RuntimeError("call full_refresh() first")
        return self.last_prediction

    @property
    def cache_length(self) -> int:
        return 0 if self.cache is None else int(self.cache.get_seq_length())
