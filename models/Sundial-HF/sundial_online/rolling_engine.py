"""Patch-aligned eager rolling cache for Sundial-base-128m."""

from __future__ import annotations

import dataclasses

import torch
from transformers import DynamicCache

from .cache_utils import get_layer_kv, num_layers, slice_cache
from .flow_sampler import FixedNoiseFlowSampler
from .rope_utils import rebase_rope_keys_minus_one_, rope_minus_one_factors


@dataclasses.dataclass
class SundialRollingConfig:
    context_length: int = 2880
    horizon: int = 96
    batch_size: int = 1
    num_samples: int = 1
    sampling_steps: int = 50
    seed: int = 7
    noise_mode: str = "antithetic"
    device: str = "cuda"
    dtype: torch.dtype = torch.float32
    rope_rebase: bool = True


class RollingSundialEngine:
    def __init__(self, model, cfg: SundialRollingConfig):
        self.model = model
        self.cfg = cfg
        self.patch = int(model.config.input_token_len)
        self.max_output = int(model.config.output_token_lens[-1])
        if cfg.context_length < self.patch or cfg.context_length % self.patch:
            raise ValueError(
                f"context_length must be a positive multiple of {self.patch}"
            )
        if not 1 <= cfg.horizon <= self.max_output:
            raise ValueError(f"horizon must be in [1, {self.max_output}]")
        self.n_tokens = cfg.context_length // self.patch
        self.sampler = FixedNoiseFlowSampler(
            model.flow_loss,
            cfg.batch_size,
            cfg.num_samples,
            cfg.sampling_steps,
            cfg.device,
            cfg.seed,
            cfg.noise_mode,
        )
        self.raw_buffer = self.mean = self.std = self.cache = None
        self.last_prediction = None
        self.model.eval()

    def _tensor(self, value, length):
        result = torch.as_tensor(value, device=self.cfg.device, dtype=torch.float32)
        if result.ndim == 1:
            result = result.unsqueeze(0)
        expected = (self.cfg.batch_size, length)
        if result.shape != expected:
            raise ValueError(f"expected {expected}, got {tuple(result.shape)}")
        return result

    def _normalize_window(self, window):
        self.mean = window.mean(dim=-1, keepdim=True)
        self.std = window.std(dim=-1, keepdim=True, unbiased=False) + 1e-5
        return ((window - self.mean) / self.std).to(self.cfg.dtype)

    def _decode(self, hidden):
        normalized = self.sampler.point_forecast(hidden, self.cfg.horizon)
        return normalized.float() * self.std + self.mean

    @torch.no_grad()
    def full_refresh(self, raw_window):
        window = self._tensor(raw_window, self.cfg.context_length)
        self.raw_buffer = window.clone()
        normalized = self._normalize_window(window)
        outputs = self.model.model(
            input_ids=normalized,
            past_key_values=DynamicCache(),
            use_cache=True,
            return_dict=True,
        )
        self.cache = outputs.past_key_values
        self.last_prediction = self._decode(outputs.last_hidden_state[:, -1])
        return self.last_prediction

    @torch.no_grad()
    def fast_update(self, new_patch):
        if self.cache is None:
            raise RuntimeError("call full_refresh() first")
        patch = self._tensor(new_patch, self.patch)
        self.raw_buffer = torch.cat((self.raw_buffer[:, self.patch :], patch), dim=-1)
        normalized = ((patch - self.mean) / self.std).to(self.cfg.dtype)
        survivors = slice_cache(self.cache, start=1)
        if self.cfg.rope_rebase:
            for layer_idx in range(num_layers(survivors)):
                key, _ = get_layer_kv(survivors, layer_idx)
                cos_one, sin_one = rope_minus_one_factors(self.model, key)
                rebase_rope_keys_minus_one_(key, cos_one, sin_one)
        outputs = self.model.model(
            input_ids=normalized,
            past_key_values=survivors,
            use_cache=True,
            return_dict=True,
        )
        self.cache = outputs.past_key_values
        self.last_prediction = self._decode(outputs.last_hidden_state[:, -1])
        return self.last_prediction
