"""Full and rolling CUDA Graph runners for Sundial."""

from __future__ import annotations

import torch
from transformers import DynamicCache

from .cache_utils import get_layer_kv, num_layers
from .flow_sampler import FixedNoiseFlowSampler
from .rope_utils import rebase_rope_keys_minus_one_, rope_minus_one_factors


class _FixedPastCache(DynamicCache):
    def __init__(self, keys, values, model, rope_rebase):
        super().__init__()
        self.key_cache = [value.clone() for value in keys]
        self.value_cache = [value.clone() for value in values]
        self.full_key_cache = [None] * len(keys)
        self.full_value_cache = [None] * len(values)
        self._seen_tokens = self.key_cache[0].shape[-2]
        self.rope_rebase = rope_rebase
        self.cos_one = self.sin_one = None
        if rope_rebase:
            self.cos_one, self.sin_one = rope_minus_one_factors(
                model, self.key_cache[0]
            )
            for key in self.key_cache:
                self.rebase_key_(key)

    def rebase_key_(self, key):
        if self.rope_rebase:
            rebase_rope_keys_minus_one_(key, self.cos_one, self.sin_one)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        key = torch.cat((self.key_cache[layer_idx], key_states), dim=-2)
        value = torch.cat((self.value_cache[layer_idx], value_states), dim=-2)
        self.full_key_cache[layer_idx] = key
        self.full_value_cache[layer_idx] = value
        return key, value

    def get_seq_length(self, layer_idx=0):
        return self.key_cache[layer_idx].shape[-2]


class _FullRefreshCache(DynamicCache):
    def __init__(self, keys, values):
        super().__init__()
        self.key_cache = [torch.empty_like(value) for value in keys]
        self.value_cache = [torch.empty_like(value) for value in values]
        self._seen_tokens = 0

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        self.key_cache[layer_idx].copy_(key_states)
        self.value_cache[layer_idx].copy_(value_states)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx=0):
        return 0


class CudaGraphRollingSundialStep:
    def __init__(self, engine, warmup=3, track_normalization_drift=False):
        if engine.cache is None:
            raise ValueError("call engine.full_refresh() first")
        self.eng = engine
        self.warmup = warmup
        cfg = engine.cfg
        self.x = torch.zeros(cfg.batch_size, engine.patch, device=cfg.device)
        self.mean = engine.mean.clone()
        self.std = engine.std.clone()
        self.track_normalization_drift = track_normalization_drift
        self.patch_sums = self.patch_sumsq = self.normalization_drift = None
        if track_normalization_drift:
            patches = engine.raw_buffer.reshape(
                cfg.batch_size, engine.n_tokens, engine.patch
            )
            self.patch_sums = patches.sum(dim=-1)
            self.patch_sumsq = patches.square().sum(dim=-1)
            self.normalization_drift = torch.zeros((), device=cfg.device)
        keys, values = [], []
        for layer_idx in range(num_layers(engine.cache)):
            key, value = get_layer_kv(engine.cache, layer_idx)
            keys.append(key[:, :, 1:, :])
            values.append(value[:, :, 1:, :])
        self.cache = _FixedPastCache(keys, values, engine.model, cfg.rope_rebase)
        self.sampler = FixedNoiseFlowSampler(
            engine.model.flow_loss,
            cfg.batch_size,
            cfg.num_samples,
            cfg.sampling_steps,
            cfg.device,
            cfg.seed,
            cfg.noise_mode,
        )
        self.graph = self.out = None

    def reset_normalization_state_(self, raw_window):
        if self.track_normalization_drift:
            patches = raw_window.reshape(
                self.eng.cfg.batch_size, self.eng.n_tokens, self.eng.patch
            )
            self.patch_sums.copy_(patches.sum(dim=-1))
            self.patch_sumsq.copy_(patches.square().sum(dim=-1))
            self.normalization_drift.zero_()

    def _update_normalization_drift(self):
        if not self.track_normalization_drift:
            return
        new_sum = self.x.sum(dim=-1, keepdim=True)
        new_sumsq = self.x.square().sum(dim=-1, keepdim=True)
        self.patch_sums.copy_(torch.cat((self.patch_sums[:, 1:], new_sum), dim=-1))
        self.patch_sumsq.copy_(
            torch.cat((self.patch_sumsq[:, 1:], new_sumsq), dim=-1)
        )
        length = self.eng.cfg.context_length
        current_mean = self.patch_sums.sum(dim=-1, keepdim=True) / length
        current_second_moment = self.patch_sumsq.sum(dim=-1, keepdim=True) / length
        current_std = (
            (current_second_moment - current_mean.square()).clamp_min(0).sqrt()
            + 1e-5
        )
        drift = torch.maximum(
            (current_mean - self.mean).abs() / self.std,
            (current_std - self.std).abs() / self.std,
        ).amax()
        self.normalization_drift.copy_(drift)

    def _body(self):
        self._update_normalization_drift()
        normalized = ((self.x - self.mean) / self.std).to(self.eng.cfg.dtype)
        outputs = self.eng.model.model(
            input_ids=normalized,
            past_key_values=self.cache,
            use_cache=True,
            return_dict=True,
        )
        for layer_idx in range(len(self.cache.key_cache)):
            self.cache.key_cache[layer_idx].copy_(
                self.cache.full_key_cache[layer_idx][:, :, 1:, :]
            )
            self.cache.rebase_key_(self.cache.key_cache[layer_idx])
            self.cache.value_cache[layer_idx].copy_(
                self.cache.full_value_cache[layer_idx][:, :, 1:, :]
            )
        prediction = self.sampler.point_forecast(
            outputs.last_hidden_state[:, -1], self.eng.cfg.horizon
        )
        return prediction.float() * self.std + self.mean

    @torch.no_grad()
    def capture(self, preserve_state=True):
        saved_keys = [value.clone() for value in self.cache.key_cache]
        saved_values = [value.clone() for value in self.cache.value_cache]
        saved_stats = None
        if self.track_normalization_drift:
            saved_stats = (
                self.patch_sums.clone(),
                self.patch_sumsq.clone(),
                self.normalization_drift.clone(),
            )
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(self.warmup):
                self._body()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = self._body()
        if preserve_state:
            for dst, src in zip(self.cache.key_cache, saved_keys):
                dst.copy_(src)
            for dst, src in zip(self.cache.value_cache, saved_values):
                dst.copy_(src)
            if saved_stats is not None:
                patch_sums, patch_sumsq, drift = saved_stats
                self.patch_sums.copy_(patch_sums)
                self.patch_sumsq.copy_(patch_sumsq)
                self.normalization_drift.copy_(drift)

    @torch.no_grad()
    def step(self, new_patch):
        self.x.copy_(torch.as_tensor(new_patch, device=self.x.device).view_as(self.x))
        self.graph.replay()
        return self.out


class CudaGraphFullSundialStep:
    def __init__(self, engine, rolling_target=None, warmup=3):
        if engine.cache is None:
            raise ValueError("call engine.full_refresh() first")
        self.eng = engine
        self.rolling_target = rolling_target
        self.warmup = warmup
        cfg = engine.cfg
        self.x = torch.zeros(cfg.batch_size, cfg.context_length, device=cfg.device)
        self.mean = torch.zeros(cfg.batch_size, 1, device=cfg.device)
        self.std = torch.ones(cfg.batch_size, 1, device=cfg.device)
        keys, values = [], []
        for layer_idx in range(num_layers(engine.cache)):
            key, value = get_layer_kv(engine.cache, layer_idx)
            keys.append(key)
            values.append(value)
        self.cache = _FullRefreshCache(keys, values)
        self.sampler = FixedNoiseFlowSampler(
            engine.model.flow_loss,
            cfg.batch_size,
            cfg.num_samples,
            cfg.sampling_steps,
            cfg.device,
            cfg.seed,
            cfg.noise_mode,
        )
        self.graph = self.out = None

    def _body(self):
        mean = self.x.mean(dim=-1, keepdim=True)
        std = self.x.std(dim=-1, keepdim=True, unbiased=False) + 1e-5
        self.mean.copy_(mean)
        self.std.copy_(std)
        normalized = ((self.x - mean) / std).to(self.eng.cfg.dtype)
        outputs = self.eng.model.model(
            input_ids=normalized,
            past_key_values=self.cache,
            use_cache=True,
            return_dict=True,
        )
        target = self.rolling_target
        if target is not None:
            for layer_idx in range(len(self.cache.key_cache)):
                target.cache.key_cache[layer_idx].copy_(
                    self.cache.key_cache[layer_idx][:, :, 1:, :]
                )
                target.cache.rebase_key_(target.cache.key_cache[layer_idx])
                target.cache.value_cache[layer_idx].copy_(
                    self.cache.value_cache[layer_idx][:, :, 1:, :]
                )
            target.mean.copy_(mean)
            target.std.copy_(std)
            target.reset_normalization_state_(self.x)
        prediction = self.sampler.point_forecast(
            outputs.last_hidden_state[:, -1], self.eng.cfg.horizon
        )
        return prediction.float() * std + mean

    @torch.no_grad()
    def capture(self, preserve_target=True):
        target = self.rolling_target
        saved = None
        if target is not None and preserve_target:
            saved = (
                [value.clone() for value in target.cache.key_cache],
                [value.clone() for value in target.cache.value_cache],
                target.mean.clone(),
                target.std.clone(),
                target.patch_sums.clone() if target.track_normalization_drift else None,
                target.patch_sumsq.clone() if target.track_normalization_drift else None,
                target.normalization_drift.clone()
                if target.track_normalization_drift
                else None,
            )
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(self.warmup):
                self._body()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = self._body()
        if saved is not None:
            keys, values, mean, std, patch_sums, patch_sumsq, drift = saved
            for dst, src in zip(target.cache.key_cache, keys):
                dst.copy_(src)
            for dst, src in zip(target.cache.value_cache, values):
                dst.copy_(src)
            target.mean.copy_(mean)
            target.std.copy_(std)
            if target.track_normalization_drift:
                target.patch_sums.copy_(patch_sums)
                target.patch_sumsq.copy_(patch_sumsq)
                target.normalization_drift.copy_(drift)

    @torch.no_grad()
    def step(self, raw_window):
        self.x.copy_(torch.as_tensor(raw_window, device=self.x.device).view_as(self.x))
        self.graph.replay()
        return self.out
