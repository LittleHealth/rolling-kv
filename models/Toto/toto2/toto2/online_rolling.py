"""Online sliding-window KV cache for Toto 2.0.

The upstream cache accelerates block decoding *within* one forecast.  This
module keeps the time-attention cache alive across successive observed
patches.  Variate-attention layers are recomputed for the new time slice.
"""

from __future__ import annotations

import dataclasses

import torch
from einops import rearrange, repeat

from .model import ExtrapolatableRotaryProjection, KVCache, Toto2Model


@dataclasses.dataclass
class Toto2RollingConfig:
    context_length: int = 4096
    horizon: int = 32
    batch_size: int = 1
    num_variates: int = 1
    device: str = "cuda"
    dtype: torch.dtype = torch.float32
    xpos_rebase: bool = True


class RollingToto2Engine:
    """Patch-aligned eager full/rolling engine.

    The new patch is normalized with the exact right-edge statistics of the
    current raw window.  Survivor KV values retain their historical causal
    normalization until the next full refresh.
    """

    def __init__(self, model: Toto2Model, cfg: Toto2RollingConfig):
        self.model = model.eval()
        self.cfg = cfg
        self.patch = int(model.config.patch_size)
        if cfg.context_length <= 0 or cfg.context_length % self.patch:
            raise ValueError(f"context_length must be a multiple of {self.patch}")
        if not 1 <= cfg.horizon <= self.patch:
            raise ValueError(
                f"online engine currently supports horizon in [1, {self.patch}]"
            )
        self.num_patches = cfg.context_length // self.patch
        self.raw_buffer = None
        self.cache = KVCache(model.num_time_layers, self.num_patches).to(cfg.device)
        self.last_loc = self.last_scale = self.last_prediction = None
        self._time_layers = [
            layer
            for index, layer in enumerate(model.transformer.layers)
            if not model.transformer._if_variate_layer(index)
        ]

    def _as_window(self, value):
        tensor = torch.as_tensor(value, device=self.cfg.device, dtype=self.cfg.dtype)
        expected = (
            self.cfg.batch_size,
            self.cfg.num_variates,
            self.cfg.context_length,
        )
        if tensor.shape != expected:
            raise ValueError(f"expected window {expected}, got {tuple(tensor.shape)}")
        return tensor

    def _as_patch(self, value):
        tensor = torch.as_tensor(value, device=self.cfg.device, dtype=self.cfg.dtype)
        expected = (self.cfg.batch_size, self.cfg.num_variates, self.patch)
        if tensor.shape != expected:
            raise ValueError(f"expected patch {expected}, got {tuple(tensor.shape)}")
        return tensor

    def _group_ids(self, seq_len):
        series_ids = torch.arange(
            self.cfg.num_variates, device=self.cfg.device, dtype=torch.long
        ).expand(self.cfg.batch_size, -1)
        return repeat(series_ids, "b v -> b v s", s=seq_len)

    def _embed_full_window(self, window):
        mask = torch.ones_like(window, dtype=torch.bool)
        scaled, loc, scale = self.model.scaler(window, mask)
        embedded = self.model._embed_patches(scaled.asinh(), mask, self.patch)
        return embedded, loc[..., -1:], scale[..., -1:]

    def _right_edge_stats(self, window):
        # This is exactly the last element produced by PatchedCausalStdScaler:
        # sample variance (correction=1), clamped to minimum_scale.
        hp = window.double()
        loc = hp.mean(dim=-1, keepdim=True)
        centered = hp - loc
        denom = max(self.cfg.context_length - 1, 1)
        scale = torch.sqrt(centered.square().sum(dim=-1, keepdim=True) / denom)
        scale = scale.clamp_min(self.model.scaler.minimum_scale)
        return loc.to(window.dtype), scale.to(window.dtype)

    def _decode(self, hidden, loc, scale):
        quantiles = self.model.output_head(hidden, q=None)
        quantiles = quantiles.sinh() * scale.unsqueeze(0) + loc.unsqueeze(0)
        quantiles = self.model._clamp_nonfinite(quantiles).sort(dim=0).values
        median = quantiles[self.model.output_head.knots.index(0.5)]
        return median[..., : self.cfg.horizon]

    def _rebase_survivor_key_(self, key, layer):
        projection = layer.attn.qk_proj
        if projection is None:
            return
        key_projection = projection.key_proj
        start, width, _ = projection.split_sizes
        if width == 0:
            return
        rotated = key[..., start : start + width]
        cos = key_projection.cos[1].to(rotated.dtype)
        sin = -key_projection.sin[1].to(rotated.dtype)
        rebased = cos * rotated + sin * key_projection._rotate(rotated)
        if isinstance(key_projection, ExtrapolatableRotaryProjection):
            scale = repeat(
                key_projection.xpos_base_scale,
                "d -> (d r)",
                r=2,
            ).to(rebased.dtype)
            rebased = rebased * scale.pow(1.0 / key_projection.xpos_scale_base)
        rotated.copy_(rebased)

    def _roll_cache_(self):
        keep = self.num_patches - 1
        for cache_layer, model_layer in zip(self.cache.cache_layers, self._time_layers):
            cache_layer.keys[:, :, :keep].copy_(
                cache_layer.keys[:, :, 1 : self.num_patches].clone()
            )
            cache_layer.values[:, :, :keep].copy_(
                cache_layer.values[:, :, 1 : self.num_patches].clone()
            )
            if self.cfg.xpos_rebase:
                self._rebase_survivor_key_(cache_layer.keys[:, :, :keep], model_layer)
            cache_layer._position.fill_(keep)

    @torch.no_grad()
    def full_refresh(self, raw_window):
        window = self._as_window(raw_window)
        self.raw_buffer = window.clone()
        embedded, loc, scale = self._embed_full_window(window)
        self.cache.reset()
        hidden = self.model.transformer(
            embedded,
            group_ids=self._group_ids(self.num_patches),
            kv_cache=self.cache,
            kv_read_len=self.num_patches,
            has_missing_values=False,
        )
        self.last_loc, self.last_scale = loc, scale
        self.last_prediction = self._decode(hidden[..., -1, :], loc, scale)
        return self.last_prediction

    @torch.no_grad()
    def fast_update(self, new_patch):
        if self.raw_buffer is None:
            raise RuntimeError("call full_refresh() first")
        patch = self._as_patch(new_patch)
        self.raw_buffer = torch.cat((self.raw_buffer[..., self.patch :], patch), -1)
        loc, scale = self.last_loc, self.last_scale
        normalized = ((patch - loc) / scale).asinh()
        mask = torch.ones_like(patch, dtype=torch.bool)
        embedded = self.model._embed_patches(normalized, mask, self.patch)
        self._roll_cache_()
        hidden = self.model.transformer(
            embedded,
            time_ids=torch.tensor(
                [self.num_patches - 1], device=self.cfg.device, dtype=torch.long
            ),
            group_ids=self._group_ids(1),
            kv_cache=self.cache,
            kv_read_len=self.num_patches,
            has_missing_values=False,
        )
        self.last_prediction = self._decode(hidden[..., -1, :], loc, scale)
        return self.last_prediction


class CudaGraphRollingToto2Step:
    """CUDA Graph replay for one observed patch and one output patch."""

    def __init__(self, engine: RollingToto2Engine, warmup: int = 3):
        if engine.raw_buffer is None:
            raise ValueError("call full_refresh() before graph capture")
        self.engine = engine
        self.warmup = warmup
        self.patch = torch.zeros(
            engine.cfg.batch_size,
            engine.cfg.num_variates,
            engine.patch,
            device=engine.cfg.device,
            dtype=engine.cfg.dtype,
        )
        self.patch_mask = torch.ones_like(self.patch, dtype=torch.bool)
        self.time_ids = torch.tensor(
            [engine.num_patches - 1], device=engine.cfg.device, dtype=torch.long
        )
        self.group_ids = engine._group_ids(1)
        self.graph = None
        self.output = None

    def _body(self):
        eng = self.engine
        eng.raw_buffer.copy_(torch.cat((eng.raw_buffer[..., eng.patch :], self.patch), -1))
        loc, scale = eng.last_loc, eng.last_scale
        embedded = eng.model._embed_patches(
            ((self.patch - loc) / scale).asinh(),
            self.patch_mask,
            eng.patch,
        )
        eng._roll_cache_()
        hidden = eng.model.transformer(
            embedded,
            time_ids=self.time_ids,
            group_ids=self.group_ids,
            kv_cache=eng.cache,
            kv_read_len=eng.num_patches,
            has_missing_values=False,
        )
        return eng._decode(hidden[..., -1, :], loc, scale)

    @torch.no_grad()
    def capture(self):
        eng = self.engine
        saved_raw = eng.raw_buffer.clone()
        saved_keys = [layer.keys.clone() for layer in eng.cache.cache_layers]
        saved_values = [layer.values.clone() for layer in eng.cache.cache_layers]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(self.warmup):
                self._body()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = self._body()
        eng.raw_buffer.copy_(saved_raw)
        for cache_layer, key, value in zip(eng.cache.cache_layers, saved_keys, saved_values):
            cache_layer.keys.copy_(key)
            cache_layer.values.copy_(value)
            cache_layer._position.fill_(eng.num_patches)

    @torch.no_grad()
    def step(self, new_patch):
        self.patch.copy_(torch.as_tensor(new_patch, device=self.patch.device).view_as(self.patch))
        self.graph.replay()
        return self.output


class CudaGraphFullToto2Step:
    """CUDA Graph full-window recompute baseline."""

    def __init__(self, engine: RollingToto2Engine, warmup: int = 3):
        self.engine = engine
        self.warmup = warmup
        self.window = torch.zeros(
            engine.cfg.batch_size,
            engine.cfg.num_variates,
            engine.cfg.context_length,
            device=engine.cfg.device,
            dtype=engine.cfg.dtype,
        )
        self.window.copy_(engine.raw_buffer)
        self.window_mask = torch.ones_like(self.window, dtype=torch.bool)
        self.group_ids = engine._group_ids(engine.num_patches)
        self.graph = None
        self.output = None

    def _body(self):
        eng = self.engine
        scaled, loc, scale = eng.model.scaler(self.window, self.window_mask)
        embedded = eng.model._embed_patches(
            scaled.asinh(), self.window_mask, eng.patch
        )
        for cache_layer in eng.cache.cache_layers:
            cache_layer._position.zero_()
        hidden = eng.model.transformer(
            embedded,
            group_ids=self.group_ids,
            kv_cache=eng.cache,
            kv_read_len=eng.num_patches,
            has_missing_values=False,
        )
        eng.last_loc.copy_(loc[..., -1:])
        eng.last_scale.copy_(scale[..., -1:])
        eng.raw_buffer.copy_(self.window)
        return eng._decode(
            hidden[..., -1, :], loc[..., -1:], scale[..., -1:]
        )

    @torch.no_grad()
    def capture(self):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(self.warmup):
                self._body()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = self._body()

    @torch.no_grad()
    def step(self, raw_window):
        self.window.copy_(
            torch.as_tensor(raw_window, device=self.window.device).view_as(self.window)
        )
        self.graph.replay()
        return self.output
