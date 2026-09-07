"""CUDA Graph runner for TimeMoE's rolling single-token fast path.

The eager MoE dispatcher uses ``torch.where`` and therefore produces a
data-dependent number of rows per expert.  The graph runner enables the
model's fixed-shape dispatch branch and keeps a fixed ``L-1`` KV cache whose
survivors are updated in-place after every replay.

The rolling graph implements the B0 step.  ``CudaGraphFullTimeMoEStep`` adds a
fixed-window full graph that can install refreshed KV and normalization state
directly into the rolling graph, enabling periodic refresh without eager model
execution or recapture.
"""

from __future__ import annotations

import torch
from transformers import DynamicCache

from .cache_utils import _get_layer_kv, _num_layers


def set_static_moe_dispatch(model, enabled: bool) -> None:
    """Toggle the graph-safe MoE branch on every sparse expert layer."""
    for module in model.modules():
        if hasattr(module, "use_static_dispatch"):
            module.use_static_dispatch = enabled


class _FixedPastCache(DynamicCache):
    """DynamicCache interface backed by fixed-address survivor tensors.

    ``update`` returns the concatenated survivor + new-token cache required by
    attention, but deliberately does not replace ``key_cache``/``value_cache``.
    The graph body copies the newest ``L-1`` entries back into those buffers at
    the end of the step, making the state advance inside graph replay.
    """

    def __init__(self, keys, values):
        super().__init__()
        if not hasattr(self, "key_cache"):
            raise RuntimeError(
                "TimeMoE CUDA Graph currently requires transformers' legacy "
                "DynamicCache layout (verified with 4.45.2)."
            )
        self.key_cache = [x.clone() for x in keys]
        self.value_cache = [x.clone() for x in values]
        self.full_key_cache = [None] * len(keys)
        self.full_value_cache = [None] * len(values)
        self._seen_tokens = self.key_cache[0].shape[-2] if self.key_cache else 0

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        full_key = torch.cat((self.key_cache[layer_idx], key_states), dim=-2)
        full_value = torch.cat((self.value_cache[layer_idx], value_states), dim=-2)
        self.full_key_cache[layer_idx] = full_key
        self.full_value_cache[layer_idx] = full_value
        return full_key, full_value

    def get_seq_length(self, layer_idx=0):
        if not self.key_cache or layer_idx >= len(self.key_cache):
            return 0
        return self.key_cache[layer_idx].shape[-2]


class _FullRefreshCache(DynamicCache):
    """Fixed-address cache written from scratch by every full replay."""

    def __init__(self, keys, values):
        super().__init__()
        if not hasattr(self, "key_cache"):
            raise RuntimeError(
                "TimeMoE CUDA Graph currently requires transformers 4.45-style DynamicCache."
            )
        self.key_cache = [torch.empty_like(x) for x in keys]
        self.value_cache = [torch.empty_like(x) for x in values]
        self._seen_tokens = 0

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        self.key_cache[layer_idx].copy_(key_states)
        self.value_cache[layer_idx].copy_(value_states)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx=0):
        # Each replay is a fresh prefill, even though storage is persistent.
        return 0


class CudaGraphRollingTimeMoEStep:
    """Capture and replay one rolling TimeMoE update.

    The engine must already have been initialized with ``full_refresh``.  The
    captured result is returned on GPU as ``[B, H]`` in the original scale.
    """

    def __init__(self, engine, warmup: int = 3):
        cfg = engine.cfg
        if cfg.device != "cuda":
            raise ValueError("TimeMoE CUDA Graph requires device='cuda'")
        if cfg.full_refresh_every or cfg.tail_recompute_every:
            raise ValueError(
                "TimeMoE CUDA Graph supports the B0 fast path only "
                "(full_refresh_every=tail_recompute_every=0)."
            )
        if engine.observed_cache is None:
            raise ValueError("call engine.full_refresh() before constructing the graph runner")

        self.eng = engine
        self.model = engine.model
        self.warmup = warmup
        self.x = torch.zeros(cfg.batch_size, 1, device=cfg.device, dtype=torch.float32)
        self.mean = torch.tensor(
            [norm.mean for norm in engine.norms], device=cfg.device, dtype=torch.float32
        )
        self.std = torch.tensor(
            [norm.std for norm in engine.norms], device=cfg.device, dtype=torch.float32
        )

        keys, values = [], []
        for layer_idx in range(_num_layers(engine.observed_cache)):
            key, value = _get_layer_kv(engine.observed_cache, layer_idx)
            keys.append(key[:, :, 1:, :])
            values.append(value[:, :, 1:, :])
        self.cache = _FixedPastCache(keys, values)
        self.graph: torch.cuda.CUDAGraph | None = None
        self.out: torch.Tensor | None = None

    def _body(self):
        cfg = self.eng.cfg
        normed = ((self.x - self.mean[:, None]) / self.std[:, None]).to(cfg.dtype)
        # Pass [B, 1, input_size] so TimeMoeModel does not mutate tensor metadata
        # with its compatibility ``unsqueeze_`` path.
        model_input = normed.unsqueeze(-1)
        result = self.model(
            input_ids=model_input,
            past_key_values=self.cache,
            use_cache=True,
            return_dict=True,
            max_horizon_length=cfg.prediction_length,
        )

        # Advance survivors for the next replay without changing addresses.
        for layer_idx in range(len(self.cache.key_cache)):
            self.cache.key_cache[layer_idx].copy_(
                self.cache.full_key_cache[layer_idx][:, :, 1:, :]
            )
            self.cache.value_cache[layer_idx].copy_(
                self.cache.full_value_cache[layer_idx][:, :, 1:, :]
            )

        logits = result.logits[:, -1, : cfg.prediction_length].float()
        return logits * self.std[:, None] + self.mean[:, None]

    @torch.no_grad()
    def capture(self, preserve_state: bool = True) -> None:
        """Warm up and capture, optionally restoring the pre-warmup history."""
        set_static_moe_dispatch(self.model, True)
        saved_keys = [x.clone() for x in self.cache.key_cache] if preserve_state else None
        saved_values = [x.clone() for x in self.cache.value_cache] if preserve_state else None

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

    @torch.no_grad()
    def step(self, new_values) -> torch.Tensor:
        """Copy ``[B]`` new observations and replay the captured step."""
        if self.graph is None:
            raise RuntimeError("call capture() first")
        if self.x.numel() == 1 and not torch.is_tensor(new_values) and not isinstance(
            new_values, (list, tuple)
        ):
            self.x.fill_(float(new_values))
        else:
            self.x.copy_(torch.as_tensor(new_values, device=self.x.device).view_as(self.x))
        self.graph.replay()
        return self.out


class CudaGraphFullTimeMoEStep:
    """Full-window recomputation captured as a CUDA Graph.

    If ``rolling_target`` is supplied, the full graph also copies its freshly
    computed survivor KV and normalization statistics into the rolling graph's
    fixed buffers.  A refresh therefore needs only one full-graph replay; the
    next rolling replay continues from the refreshed state without recapture.
    """

    def __init__(self, engine, rolling_target=None, warmup: int = 3):
        cfg = engine.cfg
        if cfg.device != "cuda":
            raise ValueError("TimeMoE CUDA Graph requires device='cuda'")
        if engine.observed_cache is None:
            raise ValueError("call engine.full_refresh() before constructing the full graph")
        self.eng = engine
        self.model = engine.model
        self.rolling_target = rolling_target
        self.warmup = warmup
        self.x = torch.zeros(
            cfg.batch_size, cfg.context_length, device=cfg.device, dtype=torch.float32
        )
        self.mean = torch.zeros(cfg.batch_size, device=cfg.device)
        self.std = torch.ones(cfg.batch_size, device=cfg.device)
        keys, values = [], []
        for layer_idx in range(_num_layers(engine.observed_cache)):
            key, value = _get_layer_kv(engine.observed_cache, layer_idx)
            keys.append(key)
            values.append(value)
        self.cache = _FullRefreshCache(keys, values)
        self.graph: torch.cuda.CUDAGraph | None = None
        self.out: torch.Tensor | None = None

    def _body(self):
        cfg = self.eng.cfg
        mean = self.x.mean(dim=1)
        std = self.x.std(dim=1).clamp_min(1e-8)
        self.mean.copy_(mean)
        self.std.copy_(std)
        normed = ((self.x - mean[:, None]) / std[:, None]).to(cfg.dtype)
        result = self.model(
            input_ids=normed.unsqueeze(-1),
            past_key_values=self.cache,
            use_cache=True,
            return_dict=True,
            max_horizon_length=cfg.prediction_length,
        )

        target = self.rolling_target
        if target is not None:
            for layer_idx in range(len(self.cache.key_cache)):
                target.cache.key_cache[layer_idx].copy_(
                    self.cache.key_cache[layer_idx][:, :, 1:, :]
                )
                target.cache.value_cache[layer_idx].copy_(
                    self.cache.value_cache[layer_idx][:, :, 1:, :]
                )
            target.mean.copy_(mean)
            target.std.copy_(std)

        logits = result.logits[:, -1, : cfg.prediction_length].float()
        return logits * std[:, None] + mean[:, None]

    @torch.no_grad()
    def capture(self, preserve_target: bool = True) -> None:
        set_static_moe_dispatch(self.model, True)
        target = self.rolling_target
        saved = None
        if target is not None and preserve_target:
            saved = (
                [x.clone() for x in target.cache.key_cache],
                [x.clone() for x in target.cache.value_cache],
                target.mean.clone(),
                target.std.clone(),
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
            keys, values, mean, std = saved
            for dst, src in zip(target.cache.key_cache, keys):
                dst.copy_(src)
            for dst, src in zip(target.cache.value_cache, values):
                dst.copy_(src)
            target.mean.copy_(mean)
            target.std.copy_(std)

    @torch.no_grad()
    def step(self, raw_window: torch.Tensor) -> torch.Tensor:
        if self.graph is None:
            raise RuntimeError("call capture() first")
        self.x.copy_(raw_window.view_as(self.x))
        self.graph.replay()
        return self.out
