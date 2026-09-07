"""DynamicCache slicing helpers for Timer-S1, compatible with old and new Transformers.

Eviction is slice-by-copy rather than ring-buffer + attention mask, for three
reasons specific to this model:

* Exact relative positions after evicting the oldest token require every
  surviving key to be rotated by R(-1) each step (see rope_utils).  That
  per-key rotation is the expensive part and is identical under any storage
  layout, so a ring buffer would save nothing on it.
* Timer-S1 slices its per-layer rotary table to the physical cache length
  (``cos_cached[:kv_seq_len]`` in modeling_TimerS1.py) and calls
  ``past_key_value.update()`` without cache_position/cache_kwargs, so the
  logical window length is pinned to the physical tensor length and there is
  no API surface through which ring slots could be addressed or masked.
* The whole-window copy is small: ~71 MB at B=1 bf16 for the full 11520-point
  context (24 layers x [1, 16, 720, 64] x K and V x 2 bytes), negligible next
  to a 0.75B-activated-parameter forward.  Time-MoE's engine sets the
  slice-by-copy precedent for MoE models; the cost the artifact README flags
  is measured in isolation by scripts/online_benchmark/bench_rolling.py.
"""

from typing import Optional

from transformers import DynamicCache

try:
    from transformers.cache_utils import DynamicLayer
except ImportError:  # Transformers < 4.57 (the 4.45.2 server takes this path)
    DynamicLayer = None


def get_layer_kv(cache: DynamicCache, layer_idx: int):
    """Return the (key, value) tensors for one layer, each [B, H, S, D]."""
    if hasattr(cache, "key_cache"):
        return cache.key_cache[layer_idx], cache.value_cache[layer_idx]
    layer = cache.layers[layer_idx]
    return layer.keys, layer.values


def num_layers(cache: DynamicCache) -> int:
    return len(cache.key_cache) if hasattr(cache, "key_cache") else len(cache.layers)


def build_cache(keys, values) -> DynamicCache:
    cache = DynamicCache()
    if DynamicLayer is None:
        cache.key_cache = list(keys)
        cache.value_cache = list(values)
        cache._seen_tokens = keys[0].shape[-2] if keys else 0
        return cache
    for key, value in zip(keys, values):
        layer = DynamicLayer()
        try:
            layer.lazy_initialization(key, value)
        except TypeError:
            layer.lazy_initialization(key)
        layer.keys = key
        layer.values = value
        cache.layers.append(layer)
    return cache


def slice_cache(
    cache: DynamicCache, start: int = 0, end: Optional[int] = None
) -> DynamicCache:
    keys, values = [], []
    for layer_idx in range(num_layers(cache)):
        key, value = get_layer_kv(cache, layer_idx)
        keys.append(key[:, :, start:end, :].clone())
        values.append(value[:, :, start:end, :].clone())
    return build_cache(keys, values)
