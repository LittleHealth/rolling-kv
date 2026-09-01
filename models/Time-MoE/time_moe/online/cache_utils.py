"""
KV cache manipulation utilities for transformers DynamicCache (v4.57+).

In transformers ≥ 4.57 DynamicCache changed its internal layout:
  OLD: cache.key_cache[layer_idx]   shape [B, H, S, D]
  NEW: cache.layers[layer_idx].keys  shape [B, H, S, D]

We support both layouts transparently via the helpers below.
"""

from typing import Optional
import torch
from transformers import DynamicCache

try:
    from transformers.cache_utils import DynamicLayer
except ImportError:  # transformers < 4.57 has no per-layer object
    DynamicLayer = None


def _get_layer_kv(cache: DynamicCache, layer_idx: int):
    """Return (key, value) tensors for one layer."""
    if hasattr(cache, "key_cache"):
        # Old layout (transformers < 4.57)
        return cache.key_cache[layer_idx], cache.value_cache[layer_idx]
    else:
        # New layout (transformers ≥ 4.57)
        layer = cache.layers[layer_idx]
        return layer.keys, layer.values


def _num_layers(cache: DynamicCache) -> int:
    if hasattr(cache, "key_cache"):
        return len(cache.key_cache)
    return len(cache.layers)


def get_cache_seq_len(cache: DynamicCache, layer_idx: int = 0) -> int:
    return cache.get_seq_length(layer_idx)


def _build_cache_from_kv(keys_list, values_list) -> DynamicCache:
    """Create a fresh DynamicCache from pre-sliced (k, v) tensors per layer.

    Compatible with transformers 4.57+ (lazy_initialization with 1 or 2 args).
    """
    new = DynamicCache()

    if DynamicLayer is None:
        # Old layout (transformers < 4.57): plain lists of per-layer tensors.
        new.key_cache = list(keys_list)
        new.value_cache = list(values_list)
        new._seen_tokens = keys_list[0].shape[-2] if keys_list else 0
        return new

    for k, v in zip(keys_list, values_list):
        layer = DynamicLayer()
        # lazy_initialization signature changed between 4.57 and 5.x
        try:
            layer.lazy_initialization(k, v)   # transformers 5.x: (key, value)
        except TypeError:
            layer.lazy_initialization(k)       # transformers 4.57.x: (key,)
        layer.keys = k
        layer.values = v
        new.layers.append(layer)
    return new


def clone_cache(cache: DynamicCache) -> DynamicCache:
    n = _num_layers(cache)
    keys_list = []
    values_list = []
    for i in range(n):
        k, v = _get_layer_kv(cache, i)
        keys_list.append(k.clone())
        values_list.append(v.clone())
    return _build_cache_from_kv(keys_list, values_list)


def detach_cache(cache: DynamicCache) -> DynamicCache:
    n = _num_layers(cache)
    keys_list = []
    values_list = []
    for i in range(n):
        k, v = _get_layer_kv(cache, i)
        keys_list.append(k.detach())
        values_list.append(v.detach())
    return _build_cache_from_kv(keys_list, values_list)


def slice_cache(
    cache: DynamicCache,
    start: int = 0,
    end: Optional[int] = None,
) -> DynamicCache:
    """Return a new DynamicCache with the seq_len dimension sliced to [start:end].

    This lets us drop the oldest cached token (start=1) or keep a prefix
    (end=prefix_len) without recomputing anything.
    """
    n = _num_layers(cache)
    keys_list = []
    values_list = []
    for i in range(n):
        k, v = _get_layer_kv(cache, i)
        keys_list.append(k[:, :, start:end, :].clone())
        values_list.append(v[:, :, start:end, :].clone())
    return _build_cache_from_kv(keys_list, values_list)
