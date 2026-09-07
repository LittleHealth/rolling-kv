"""DynamicCache slicing helpers compatible with old and new Transformers."""

from typing import Optional

from transformers import DynamicCache

try:
    from transformers.cache_utils import DynamicLayer
except ImportError:  # Transformers < 4.57
    DynamicLayer = None


def get_layer_kv(cache: DynamicCache, layer_idx: int):
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
