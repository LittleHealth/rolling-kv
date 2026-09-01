"""DynamicCache helpers compatible with Transformers 4.40--4.45."""

from typing import Optional

from transformers import DynamicCache


def get_layer_kv(cache: DynamicCache, layer_idx: int):
    return cache.key_cache[layer_idx], cache.value_cache[layer_idx]


def num_layers(cache: DynamicCache) -> int:
    return len(cache.key_cache)


def build_cache(keys, values) -> DynamicCache:
    cache = DynamicCache()
    cache.key_cache = list(keys)
    cache.value_cache = list(values)
    cache._seen_tokens = keys[0].shape[-2] if keys else 0
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

