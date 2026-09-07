"""Window-relative RoPE correction for Sundial sliding caches."""

import torch


def rotate_half(value: torch.Tensor) -> torch.Tensor:
    half = value.shape[-1] // 2
    return torch.cat((-value[..., half:], value[..., :half]), dim=-1)


def rope_minus_one_factors(model, reference: torch.Tensor):
    rotary = model.get_decoder().layers[0].self_attn.rotary_emb
    cos = rotary.cos_cached[1].to(device=reference.device, dtype=reference.dtype)
    sin = rotary.sin_cached[1].to(device=reference.device, dtype=reference.dtype)
    shape = [1] * (reference.ndim - 1) + [reference.shape[-1]]
    return cos.view(shape), sin.view(shape)


def rebase_rope_keys_minus_one_(keys, cos_one, sin_one):
    keys.copy_(keys * cos_one - rotate_half(keys) * sin_one)
    return keys

