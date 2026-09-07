"""RoPE + qk-norm aware key rebasing for sliding Timer-S1 KV caches.

Timer-S1 caches keys AFTER both RoPE and qk-norm (modeling_TimerS1.py):

    cached_k = k_scale * ( s * R(p) k_raw ),
    s        = rsqrt(mean((R(p) k_raw)^2) + eps)

R(p) rotates head-dim pairs (i, i + D/2) by a common per-pair angle, which
preserves the per-head-dim mean square, so the qk-norm scalar ``s`` is
position-invariant and needs no correction.  The per-dim ``k_scale`` does NOT
commute with the rotation, so Timer-HF's plain rebase (rotate the cached key
as-is) would be wrong for this model.  The exact rebase from position p to
p-1 peels the scale off first:

    k' = k_scale * R(-1) (cached_k / k_scale),
    R(-1) x = x * cos(theta_1) - rotate_half(x) * sin(theta_1)

with theta_1 = inv_freq, the angles of row 1 of the ideal rotary table (cos
is even and sin is odd in the angle).  The factors are rebuilt exactly in
fp32 from ``inv_freq`` -- never read from the model's cached tables, which
are bf16 under bf16 loading (see make_rebase_factors).
"""

from __future__ import annotations

import warnings

import torch


def rotate_half(value: torch.Tensor) -> torch.Tensor:
    """Match Timer-S1's split-half rotary convention."""
    half = value.shape[-1] // 2
    return torch.cat((-value[..., half:], value[..., :half]), dim=-1)


def make_rebase_factors(model) -> list[dict]:
    """Precompute per-layer fp32 rebase factors, each shaped [1, 1, 1, D].

    One entry per trunk decoder layer (``model.get_decoder().layers`` -- after
    any test-time layer truncation this is automatically the surviving list).
    The rotary embeddings are per-layer but built from the same dim/base; each
    layer's own ``inv_freq`` is read anyway for robustness.

    cos1/sin1 are built exactly in fp32 from ``inv_freq`` (which
    from_pretrained never retypes: it is created via ``.float()`` and only
    ever moved across devices), NOT read from the model's cos_cached /
    sin_cached rows -- under bf16 loading those tables are bf16
    (_set_cos_sin_cache uses the ambient default dtype), and a bf16-rounded
    (cos1, sin1) pair is not an isometry: cos1^2 + sin1^2 = 1 +/- ~2e-3 per
    dim.  A key that survives m evictions is rebased m times, so that
    per-step amplitude bias would compound to (1 +/- 2e-3)^m -- at the full
    11520-point window a key can be rebased up to 719 times, i.e. up to
    ~0.25x-4x amplitude corruption on the oldest keys in steady state.  With
    exact fp32 factors the rotation is an isometry and nothing compounds.

    The residual bf16 effect is the re-quantization of the rebased keys back
    into bf16 cache storage each step: an unbiased random walk of roughly
    2e-3 * sqrt(m) per dim (~5% at m=719).  For long bf16 steady-state runs
    set a non-zero ``full_refresh_every`` (TimerS1RollingConfig defaults to
    0 = never) to cap m.
    """
    factors = []
    for layer in model.get_decoder().layers:
        attn = layer.self_attn
        theta = attn.rotary_emb.inv_freq.detach().float()
        cos1 = torch.cat((theta.cos(), theta.cos())).view(1, 1, 1, -1)
        sin1 = torch.cat((theta.sin(), theta.sin())).view(1, 1, 1, -1)
        scale = attn.k_scale.detach().float().view(1, 1, 1, -1)
        min_abs = scale.abs().min().item()
        if min_abs < 1e-8:
            raise ValueError(
                f"min |k_scale| = {min_abs:.3e} < 1e-8: those dims are "
                "annihilated in the cached keys and the RoPE rebase cannot "
                "be inverted"
            )
        if min_abs < 1e-3:
            warnings.warn(
                f"min |k_scale| = {min_abs:.3e} < 1e-3: unscale/rescale in "
                "the RoPE rebase may amplify rounding error"
            )
        factors.append(
            {
                "cos1": cos1,
                "sin1": sin1,
                "scale": scale,
                "inv_scale": scale.reciprocal(),
            }
        )
    return factors


def rebase_rope_keys_minus_one_(keys: torch.Tensor, factors: dict) -> torch.Tensor:
    """Move cached keys from positions p to p-1 in place, exactly per key.

    Since RoPE rotations compose, R(p-1) u = R(-1) R(p) u.  The per-dim
    k_scale is divided out before the rotation and re-applied after; the
    qk-norm scalar is rotation-invariant and stays untouched.  Arithmetic
    runs in fp32 and is cast back to ``keys.dtype`` on the in-place copy.
    """
    k32 = keys.float() * factors["inv_scale"]
    k32 = k32 * factors["cos1"] - rotate_half(k32) * factors["sin1"]
    keys.copy_(k32 * factors["scale"])
    return keys
