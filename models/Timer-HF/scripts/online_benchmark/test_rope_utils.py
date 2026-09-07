"""Small algebraic correctness test for sliding-window RoPE rebasing."""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from timer_online.rope_utils import rebase_rope_keys_minus_one_, rotate_half


def apply_rope(value, cos, sin):
    return value * cos + rotate_half(value) * sin


def main():
    torch.manual_seed(7)
    head_dim = 128
    raw = torch.randn(2, 8, 29, head_dim, dtype=torch.float64)
    inv_freq = 1.0 / (
        10000 ** (torch.arange(0, head_dim, 2, dtype=torch.float64) / head_dim)
    )
    positions = torch.arange(1, 30, dtype=torch.float64)
    angles = torch.outer(positions, inv_freq)
    cos = torch.cat((angles, angles), dim=-1).cos()[None, None]
    sin = torch.cat((angles, angles), dim=-1).sin()[None, None]
    cached = apply_rope(raw, cos, sin)

    angle_one = torch.cat((inv_freq, inv_freq), dim=-1)[None, None, None]
    rebase_rope_keys_minus_one_(cached, angle_one.cos(), angle_one.sin())

    shifted = torch.outer(positions - 1, inv_freq)
    expected = apply_rope(
        raw,
        torch.cat((shifted, shifted), dim=-1).cos()[None, None],
        torch.cat((shifted, shifted), dim=-1).sin()[None, None],
    )
    error = (cached - expected).abs().max().item()
    print(f"RoPE p->p-1 max error: {error:.3e}")
    if error > 1e-12:
        raise RuntimeError(f"RoPE rebase mismatch: {error:.3e}")


if __name__ == "__main__":
    main()
