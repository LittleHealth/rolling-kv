"""Correctness gates for the TimesFM-2.5 rolling engine.

T1  fidelity      : engine.full_refresh + forecast  ==  upstream module.decode
                    on the same window.  Proves the re-expressed forward
                    (RoPE / qk-norm / per-dim-scale / mask / readout) is
                    faithful to upstream.
T2  V-1 exactness : growing window (append only, no eviction) must reproduce
                    upstream's per-patch outputs exactly.  This is the ground
                    truth for the ring buffer, position tags and mask.  Any
                    failure here is an implementation bug, not a cache gap.
T3  eviction      : once the window is full, report the actual rolling-vs-full
                    gap per cache age.  No assertion -- this *is* the quantity
                    the theory bounds.

Usage:
  python test_exactness.py --ckpt /path/to/model.safetensors [--device cuda]
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from timesfm.online import RollingConfig, RollingTimesFMEngine  # noqa: E402
from timesfm.timesfm_2p5.timesfm_2p5_torch import (  # noqa: E402
  TimesFM_2p5_200M_torch_module,
)
from timesfm.torch import util  # noqa: E402

revin = util.revin


def make_series(n, seed=0):
  rng = np.random.RandomState(seed)
  t = np.arange(n, dtype=np.float32)
  return (
    np.sin(2 * np.pi * t / 96)
    + 0.5 * np.sin(2 * np.pi * t / 336)
    + 0.2 * rng.randn(n).astype(np.float32)
  ).astype(np.float32)


def upstream_decode(module, window, horizon):
  """Native TimesFM path: patchify, prefix stats, prefill, readout."""
  inputs = window
  masks = torch.zeros_like(inputs, dtype=torch.bool)
  pf, _, ar = module.decode(horizon, inputs, masks)
  to_cat = [pf[:, -1, ...]]
  if ar is not None:
    to_cat.append(ar.reshape(inputs.shape[0], -1, module.q))
  return torch.cat(to_cat, dim=1)[:, :horizon, module.aridx]


def upstream_all_patch_outputs(module, window):
  """Per-patch point forecasts from a single full prefill: [B, N, 128]."""
  B = window.shape[0]
  p = module.p
  patches = window.view(B, -1, p)
  masks = torch.zeros_like(patches, dtype=torch.bool)

  n = torch.zeros(B, device=window.device)
  mu = torch.zeros(B, device=window.device)
  sigma = torch.zeros(B, device=window.device)
  mus, sigmas = [], []
  for i in range(patches.shape[1]):
    (n, mu, sigma), _ = util.update_running_stats(n, mu, sigma, patches[:, i], masks[:, i])
    mus.append(mu)
    sigmas.append(sigma)
  cmu, csigma = torch.stack(mus, 1), torch.stack(sigmas, 1)

  normed = revin(patches, cmu, csigma, reverse=False)
  (_, _, normed_out, _), _ = module(normed, masks, None)
  renormed = revin(normed_out, cmu, csigma, reverse=True)
  return renormed.reshape(B, -1, module.o, module.q)[..., module.aridx]


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--ckpt", required=True)
  ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  ap.add_argument("--context_length", type=int, default=1024)
  ap.add_argument("--horizon", type=int, default=128)
  ap.add_argument("--evict_steps", type=int, default=8)
  args = ap.parse_args()

  torch.manual_seed(0)
  module = TimesFM_2p5_200M_torch_module()
  module.device = torch.device(args.device)
  module.load_checkpoint(args.ckpt)
  module.eval()

  L, H, p = args.context_length, args.horizon, module.p
  N = L // p
  dev = args.device

  series = make_series(L + args.evict_steps * p + p, seed=7)
  window = torch.tensor(series[:L], device=dev)[None, :]

  cfg = RollingConfig(
    context_length=L, horizon=H, full_refresh_every=0, batch_size=1,
    device=dev, dtype=torch.float32,
  )
  eng = RollingTimesFMEngine(module, cfg)

  fail = 0

  # ---------------------------------------------------------------- T1 ----
  eng.full_refresh(window[0])
  got = eng.forecast()
  ref = upstream_decode(module, window, H)
  d1 = (got - ref).abs().max().item()
  scale = ref.abs().max().item()
  ok1 = d1 <= 1e-4 * max(scale, 1.0)
  fail += not ok1
  print(f"\n{'='*68}")
  print("T1  engine.full_refresh  vs  upstream module.decode")
  print(f"{'='*68}")
  print(f"  L={L} (N={N} patches)  H={H}")
  print(f"  max |rolling - upstream| : {d1:.3e}   (output scale {scale:.3f})")
  print(f"  rel                      : {d1/max(scale,1e-9):.3e}")
  print(f"  -> {'PASS' if ok1 else 'FAIL'}")

  # ---------------------------------------------------------------- T2 ----
  # Growing window: append patch by patch into an empty cache, no eviction.
  # Mathematically identical to full recompute -- must match exactly.
  eng2 = RollingTimesFMEngine(module, cfg)
  eng2.cache.reset()
  eng2.raw_buffer = torch.zeros(1, L, device=dev)
  patches = window.view(1, N, p)
  masks = torch.zeros_like(patches, dtype=torch.bool)

  grown = []
  with torch.no_grad():
    for j in range(N):
      mu, sigma = eng2._advance_stats(patches[:, j : j + 1], masks[:, j : j + 1])
      normed = revin(patches[:, j : j + 1], mu, sigma, reverse=False)
      emb = eng2._encode(normed, masks[:, j : j + 1])
      out = eng2._readout(emb, mu[:, -1], sigma[:, -1])
      grown.append(out[..., module.aridx])
  grown = torch.cat(grown, dim=0)  # [N, 128]

  ref_all = upstream_all_patch_outputs(module, window)[0]  # [N, 128]
  d2 = (grown - ref_all).abs().max().item()
  s2 = ref_all.abs().max().item()
  ok2 = d2 <= 1e-4 * max(s2, 1.0)
  fail += not ok2
  print(f"\n{'='*68}")
  print("T2  V-1 growing-window exactness (append only, no eviction)")
  print(f"{'='*68}")
  print(f"  compared {N} successive patch outputs against one full prefill")
  print(f"  max |rolling - full|     : {d2:.3e}   (output scale {s2:.3f})")
  print(f"  rel                      : {d2/max(s2,1e-9):.3e}")
  per_patch = (grown - ref_all).abs().max(dim=1).values
  print(f"  worst patch index        : {int(per_patch.argmax())} / {N-1}")
  print(f"  -> {'PASS' if ok2 else 'FAIL'}")

  # ---------------------------------------------------------------- T3 ----
  print(f"\n{'='*68}")
  print("T3  cache gap after eviction (measurement, not an assertion)")
  print(f"{'='*68}")
  eng3 = RollingTimesFMEngine(module, cfg)
  eng3.full_refresh(window[0])
  print(f"  {'age k':>6}  {'max|Y~-Y^F|':>13}  {'rel':>10}  {'MAE':>10}")
  for k in range(1, args.evict_steps + 1):
    lo = L + (k - 1) * p
    new_patch = torch.tensor(series[lo : lo + p], device=dev)[None, :]
    roll = eng3.step_patch(new_patch)

    cur = torch.tensor(series[lo + p - L : lo + p], device=dev)[None, :]
    full = upstream_decode(module, cur, H)
    gap = (roll - full).abs()
    print(
      f"  {k:>6}  {gap.max().item():>13.4e}  "
      f"{(gap.max()/full.abs().max()).item():>10.3e}  {gap.mean().item():>10.4e}"
    )

  print(f"\n{'='*68}")
  print(f"RESULT: {'ALL GATES PASS' if fail == 0 else f'{fail} GATE(S) FAILED'}")
  print(f"{'='*68}\n")
  return 1 if fail else 0


if __name__ == "__main__":
  sys.exit(main())
