"""Latency benchmark: TimesFM-2.5 rolling KV cache vs native full recompute.

Both paths produce the same thing -- an H-step forecast from the newest window
-- so the wall-clock ratio is the quantity of interest.

The full-recompute path is additionally split into

  (a) prefix running-stats loop : a serial Python loop of N Welford updates
                                  that upstream `decode()` runs before every
                                  forecast;
  (b) transformer prefill       : N tokens through 20 layers;
  (c) readout                   : output projection + revin.

Rolling replaces (a) with a single Welford update and (b) with a 1-token
forward, so the split shows where the speedup actually comes from.

Usage:
  python bench_rolling.py --ckpt model.safetensors --device cuda \
      --context_lengths 512,2048,8192,16384 --horizon 128
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from timesfm.online import RollingConfig, RollingTimesFMEngine  # noqa: E402
from timesfm.timesfm_2p5.timesfm_2p5_torch import (  # noqa: E402
  TimesFM_2p5_200M_torch_module,
)
from timesfm.torch import util  # noqa: E402

revin = util.revin


def sync(device):
  if device == "cuda":
    torch.cuda.synchronize()
  elif device == "mps":
    torch.mps.synchronize()


def timeit(fn, device, warmup=3, n=15):
  for _ in range(warmup):
    fn()
  sync(device)
  lats = []
  for _ in range(n):
    t0 = time.perf_counter()
    fn()
    sync(device)
    lats.append((time.perf_counter() - t0) * 1000.0)
  a = np.array(lats)
  return dict(
    median=float(np.median(a)),
    mean=float(a.mean()),
    p95=float(np.percentile(a, 95)),
    std=float(a.std()),
  )


def make_series(n, seed=0):
  rng = np.random.RandomState(seed)
  t = np.arange(n, dtype=np.float32)
  return (
    np.sin(2 * np.pi * t / 96)
    + 0.5 * np.sin(2 * np.pi * t / 336)
    + 0.2 * rng.randn(n).astype(np.float32)
  ).astype(np.float32)


def stats_loop(module, patches, masks, device):
  """Upstream's serial prefix running-stats loop."""
  B = patches.shape[0]
  n = torch.zeros(B, device=device)
  mu = torch.zeros(B, device=device)
  sigma = torch.zeros(B, device=device)
  mus, sigmas = [], []
  for i in range(patches.shape[1]):
    (n, mu, sigma), _ = util.update_running_stats(n, mu, sigma, patches[:, i], masks[:, i])
    mus.append(mu)
    sigmas.append(sigma)
  return torch.stack(mus, 1), torch.stack(sigmas, 1)


def stats_vectorized(patches):
  """Same prefix statistics as `stats_loop`, as a parallel scan.

  Upstream computes the causal prefix mean/std with a serial Welford loop over
  the N patches.  For a fully-observed window that loop is just a prefix sum,
  so it collapses to two cumsums.  Used as the *fair* full-recompute baseline:
  without it, most of what rolling "saves" is a Python-loop artifact rather
  than transformer compute.
  """
  B, N, p = patches.shape
  counts = torch.arange(1, N + 1, device=patches.device, dtype=patches.dtype) * p
  s1 = torch.cumsum(patches.sum(-1), dim=1)
  s2 = torch.cumsum(patches.pow(2).sum(-1), dim=1)
  mu = s1 / counts
  var = s2 / counts - mu.pow(2)
  return mu, torch.sqrt(torch.clamp(var, min=0.0))


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--ckpt", required=True)
  ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  ap.add_argument("--context_lengths", default="512,2048,8192,16384")
  ap.add_argument("--horizon", type=int, default=128)
  ap.add_argument("--batch_size", type=int, default=1)
  ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
  ap.add_argument("--runs", type=int, default=15)
  ap.add_argument("--output", default=None)
  args = ap.parse_args()

  dev = args.device
  dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
  Ls = [int(x) for x in args.context_lengths.split(",")]
  H, B = args.horizon, args.batch_size

  torch.manual_seed(0)
  module = TimesFM_2p5_200M_torch_module()
  module.device = torch.device(dev)
  module.load_checkpoint(args.ckpt)
  module.eval()
  if dtype != torch.float32:
    module.to(dtype)

  n_params = sum(p.numel() for p in module.parameters())
  print(f"\ndevice={dev}  dtype={args.dtype}  batch={B}  horizon={H}")
  print(f"params: {n_params/1e6:.1f}M   weights: {n_params*module.tokenizer.hidden_layer.weight.element_size()/1e6:.0f} MB")
  if dev == "cuda":
    print(f"gpu: {torch.cuda.get_device_name(0)}")

  results = {}
  p = module.p

  for L in Ls:
    N = L // p
    series = make_series(L + 64 * p, seed=7)
    window = torch.tensor(
      np.stack([series[:L]] * B), device=dev, dtype=dtype
    )  # [B, L]
    masks = torch.zeros_like(window, dtype=torch.bool)
    patches = window.view(B, N, p)
    pmasks = torch.zeros_like(patches, dtype=torch.bool)

    # ---- full recompute (native upstream path) ----
    t_full = timeit(lambda: module.decode(H, window, masks), dev, n=args.runs)

    # ---- component split of the full path ----
    t_stats = timeit(lambda: stats_loop(module, patches, pmasks, dev), dev, n=args.runs)
    cmu, csigma = stats_loop(module, patches, pmasks, dev)
    normed = revin(patches, cmu, csigma, reverse=False)

    def prefill():
      with torch.no_grad():
        module(normed, pmasks, None)

    t_prefill = timeit(prefill, dev, n=args.runs)

    # ---- fair baseline: vectorized prefix stats + prefill + readout ----
    vmu, vsigma = stats_vectorized(patches)
    stat_err = max(
      (vmu - cmu).abs().max().item(), (vsigma - csigma).abs().max().item()
    )

    def full_fair():
      with torch.no_grad():
        mu_, sd_ = stats_vectorized(patches)
        nz = revin(patches, mu_, sd_, reverse=False)
        (_, _, out, _), _ = module(nz, pmasks, None)
        return revin(out, mu_, sd_, reverse=True)

    t_fair = timeit(full_fair, dev, n=args.runs)

    # ---- rolling ----
    cfg = RollingConfig(
      context_length=L, horizon=H, full_refresh_every=0,
      batch_size=B, device=dev, dtype=dtype,
    )
    eng = RollingTimesFMEngine(module, cfg)
    eng.full_refresh(window)

    counter = {"i": 0}

    def roll_step():
      i = counter["i"] % 32
      counter["i"] += 1
      lo = L + i * p
      np_ = torch.tensor(
        np.stack([series[lo : lo + p]] * B), device=dev, dtype=dtype
      )
      eng.step_patch(np_)

    t_roll = timeit(roll_step, dev, warmup=5, n=args.runs)

    kv_mb = eng.cache.nbytes() / 1e6
    speedup = t_full["median"] / t_roll["median"]
    speedup_fair = t_fair["median"] / t_roll["median"]

    results[L] = dict(
      N=N, full=t_full, full_fair=t_fair, rolling=t_roll, stats_loop=t_stats,
      prefill=t_prefill, speedup=speedup, speedup_fair=speedup_fair,
      kv_mb=kv_mb, stat_err=stat_err,
    )

    print(f"\n  L={L:>6}  (N={N} patches)")
    print(f"    full (native)    : {t_full['median']:8.2f} ms  (p95 {t_full['p95']:7.2f})")
    print(f"      |- stats loop  : {t_stats['median']:8.2f} ms  "
          f"({100*t_stats['median']/t_full['median']:5.1f}%)")
    print(f"      |- prefill     : {t_prefill['median']:8.2f} ms  "
          f"({100*t_prefill['median']/t_full['median']:5.1f}%)")
    print(f"    full (fair)      : {t_fair['median']:8.2f} ms   "
          f"[vectorized stats, max stat err {stat_err:.2e}]")
    print(f"    rolling update   : {t_roll['median']:8.2f} ms  (p95 {t_roll['p95']:7.2f})")
    print(f"    speedup vs native: {speedup:8.2f}x")
    print(f"    speedup vs fair  : {speedup_fair:8.2f}x      KV cache {kv_mb:.1f} MB")

  # ---- summary table ----
  print(f"\n{'='*84}")
  print("TimesFM-2.5-200M   rolling KV cache vs full recompute")
  print(f"{'='*84}")
  hdr = (f"{'L':>7} {'N':>5} {'native(ms)':>11} {'stats(ms)':>10} {'prefill(ms)':>12} "
         f"{'fair(ms)':>9} {'roll(ms)':>9} {'sp/native':>10} {'sp/fair':>8} {'KV(MB)':>8}")
  print(hdr)
  print("-" * len(hdr))
  for L, r in results.items():
    print(f"{L:>7} {r['N']:>5} {r['full']['median']:>11.2f} "
          f"{r['stats_loop']['median']:>10.2f} {r['prefill']['median']:>12.2f} "
          f"{r['full_fair']['median']:>9.2f} {r['rolling']['median']:>9.2f} "
          f"{r['speedup']:>9.2f}x {r['speedup_fair']:>7.2f}x {r['kv_mb']:>8.1f}")
  print(f"{'='*84}\n")

  if args.output:
    with open(args.output, "w") as f:
      json.dump(
        {"config": vars(args), "params_M": n_params / 1e6,
         "results": {str(k): v for k, v in results.items()}}, f, indent=2)
    print(f"saved -> {args.output}")


if __name__ == "__main__":
  main()
