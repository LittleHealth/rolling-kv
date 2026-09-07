"""Latency benchmark: TimesFM-3.0 rolling KV cache vs native full recompute.

Both paths produce the same thing -- an H-step quantile forecast from the
newest window -- so the wall-clock ratio is the quantity of interest.

The full-recompute path is additionally split into

  (a) prefix running-stats loop : a serial Python loop of N Welford updates
                                  (`util.get_running_stats`) that upstream
                                  `decode()` runs before every forecast;
  (b) transformer prefill       : `model.forward` on all N patches.

The *fair* baseline re-expresses the full pipeline with upstream modules but
vectorized prefix stats (two cumsums), so the rolling speedup is not inflated
by a Python-loop artifact.  Unlike TimesFM-2.5, the 3.0 forward normalizes
internally, so the fair path rebuilds tokens explicitly from vectorized stats
and drives `pre_transformer_resblock` / `transformer_stack` / `output_head`
directly (readout of the last patch only, which is all a horizon <= o
forecast reads).

Note the native path is *more* expensive than the fair one for an honest
reason: every `decode()` call additionally pays the whole-window detrend fit,
the horizon (CPM) tokens, and the iterative CPM revin refinement.

Usage:
  python bench_rolling.py --device cuda:0 \
      --context_lengths 512,2048,8192,16384 --horizon 64
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from timesfm3 import util  # noqa: E402
from timesfm3.model import TimesFM3Torch  # noqa: E402
from timesfm3.online import RollingConfig, RollingTimesFM3Engine  # noqa: E402

revin = util.revin

DEFAULT_CKPT = os.path.join(
  os.environ.get("ROLLKV_CKPT", "checkpoints"), "TimesFM-3.0"
)


def sync(device):
  if str(device).startswith("cuda"):
    torch.cuda.synchronize()
  elif str(device).startswith("mps"):
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


def stats_vectorized(patches):
  """Same prefix statistics as the serial Welford loop, as a parallel scan.

  Upstream computes the causal prefix mean/std with a serial loop over the N
  patches.  For a fully-observed window that loop is just a prefix sum, so it
  collapses to two cumsums.  Used by the *fair* full-recompute baseline.

  patches: [B, V, N, p].  Returns (mu, sigma), each [B, V, N].
  """
  _, _, N, p = patches.shape
  counts = torch.arange(1, N + 1, device=patches.device, dtype=patches.dtype) * p
  s1 = torch.cumsum(patches.sum(-1), dim=2)
  s2 = torch.cumsum(patches.pow(2).sum(-1), dim=2)
  mu = s1 / counts
  var = s2 / counts - mu.pow(2)
  return mu, torch.sqrt(torch.clamp(var, min=0.0))


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  ap.add_argument(
    "--dtype", default="float32", choices=["float32", "bfloat16"],
    help="bfloat16 is refused with an explanation (upstream forward is "
    "fp32-only); kept as a choice so the refusal is discoverable",
  )
  ap.add_argument("--ckpt", default=DEFAULT_CKPT)
  ap.add_argument("--random_init", action="store_true")
  ap.add_argument("--context_lengths", default="512,2048,8192,16384")
  ap.add_argument("--horizon", type=int, default=64)
  ap.add_argument("--batch_size", type=int, default=1)
  ap.add_argument("--num_variates", type=int, default=1)
  ap.add_argument("--runs", type=int, default=15)
  ap.add_argument("--output", default=None)
  args = ap.parse_args()

  if args.dtype == "bfloat16":
    # Same upstream limitation as in test_exactness.py: the model.decode /
    # model.forward baselines build an fp32 resblock input (fp32 running
    # stats + masks.float()), which torch 2.5.1 rejects against bf16 weights.
    ap.error(
      "--dtype bfloat16 unsupported: the vendored TimesFM-3.0 forward is "
      "fp32-only, so the decode/forward baselines crash before any timing. "
      "Run with --dtype float32."
    )

  dev = args.device
  dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
  Ls = [int(x) for x in args.context_lengths.split(",")]
  H, B, V = args.horizon, args.batch_size, args.num_variates

  if args.random_init:
    torch.manual_seed(0)
    model = TimesFM3Torch()
  else:
    model = TimesFM3Torch.from_pretrained(args.ckpt)
  model.to(device=dev, dtype=dtype)
  model.eval()

  n_params = sum(prm.numel() for prm in model.parameters())
  elem = next(model.parameters()).element_size()
  print(f"\ndevice={dev}  dtype={args.dtype}  batch={B}  variates={V}  horizon={H}")
  print(f"params: {n_params / 1e6:.1f}M   weights: {n_params * elem / 1e6:.0f} MB")
  if str(dev).startswith("cuda"):
    print(f"gpu: {torch.cuda.get_device_name(0)}")

  results = {}
  p = model.input_patch_len
  o = model.output_patch_len
  clip = model.value_clip

  for L in Ls:
    N = L // p
    series = make_series(L + 64 * p, seed=7)
    window = torch.tensor(
      np.tile(series[:L], (B, V, 1)), device=dev, dtype=dtype
    )  # [B, V, L]
    patches = window.view(B, V, N, p)
    pmasks = torch.zeros_like(patches, dtype=torch.bool)
    pmask_bvn = torch.zeros(B, V, N, dtype=torch.bool, device=dev)
    pit = torch.ones(B, V, N, dtype=torch.bool, device=dev)

    # ---- full recompute (native upstream path) ----
    t_full = timeit(
      lambda: model.decode(target=window, horizon=H, mask=None), dev, n=args.runs
    )

    # ---- component split of the full path ----
    t_stats = timeit(
      lambda: util.get_running_stats(patches, pmasks), dev, n=args.runs
    )

    def prefill():
      with torch.no_grad():
        model(
          {"values": patches, "masks": pmasks, "patch_is_target": pit},
          patch_cpm_mask=None,
        )

    t_prefill = timeit(prefill, dev, n=args.runs)

    # ---- fair baseline: vectorized prefix stats + prefill + readout ----
    _, cmu, csigma = util.get_running_stats(patches, pmasks)
    vmu, vsigma = stats_vectorized(patches)
    stat_err = max(
      (vmu - cmu).abs().max().item(), (vsigma - csigma).abs().max().item()
    )

    vals_fcov = torch.zeros(B, V, N, o, device=dev, dtype=dtype)
    mask_vals = torch.zeros(B, V, N, p, device=dev, dtype=dtype)
    mask_fcov = torch.ones(B, V, N, o, device=dev, dtype=dtype)

    def full_fair():
      with torch.no_grad():
        mu_, sd_ = stats_vectorized(patches)
        nz = revin(patches, mu_, sd_, reverse=False).to(dtype)
        tokens = torch.cat([nz, vals_fcov, mask_vals, mask_fcov], dim=-1)
        x = model.pre_transformer_resblock(tokens)
        out, _, _ = model.transformer_stack(x, pmask_bvn)
        raw = model.output_head(out[:, :, -1:])
        res = revin(raw, mu_[:, :, -1:], sd_[:, :, -1:], reverse=True)
        return torch.clamp(res, -clip, clip)

    # One-time sanity: fair path vs upstream forward on the last patch.
    with torch.no_grad():
      fwd_logits = model(
        {"values": patches, "masks": pmasks, "patch_is_target": pit},
        patch_cpm_mask=None,
      )["logits"]
    fair_dev = (
      (full_fair().view(B, V, 1, o, model.num_quantiles) - fwd_logits[:, :, -1:])
      .abs()
      .max()
      .item()
    )

    t_fair = timeit(full_fair, dev, n=args.runs)

    # ---- rolling ----
    cfg = RollingConfig(
      context_length=L, horizon=H, full_refresh_every=0,
      batch_size=B, num_variates=V, device=dev, dtype=dtype,
    )
    eng = RollingTimesFM3Engine(model, cfg)
    eng.full_refresh(window)

    # Pre-stage the 32 rotating patches on-device so the timed region
    # measures only the engine step, not host-side tensor construction and
    # the H2D copy (which would inflate rolling latency, most visibly at
    # small L where the encode itself is cheapest).
    roll_patches = [
      torch.tensor(
        np.tile(series[L + i * p : L + (i + 1) * p], (B, V, 1)),
        device=dev, dtype=dtype,
      )
      for i in range(32)
    ]
    counter = {"i": 0}

    def roll_step():
      i = counter["i"] % 32
      counter["i"] += 1
      eng.step_patch(roll_patches[i])

    t_roll = timeit(roll_step, dev, warmup=5, n=args.runs)

    kv_mb = eng.cache.nbytes() / 1e6
    speedup = t_full["median"] / t_roll["median"]
    speedup_fair = t_fair["median"] / t_roll["median"]

    results[L] = dict(
      N=N, full=t_full, full_fair=t_fair, rolling=t_roll, stats_loop=t_stats,
      prefill=t_prefill, speedup=speedup, speedup_fair=speedup_fair,
      kv_mb=kv_mb, stat_err=stat_err, fair_vs_forward=fair_dev,
    )

    print(f"\n  L={L:>6}  (N={N} patches)")
    print(f"    full (native)    : {t_full['median']:8.2f} ms  (p95 {t_full['p95']:7.2f})")
    print(f"      |- stats loop  : {t_stats['median']:8.2f} ms  "
          f"({100 * t_stats['median'] / t_full['median']:5.1f}%)")
    print(f"      |- prefill     : {t_prefill['median']:8.2f} ms  "
          f"({100 * t_prefill['median'] / t_full['median']:5.1f}%)")
    print(f"    full (fair)      : {t_fair['median']:8.2f} ms   "
          f"[vectorized stats, stat err {stat_err:.2e}, "
          f"vs forward {fair_dev:.2e}]")
    print(f"    rolling update   : {t_roll['median']:8.2f} ms  (p95 {t_roll['p95']:7.2f})")
    print(f"    speedup vs native: {speedup:8.2f}x")
    print(f"    speedup vs fair  : {speedup_fair:8.2f}x      KV cache {kv_mb:.1f} MB")

  # ---- summary table ----
  print(f"\n{'=' * 84}")
  print("TimesFM-3.0   rolling KV cache vs full recompute")
  print("(native additionally pays detrend fit + horizon/CPM tokens + CPM revin "
        "refine per call)")
  print(f"{'=' * 84}")
  hdr = (f"{'L':>7} {'N':>5} {'native(ms)':>11} {'stats(ms)':>10} {'prefill(ms)':>12} "
         f"{'fair(ms)':>9} {'roll(ms)':>9} {'sp/native':>10} {'sp/fair':>8} {'KV(MB)':>8}")
  print(hdr)
  print("-" * len(hdr))
  for L, r in results.items():
    print(f"{L:>7} {r['N']:>5} {r['full']['median']:>11.2f} "
          f"{r['stats_loop']['median']:>10.2f} {r['prefill']['median']:>12.2f} "
          f"{r['full_fair']['median']:>9.2f} {r['rolling']['median']:>9.2f} "
          f"{r['speedup']:>9.2f}x {r['speedup_fair']:>7.2f}x {r['kv_mb']:>8.1f}")
  print(f"{'=' * 84}\n")

  if args.output:
    with open(args.output, "w") as f:
      json.dump(
        {"config": vars(args), "params_M": n_params / 1e6,
         "results": {str(k): v for k, v in results.items()}}, f, indent=2)
    print(f"saved -> {args.output}")


if __name__ == "__main__":
  main()
