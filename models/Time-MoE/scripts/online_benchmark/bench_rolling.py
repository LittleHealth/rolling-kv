"""Latency benchmark: TimeMoE rolling KV cache vs full recompute.

Output format mirrors TimesFM-2.5/scripts/online_benchmark/bench_rolling.py so
the two models can be put side by side.

The rolling path is additionally split into

  (a) slice_cache : physically copies the KV tensors minus the oldest slot --
                    TimeMoE's eviction mechanism, which the TimesFM ring-buffer
                    design replaces with a pure integer update;
  (b) 1-token forward through all layers.

Usage:
  python bench_rolling.py --model /path/to/TimeMoE-50M --device cuda \
      --context_lengths 512,2048,4096 --horizon 64
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from time_moe.online.cache_utils import slice_cache  # noqa: E402
from time_moe.online.rolling_engine import EngineConfig, RollingTimeMoEEngine  # noqa: E402


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
    median=float(np.median(a)), mean=float(a.mean()),
    p95=float(np.percentile(a, 95)), std=float(a.std()),
  )


def make_series(n, seed=0):
  rng = np.random.RandomState(seed)
  t = np.arange(n, dtype=np.float32)
  return (
    np.sin(2 * np.pi * t / 96)
    + 0.5 * np.sin(2 * np.pi * t / 336)
    + 0.2 * rng.randn(n).astype(np.float32)
  ).astype(np.float32)


def load_model(path, device, dtype):
  from time_moe.models.modeling_time_moe import TimeMoeForPrediction

  m = TimeMoeForPrediction.from_pretrained(path, device_map=device, torch_dtype=dtype)
  m.eval()
  return m


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--model", default="Maple728/TimeMoE-50M")
  ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  ap.add_argument("--context_lengths", default="512,2048,4096")
  ap.add_argument("--horizon", type=int, default=64)
  ap.add_argument("--batch_size", type=int, default=1)
  ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
  ap.add_argument("--runs", type=int, default=15)
  ap.add_argument("--output", default=None)
  args = ap.parse_args()

  dev = args.device
  dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
  Ls = [int(x) for x in args.context_lengths.split(",")]
  H, B = args.horizon, args.batch_size

  model = load_model(args.model, dev, dtype)
  n_params = sum(p.numel() for p in model.parameters())
  print(f"\ndevice={dev}  dtype={args.dtype}  batch={B}  horizon={H}")
  print(f"params: {n_params/1e6:.1f}M")
  if dev == "cuda":
    print(f"gpu: {torch.cuda.get_device_name(0)}")

  results = {}

  for L in Ls:
    series = make_series(L + 512, seed=7)
    raw = np.stack([series[:L]] * B)  # [B, L]
    mu = raw.mean(axis=1, keepdims=True)
    sd = np.maximum(raw.std(axis=1, keepdims=True), 1e-8)
    normed = torch.tensor((raw - mu) / sd, device=dev, dtype=dtype)

    def full_step():
      with torch.no_grad():
        out = model(input_ids=normed, use_cache=False, return_dict=True,
                    max_horizon_length=H)
        return out.logits[:, -1, :H]

    t_full = timeit(full_step, dev, n=args.runs)

    cfg = EngineConfig(
      context_length=L, prediction_length=H, full_refresh_every=0,
      tail_recompute_every=0, batch_size=B, device=dev, dtype=dtype,
    )
    eng = RollingTimeMoEEngine(model, cfg)
    eng.full_refresh(torch.tensor(raw, dtype=torch.float32))

    counter = {"i": 0}

    def roll_step():
      i = counter["i"] % 256
      counter["i"] += 1
      eng.step(torch.tensor([series[L + i]] * B))

    t_roll = timeit(roll_step, dev, warmup=5, n=args.runs)

    # eviction cost in isolation: slice_cache physically copies the KV tensors
    def evict_only():
      slice_cache(eng.observed_cache, start=1)

    t_evict = timeit(evict_only, dev, n=args.runs)

    from time_moe.online.cache_utils import _get_layer_kv, _num_layers

    nl = _num_layers(eng.observed_cache)
    k0, _ = _get_layer_kv(eng.observed_cache, 0)
    kv_mb = sum(
      _get_layer_kv(eng.observed_cache, i)[0].numel() * k0.element_size() * 2
      for i in range(nl)
    ) / 1e6

    speedup = t_full["median"] / t_roll["median"]
    results[L] = dict(full=t_full, rolling=t_roll, evict=t_evict,
                      speedup=speedup, kv_mb=kv_mb)

    print(f"\n  L={L:>6}")
    print(f"    full recompute   : {t_full['median']:8.2f} ms  (p95 {t_full['p95']:7.2f})")
    print(f"    rolling update   : {t_roll['median']:8.2f} ms  (p95 {t_roll['p95']:7.2f})")
    print(f"      |- slice_cache : {t_evict['median']:8.2f} ms  "
          f"({100*t_evict['median']/t_roll['median']:5.1f}% of rolling)")
    print(f"    speedup          : {speedup:8.2f}x       KV cache {kv_mb:.1f} MB")

  print(f"\n{'='*78}")
  print("TimeMoE-50M   rolling KV cache vs full recompute")
  print(f"{'='*78}")
  hdr = (f"{'L':>7} {'full(ms)':>10} {'roll(ms)':>10} {'evict(ms)':>11} "
         f"{'speedup':>9} {'KV(MB)':>8}")
  print(hdr)
  print("-" * len(hdr))
  for L, r in results.items():
    print(f"{L:>7} {r['full']['median']:>10.2f} {r['rolling']['median']:>10.2f} "
          f"{r['evict']['median']:>11.2f} {r['speedup']:>8.2f}x {r['kv_mb']:>8.1f}")
  print(f"{'='*78}\n")

  if args.output:
    with open(args.output, "w") as f:
      json.dump({"config": vars(args), "params_M": n_params / 1e6,
                 "results": {str(k): v for k, v in results.items()}}, f, indent=2)
    print(f"saved -> {args.output}")


if __name__ == "__main__":
  main()
