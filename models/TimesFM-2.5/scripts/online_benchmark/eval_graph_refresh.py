"""Graph-only TimesFM rolling/full-refresh evaluation on one real series.

Both update branches are CUDA Graph replays.  ``refresh=1`` is full
recomputation at every patch update, while ``refresh=0`` never refreshes.
Positive values greater than one replay the full graph periodically and the
rolling graph on all intervening updates.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from timesfm.online import RollingConfig, RollingTimesFMEngine
from timesfm.online.graph_runner import CudaGraphFullDecode, CudaGraphRollingStep
from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch_module


def parse_refresh_lengths(text):
  values = [int(x) for x in text.split(",")]
  if not values or any(x < 0 for x in values):
    raise ValueError("--refresh-lengths must contain non-negative integers")
  return values


def summarize(pred, target, latency, naive_scale):
  pred, target, latency = map(np.asarray, (pred, target, latency))
  err = pred - target
  mae = float(np.abs(err).mean())
  mse = float(np.square(err).mean())
  return {
      "steps": int(len(latency)),
      "mean_latency_ms": float(latency.mean()),
      "p50_latency_ms": float(np.percentile(latency, 50)),
      "p95_latency_ms": float(np.percentile(latency, 95)),
      "updates_per_sec": float(1000.0 / latency.mean()),
      "mae": mae,
      "mse": mse,
      "rmse": float(np.sqrt(mse)),
      "smape": float(
          (np.abs(err) / ((np.abs(pred) + np.abs(target)) / 2 + 1e-8)).mean() * 100
      ),
      "mase": float(mae / max(naive_scale, 1e-8)),
  }


@torch.no_grad()
def main(args):
  if not torch.cuda.is_available() or args.device != "cuda":
    raise RuntimeError("this benchmark is CUDA-Graph-only and requires --device cuda")
  if args.context_length < 32 or args.context_length % 32:
    raise ValueError("--context-length must be >=32 and divisible by 32")
  if args.horizon > 128:
    raise ValueError("graph-only evaluation currently requires --horizon <= 128")

  df = pd.read_csv(args.csv)
  if args.column not in df:
    raise ValueError(f"column {args.column!r} absent; available={list(df.columns)}")
  series = pd.to_numeric(df[args.column], errors="raise").to_numpy(np.float32)
  if not np.isfinite(series).all():
    raise ValueError("series contains NaN or infinite values")
  patch = 32
  required = args.start_index + args.steps * patch + args.horizon
  if args.start_index < args.context_length or required > len(series):
    raise ValueError(
        f"need start>={args.context_length} and {required} values; series has {len(series)}"
    )

  refresh_lengths = parse_refresh_lengths(args.refresh_lengths)
  module = TimesFM_2p5_200M_torch_module()
  module.device = torch.device(args.device)
  module.load_checkpoint(args.ckpt)
  module.eval()
  cfg = RollingConfig(
      context_length=args.context_length,
      horizon=args.horizon,
      full_refresh_every=0,
      batch_size=1,
      device=args.device,
      dtype=torch.float32,
  )
  initial = torch.as_tensor(
      series[args.start_index - args.context_length:args.start_index][None, :],
      device=args.device,
  )

  rolling_engine = RollingTimesFMEngine(module, cfg)
  rolling_engine.full_refresh(initial)
  rolling = CudaGraphRollingStep(rolling_engine)
  rolling.capture(preserve_state=True)
  full_engine = RollingTimesFMEngine(module, cfg)
  full_engine.full_refresh(initial)
  full = CudaGraphFullDecode(full_engine, rolling_target=rolling)
  full.capture(preserve_target=True)
  torch.cuda.synchronize()

  naive_scale = float(
      np.abs(
          series[args.naive_period:args.start_index]
          - series[:args.start_index - args.naive_period]
      ).mean()
  )
  targets = np.stack([
      series[s + patch:s + patch + args.horizon]
      for s in range(
          args.start_index, args.start_index + args.steps * patch, patch
      )
  ])
  all_predictions = {}
  methods = {}

  for refresh in refresh_lengths:
    # A full replay of the initial window resets every fixed-address rolling
    # state buffer.  It is setup, not an evaluated update.
    full.step(initial)
    torch.cuda.synchronize()
    window = initial.clone()
    predictions, latencies = [], []
    full_calls = 0
    for i, patch_start in enumerate(
        range(args.start_index, args.start_index + args.steps * patch, patch)
    ):
      new_patch = torch.as_tensor(
          series[patch_start:patch_start + patch][None, :], device=args.device
      )
      window = torch.cat((window[:, patch:], new_patch), dim=1)
      use_full = refresh > 0 and (i + 1) % refresh == 0
      t0 = time.perf_counter()
      pred = full.step(window) if use_full else rolling.step(new_patch)
      torch.cuda.synchronize()
      latencies.append((time.perf_counter() - t0) * 1000)
      predictions.append(pred[0].cpu().numpy().copy())
      full_calls += int(use_full)

    predictions = np.stack(predictions)
    key = str(refresh)
    all_predictions[key] = predictions
    metrics = summarize(predictions, targets, latencies, naive_scale)
    metrics.update({
        "refresh_length": refresh,
        "refresh_unit": "32-point patch updates",
        "full_graph_replays": full_calls,
        "rolling_graph_replays": args.steps - full_calls,
    })
    methods[key] = metrics
    print(
        f"K={refresh:<4} full={full_calls:<4} latency={metrics['mean_latency_ms']:.3f} ms "
        f"MAE={metrics['mae']:.6f}"
    )

  if "1" not in all_predictions:
    raise ValueError("--refresh-lengths must include 1 as the full-recompute reference")
  full_predictions = all_predictions["1"]
  full_mae = methods["1"]["mae"]
  for key, pred in all_predictions.items():
    methods[key]["prediction_gap_mae_vs_full_k1"] = float(
        np.abs(pred - full_predictions).mean()
    )
    methods[key]["forecast_mae_delta_vs_full_k1"] = methods[key]["mae"] - full_mae

  output = {
      "model": "TimesFM-2.5-200M",
      "execution": "cuda_graph_only",
      "dataset": {
          "csv": os.path.abspath(args.csv),
          "column": args.column,
          "series_length": int(len(series)),
          "start_index": args.start_index,
          "naive_period": args.naive_period,
          "naive_mae": naive_scale,
      },
      "protocol": (
          "observe one 32-point patch, then forecast H points; K=1 is full graph "
          "every update, K=0 is rolling graph without refresh"
      ),
      "context_length": args.context_length,
      "prediction_length": args.horizon,
      "methods": methods,
  }
  if args.output:
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
      json.dump(output, f, indent=2)
    print(f"saved {args.output}")
  else:
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--ckpt", required=True)
  parser.add_argument("--csv", required=True)
  parser.add_argument("--column", required=True)
  parser.add_argument("--start-index", type=int, required=True)
  parser.add_argument("--steps", type=int, default=64)
  parser.add_argument("--context-length", type=int, default=512)
  parser.add_argument("--horizon", type=int, default=128)
  parser.add_argument("--refresh-lengths", default="1,4,16,64,0")
  parser.add_argument("--naive-period", type=int, default=1)
  parser.add_argument("--device", default="cuda")
  parser.add_argument("--output")
  main(parser.parse_args())
