"""Evaluate TimesFM rolling KV cache on a numeric CSV column.

At each patch-aligned online update, all methods first observe the same 32 new
values and then forecast the same future horizon.  ``full_recompute`` calls the
upstream TimesFM ``decode`` implementation; B0/B1 use the rolling engine.

Example::

  CUDA_VISIBLE_DEVICES=0 python scripts/online_benchmark/eval_real_series.py \
    --ckpt checkpoints/TimesFM-2.5-200M/model.safetensors \
    --csv datasets/ETT-small/ETTh1.csv \
    --column OT --start-index 11520 --steps 64 --context-length 512 \
    --horizon 128 --naive-period 24
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


def upstream_decode(module, window, horizon):
  """Native patchify, prefix-normalize, prefill, and readout reference."""
  masks = torch.zeros_like(window, dtype=torch.bool)
  point_forecast, _, ar = module.decode(horizon, window, masks)
  pieces = [point_forecast[:, -1, ...]]
  if ar is not None:
    pieces.append(ar.reshape(window.shape[0], -1, module.q))
  return torch.cat(pieces, dim=1)[:, :horizon, module.aridx]


def synchronize(device):
  if device == "cuda":
    torch.cuda.synchronize()


def metric_summary(predictions, targets, latencies, naive_scale):
  pred = np.stack(predictions)
  target = np.stack(targets)
  latency = np.asarray(latencies)
  err = pred - target
  mae = float(np.abs(err).mean())
  mse = float(np.square(err).mean())
  smape = float((np.abs(err) / ((np.abs(pred) + np.abs(target)) / 2 + 1e-8)).mean() * 100)
  return {
      "steps": int(len(latency)),
      "mean_latency_ms": float(latency.mean()),
      "p50_latency_ms": float(np.percentile(latency, 50)),
      "p95_latency_ms": float(np.percentile(latency, 95)),
      "samples_per_sec": float(1000.0 / latency.mean()),
      "mae": mae,
      "mse": mse,
      "rmse": float(np.sqrt(mse)),
      "smape": smape,
      "mase": float(mae / max(naive_scale, 1e-8)),
  }


def evaluate_full(module, series, start, steps, context, horizon, patch, device, naive_scale,
                  graph_runner=None):
  predictions, targets, latencies = [], [], []
  for patch_start in range(start, start + steps * patch, patch):
    window = torch.as_tensor(
        series[patch_start - context + patch:patch_start + patch][None, :], device=device
    )
    target = series[patch_start + patch:patch_start + patch + horizon]
    t0 = time.perf_counter()
    pred = graph_runner.step(window) if graph_runner else upstream_decode(module, window, horizon)
    synchronize(device)
    latencies.append((time.perf_counter() - t0) * 1000)
    predictions.append(pred[0].cpu().numpy())
    targets.append(target)
  return metric_summary(predictions, targets, latencies, naive_scale), np.stack(predictions)


def evaluate_rolling(module, series, start, steps, cfg, naive_scale, use_cuda_graph=False):
  engine = RollingTimesFMEngine(module, cfg)
  patch = module.p
  engine.full_refresh(torch.as_tensor(series[start - cfg.context_length:start]))
  synchronize(cfg.device)
  runner = None
  if use_cuda_graph:
    runner = CudaGraphRollingStep(engine)
    runner.capture(preserve_state=True)
  predictions, targets, latencies = [], [], []
  for patch_start in range(start, start + steps * patch, patch):
    new_patch = torch.as_tensor(series[patch_start:patch_start + patch][None, :])
    target = series[patch_start + patch:patch_start + patch + cfg.horizon]
    t0 = time.perf_counter()
    pred = runner.step(new_patch) if runner else engine.step_patch(new_patch)
    synchronize(cfg.device)
    latencies.append((time.perf_counter() - t0) * 1000)
    predictions.append(pred[0].cpu().numpy())
    targets.append(target)
  return metric_summary(predictions, targets, latencies, naive_scale), np.stack(predictions)


def validate_cuda_graphs(module, series, start, args):
  """Reject graph timing if either graph changes the eager result."""
  patch = module.p
  window = torch.as_tensor(
      series[start - args.context_length + patch:start + patch][None, :], device=args.device
  )
  full_cfg = RollingConfig(context_length=args.context_length, horizon=args.horizon,
                           full_refresh_every=0, batch_size=1, device=args.device,
                           dtype=torch.float32)
  full_graph = CudaGraphFullDecode(RollingTimesFMEngine(module, full_cfg))
  full_graph.capture()
  got = full_graph.step(window).clone()
  expected = upstream_decode(module, window, args.horizon)
  full_err = (got - expected).abs().max().item()
  full_scale = max(expected.abs().max().item(), 1.0)
  if full_err > 1e-4 * full_scale:
    raise RuntimeError(f"full CUDA graph mismatch: {full_err:.3e}")

  cfg = RollingConfig(context_length=args.context_length, horizon=args.horizon,
                      full_refresh_every=0, batch_size=1, device=args.device,
                      dtype=torch.float32)
  initial = torch.as_tensor(series[start - args.context_length:start])
  graph_engine = RollingTimesFMEngine(module, cfg)
  graph_engine.full_refresh(initial)
  rolling_graph = CudaGraphRollingStep(graph_engine)
  rolling_graph.capture(preserve_state=True)
  eager_engine = RollingTimesFMEngine(module, cfg)
  eager_engine.full_refresh(initial)
  new_patch = torch.as_tensor(series[start:start + patch][None, :])
  got = rolling_graph.step(new_patch).clone()
  expected = eager_engine.step_patch(new_patch)
  roll_err = (got - expected).abs().max().item()
  roll_scale = max(expected.abs().max().item(), 1.0)
  if roll_err > 1e-4 * roll_scale:
    raise RuntimeError(f"rolling CUDA graph mismatch: {roll_err:.3e}")
  print(f"CUDA Graph validation: full max_err={full_err:.3e}, rolling max_err={roll_err:.3e}")
  return full_graph


def main(args):
  if args.context_length % 32:
    raise ValueError("--context-length must be a multiple of TimesFM's 32-point patch size")
  df = pd.read_csv(args.csv)
  if args.column not in df:
    raise ValueError(f"Column {args.column!r} is absent; available: {list(df.columns)}")
  series = pd.to_numeric(df[args.column], errors="raise").to_numpy(dtype=np.float32)
  if not np.isfinite(series).all():
    raise ValueError("Series contains NaN or infinite values")
  patch = 32
  required = args.start_index + args.steps * patch + args.horizon
  if args.start_index < args.context_length or required > len(series):
    raise ValueError(f"Need {required} points and initial context; series has {len(series)}")

  device = args.device
  print(f"device={device}, series={args.column}, n={len(series)}, patch={patch}")
  print(f"evaluation: {args.steps} updates, context={args.context_length}, horizon={args.horizon}")
  module = TimesFM_2p5_200M_torch_module()
  module.device = torch.device(device)
  module.load_checkpoint(args.ckpt)
  module.eval()
  naive_scale = float(np.abs(series[args.naive_period:args.start_index] -
                             series[:args.start_index - args.naive_period]).mean())

  methods = [m.strip() for m in args.methods.split(",") if m.strip()]
  if not methods or any(m not in {"B0", "B1"} for m in methods):
    raise ValueError("--methods must be a non-empty comma-separated subset of B0,B1")
  if args.cuda_graph and args.device != "cuda":
    raise ValueError("--cuda-graph requires --device cuda")
  if args.cuda_graph and methods != ["B0"]:
    raise ValueError(
        "fully graph-captured rolling currently supports B0 only; B1 refresh needs a "
        "separate full-refresh graph. Use --methods B0 for a graph-to-graph comparison."
    )
  full_graph = validate_cuda_graphs(module, series, args.start_index, args) if args.cuda_graph else None

  print("\n[full recompute]")
  full, full_preds = evaluate_full(
      module, series, args.start_index, args.steps, args.context_length,
      args.horizon, patch, device, naive_scale, graph_runner=full_graph
  )
  results = {"full_recompute": full}
  for name in methods:
    refresh = 0 if name == "B0" else args.full_refresh_every
    cfg = RollingConfig(
        context_length=args.context_length, horizon=args.horizon,
        full_refresh_every=refresh, batch_size=1, device=device, dtype=torch.float32,
    )
    print(f"[{name}] full_refresh_every={refresh} patch updates")
    metrics, preds = evaluate_rolling(
        module, series, args.start_index, args.steps, cfg, naive_scale,
        use_cuda_graph=args.cuda_graph,
    )
    metrics["prediction_gap_mae_vs_full"] = float(np.abs(preds - full_preds).mean())
    metrics["mae_delta_vs_full"] = metrics["mae"] - full["mae"]
    results[name] = metrics

  print("\nmethod                 latency(ms)      MAE       RMSE      sMAPE    MASE   gap/full")
  for name, metrics in results.items():
    print(f"{name:<22} {metrics['mean_latency_ms']:>10.3f} "
          f"{metrics['mae']:>9.5f} {metrics['rmse']:>9.5f} "
          f"{metrics['smape']:>9.3f} {metrics['mase']:>7.3f} "
          f"{metrics.get('prediction_gap_mae_vs_full', 0.0):>10.5f}")

  output = {
      "dataset": {"csv": os.path.abspath(args.csv), "column": args.column,
                  "series_length": int(len(series)), "start_index": args.start_index,
                  "naive_period": args.naive_period, "naive_mae": naive_scale},
      "protocol": "observe a 32-point patch, then forecast the same future H points",
      "context_length": args.context_length,
      "prediction_length": args.horizon,
      "execution": "cuda_graph" if args.cuda_graph else "eager",
      "methods": results,
  }
  if args.output:
    with open(args.output, "w") as f:
      json.dump(output, f, indent=2)
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--ckpt", required=True)
  parser.add_argument("--csv", required=True)
  parser.add_argument("--column", required=True)
  parser.add_argument("--start-index", type=int, required=True)
  parser.add_argument("--steps", type=int, default=64, help="number of 32-point updates")
  parser.add_argument("--context-length", type=int, default=512)
  parser.add_argument("--horizon", type=int, default=128)
  parser.add_argument("--full-refresh-every", type=int, default=16)
  parser.add_argument("--methods", default="B0,B1")
  parser.add_argument("--cuda-graph", action="store_true")
  parser.add_argument("--naive-period", type=int, default=1)
  parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  parser.add_argument("--output")
  main(parser.parse_args())
