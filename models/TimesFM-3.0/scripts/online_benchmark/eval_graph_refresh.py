"""Graph-only TimesFM-3.0 rolling/full-refresh evaluation on one real series.

TimesFM-3.0 port of ``TimesFM-2.5/scripts/online_benchmark/eval_graph_refresh.py``
(the driver of the 0807 graph-refresh matrix).  The algorithm, timing and
metrics are line-for-line the same; only the model plumbing differs.

Both update branches are CUDA Graph replays.  ``refresh=1`` is full
recomputation at every patch update, while ``refresh=0`` never refreshes.
Positive values greater than one replay the full graph periodically and the
rolling graph on all intervening updates.

TimesFM-3.0 specifics (none of them change the protocol):

* The engine is multivariate-capable and works on ``[B, V, L]`` tensors; this
  driver uses B = V = 1, so windows are ``[1, 1, L]`` and patches ``[1, 1, 32]``.
* The graphs return ``[B, V, H, num_quantiles]``.  The point forecast that is
  scored is the median quantile (``engine.median_q_idx``; q = 0.5), the same
  channel upstream ``TimesFM3Forecaster`` reports as its point forecast and the
  counterpart of TimesFM-2.5's ``aridx`` (its median channel).
* H must need no horizon scratch tokens (H <= stitching extract_len = 64 for
  the released checkpoint), exactly like the 2.5 graph path required
  H <= output_patch_len.
* ``--ckpt`` is the checkpoint DIRECTORY (config.json + model.safetensors);
  a path to the ``model.safetensors`` file inside it is also accepted.

Additions that never touch the timed loop (both optional, off by default):

* ``--verify`` re-runs every refresh schedule on the EAGER engine
  (``RollingTimesFM3Engine.step_patch`` with ``full_refresh_every=K``) and the
  K=1 schedule on upstream ``model.decode``, after all timing has finished,
  and reports the max deviation of the graph predictions.  It checks the
  driver's window/patch alignment and refresh schedule, not the engine.
* ``--meta-output`` writes run metadata (GPU, torch/CUDA versions, TF32 flags,
  capture/eval wall time, verification numbers) to a separate JSON so the
  ``--output`` file keeps exactly the TimesFM-2.5 schema.
"""

import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from timesfm3.model import TimesFM3Torch  # noqa: E402
from timesfm3.online import RollingConfig, RollingTimesFM3Engine  # noqa: E402
from timesfm3.online.graph_runner import (  # noqa: E402
    CudaGraphFullDecode,
    CudaGraphRollingStep,
)


MODEL_NAME = "TimesFM-3.0"


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


def resolve_ckpt_dir(path):
  """Checkpoint directory; a ``model.safetensors`` path maps to its folder."""
  path = os.path.abspath(os.path.expanduser(path))
  if os.path.isfile(path):
    path = os.path.dirname(path)
  if not os.path.isfile(os.path.join(path, "config.json")):
    raise ValueError(
        f"--ckpt must be the TimesFM-3.0 checkpoint directory holding "
        f"config.json + model.safetensors; got {path!r}"
    )
  return path


@torch.no_grad()
def verify_against_eager(model, cfg_kwargs, series, args, refresh_lengths,
                         all_predictions, qm, patch):
  """Post-hoc check of the driver: graph predictions vs eager references.

  Runs strictly after every timed loop, so it cannot influence latency.
  """
  dev = args.device
  L = args.context_length
  initial = torch.as_tensor(
      series[args.start_index - L:args.start_index][None, None, :], device=dev
  )
  starts = list(range(args.start_index, args.start_index + args.steps * patch, patch))
  report = {}
  for refresh in refresh_lengths:
    eng = RollingTimesFM3Engine(
        model, RollingConfig(full_refresh_every=refresh, **cfg_kwargs)
    )
    eng.full_refresh(initial)
    ref = []
    for s in starts:
      new_patch = torch.as_tensor(series[s:s + patch][None, None, :], device=dev)
      ref.append(eng.step_patch(new_patch)[0, 0, :, qm].cpu().numpy().copy())
    ref = np.stack(ref)
    got = all_predictions[str(refresh)]
    diff = np.abs(got - ref)
    report[str(refresh)] = {
        "max_abs_diff_vs_eager": float(diff.max()),
        "max_rel_diff_vs_eager": float(diff.max() / max(np.abs(ref).max(), 1e-9)),
    }
    del eng
  # K=1 must reproduce upstream full decode on every slid window.
  if "1" in all_predictions:
    ref = []
    for s in starts:
      win = torch.as_tensor(
          series[s + patch - L:s + patch][None, None, :], device=dev
      )
      out = model.decode(target=win, horizon=args.horizon, mask=None)
      ref.append(out[0, 0, :, qm].float().cpu().numpy().copy())
    ref = np.stack(ref)
    diff = np.abs(all_predictions["1"] - ref)
    report["1"]["max_abs_diff_vs_decode"] = float(diff.max())
    report["1"]["max_rel_diff_vs_decode"] = float(
        diff.max() / max(np.abs(ref).max(), 1e-9)
    )
  torch.cuda.synchronize()
  return report


@torch.no_grad()
def main(args):
  t_start = time.perf_counter()
  if not torch.cuda.is_available() or args.device != "cuda":
    raise RuntimeError("this benchmark is CUDA-Graph-only and requires --device cuda")
  if args.context_length < 32 or args.context_length % 32:
    raise ValueError("--context-length must be >=32 and divisible by 32")

  # FP32 protocol: keep cuBLAS off TF32 (this is also PyTorch's default; set
  # explicitly so an environment override cannot silently change numerics).
  torch.backends.cuda.matmul.allow_tf32 = False
  torch.backends.cudnn.allow_tf32 = False

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
  model = TimesFM3Torch.from_pretrained(resolve_ckpt_dir(args.ckpt))
  model.to(device=args.device, dtype=torch.float32)
  model.eval()
  if model.input_patch_len != patch:
    raise ValueError(f"expected input_patch_len {patch}, got {model.input_patch_len}")
  graph_horizon_cap = (
      model._stitching_extract_len if model.use_stitching else model.output_patch_len
  )
  if args.horizon > graph_horizon_cap:
    raise ValueError(
        f"graph-only evaluation requires --horizon <= {graph_horizon_cap} "
        "(no horizon scratch tokens)"
    )

  cfg_kwargs = dict(
      context_length=args.context_length,
      horizon=args.horizon,
      batch_size=1,
      num_variates=1,
      device=args.device,
      dtype=torch.float32,
  )
  cfg = RollingConfig(full_refresh_every=0, **cfg_kwargs)
  initial = torch.as_tensor(
      series[args.start_index - args.context_length:args.start_index][None, None, :],
      device=args.device,
  )

  t_cap = time.perf_counter()
  rolling_engine = RollingTimesFM3Engine(model, cfg)
  rolling_engine.full_refresh(initial)
  rolling = CudaGraphRollingStep(rolling_engine)
  rolling.capture(preserve_state=True)
  full_engine = RollingTimesFM3Engine(model, cfg)
  full_engine.full_refresh(initial)
  full = CudaGraphFullDecode(full_engine, rolling_target=rolling)
  full.capture(preserve_target=True)
  torch.cuda.synchronize()
  capture_s = time.perf_counter() - t_cap
  qm = rolling_engine.median_q_idx

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

  t_eval = time.perf_counter()
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
          series[patch_start:patch_start + patch][None, None, :], device=args.device
      )
      window = torch.cat((window[..., patch:], new_patch), dim=-1)
      use_full = refresh > 0 and (i + 1) % refresh == 0
      t0 = time.perf_counter()
      pred = full.step(window) if use_full else rolling.step(new_patch)
      torch.cuda.synchronize()
      latencies.append((time.perf_counter() - t0) * 1000)
      predictions.append(pred[0, 0, :, qm].cpu().numpy().copy())
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
  eval_s = time.perf_counter() - t_eval

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
      "model": MODEL_NAME,
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

  verification = None
  if args.verify:
    t_v = time.perf_counter()
    verification = verify_against_eager(
        model, cfg_kwargs, series, args, refresh_lengths, all_predictions, qm, patch
    )
    verification["wall_s"] = time.perf_counter() - t_v
    for key, rep in verification.items():
      if key == "wall_s":
        continue
      line = f"verify K={key:<4} graph vs eager max|d|={rep['max_abs_diff_vs_eager']:.3e}"
      if "max_abs_diff_vs_decode" in rep:
        line += f"  K=1 graph vs model.decode max|d|={rep['max_abs_diff_vs_decode']:.3e}"
      print(line)

  if args.output:
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
      json.dump(output, f, indent=2)
    print(f"saved {args.output}")
  else:
    print(json.dumps(output, indent=2))

  if args.meta_output:
    props = torch.cuda.get_device_properties(0)
    meta = {
        "model": MODEL_NAME,
        "ckpt": resolve_ckpt_dir(args.ckpt),
        "argv": sys.argv,
        "hostname": platform.node(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": props.name,
        "gpu_uuid": str(getattr(props, "uuid", "")),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dtype": "float32",
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "point_forecast": f"median quantile (index {qm} of {rolling_engine.nq})",
        "ring_capacity": int(rolling_engine.cache.capacity),
        "n_patches": int(rolling_engine.n_patches),
        "capture_s": capture_s,
        "eval_loop_s": eval_s,
        "total_wall_s": time.perf_counter() - t_start,
        "verification": verification,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.meta_output)), exist_ok=True)
    with open(args.meta_output, "w") as f:
      json.dump(meta, f, indent=2)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--ckpt", required=True,
                      help="TimesFM-3.0 checkpoint directory (config.json + "
                      "model.safetensors)")
  parser.add_argument("--csv", required=True)
  parser.add_argument("--column", required=True)
  parser.add_argument("--start-index", type=int, required=True)
  parser.add_argument("--steps", type=int, default=64)
  parser.add_argument("--context-length", type=int, default=512)
  parser.add_argument("--horizon", type=int, default=64)
  parser.add_argument("--refresh-lengths", default="1,4,16,64,0")
  parser.add_argument("--naive-period", type=int, default=1)
  parser.add_argument("--device", default="cuda")
  parser.add_argument("--output")
  parser.add_argument("--verify", action="store_true",
                      help="after timing, check graph predictions against the "
                      "eager engine (every K) and model.decode (K=1)")
  parser.add_argument("--meta-output",
                      help="optional JSON with run metadata and verification")
  main(parser.parse_args())
