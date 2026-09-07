"""Evaluate TimeMoE rolling KV cache against true future observations.

The existing ``compare_methods.py`` is a synthetic-data performance harness.
This script evaluates one numeric CSV column in an online setting, aligning every
method at the same information set: after observing ``x[t]``, each method
forecasts ``x[t+1:t+1+H]``.  It reports both ordinary forecasting error and the
prediction gap introduced by cached/stale states relative to full recomputation.

Example (ETTh1 test region)::

  CUDA_VISIBLE_DEVICES=0 python scripts/online_benchmark/eval_real_series.py \
    --model checkpoints/TimeMoE-50M \
    --csv datasets/ETT-small/ETTh1.csv \
    --column OT --start-index 11520 --steps 96 --context-length 512 \
    --prediction-length 64 --naive-period 24 \
    --output results/timemoe_etth1_ot.json
"""

import argparse
import json
import os
import sys
import time
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from time_moe.online import CudaGraphRollingTimeMoEStep, RollingTimeMoEEngine
from time_moe.online.metrics import compute_mae, compute_mse, compute_smape, naive_mae
from time_moe.online.rolling_engine import EngineConfig


def get_device_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def synchronize(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def load_model(model_path: str, device: str, dtype: torch.dtype):
    from time_moe.models.modeling_time_moe import TimeMoeForPrediction

    model = TimeMoeForPrediction.from_pretrained(
        model_path, device_map=device, torch_dtype=dtype
    )
    model.eval()
    return model


def full_forecast(model, window: np.ndarray, horizon: int, device: str, dtype: torch.dtype):
    """Forecast immediately after the final observation in ``window``."""
    mean = float(window.mean())
    std = max(float(window.std()), 1e-8)
    normed = torch.as_tensor(
        ((window - mean) / std)[None, :], device=device, dtype=dtype
    )
    with torch.no_grad():
        out = model(
            input_ids=normed,
            use_cache=False,
            return_dict=True,
            max_horizon_length=horizon,
        )
    return out.logits[0, -1, :horizon].float().cpu().numpy() * std + mean


def summary(predictions, targets, latencies_ms, naive_scale):
    preds = np.stack(predictions)
    actual = np.stack(targets)
    lats = np.asarray(latencies_ms)
    mae = compute_mae(preds, actual)
    mse = compute_mse(preds, actual)
    return {
        "steps": int(len(lats)),
        "mean_latency_ms": float(lats.mean()),
        "p50_latency_ms": float(np.percentile(lats, 50)),
        "p95_latency_ms": float(np.percentile(lats, 95)),
        "samples_per_sec": float(1000.0 / lats.mean()),
        "mae": mae,
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "smape": compute_smape(preds, actual),
        "mase": float(mae / max(naive_scale, 1e-8)),
    }


def evaluate_full(model, series, start, steps, context, horizon, device, dtype, naive_scale):
    predictions, targets, latencies = [], [], []
    for t in range(start, start + steps):
        window = series[t - context + 1 : t + 1]
        target = series[t + 1 : t + 1 + horizon]
        t0 = time.perf_counter()
        pred = full_forecast(model, window, horizon, device, dtype)
        synchronize(device)
        latencies.append((time.perf_counter() - t0) * 1000)
        predictions.append(pred)
        targets.append(target)
    return summary(predictions, targets, latencies, naive_scale), np.stack(predictions)


def evaluate_rolling(model, series, start, steps, cfg, device, naive_scale,
                     use_cuda_graph=False):
    engine = RollingTimeMoEEngine(model, cfg)
    initial = torch.as_tensor(series[start - cfg.context_length : start][None, :])
    engine.full_refresh(initial)
    synchronize(device)
    runner = None
    if use_cuda_graph:
        runner = CudaGraphRollingTimeMoEStep(engine)
        runner.capture(preserve_state=True)

    predictions, targets, latencies = [], [], []
    for t in range(start, start + steps):
        target = series[t + 1 : t + 1 + cfg.prediction_length]
        t0 = time.perf_counter()
        pred_t = runner.step(float(series[t])) if runner else engine.step(float(series[t]))
        pred = pred_t.cpu().numpy()[0] if runner else pred_t.numpy()
        synchronize(device)
        latencies.append((time.perf_counter() - t0) * 1000)
        predictions.append(pred)
        targets.append(target)
    return summary(predictions, targets, latencies, naive_scale), np.stack(predictions)


def make_config(args, method, device, dtype):
    common = dict(
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        tail_length=args.tail_length,
        batch_size=1,
        device=device,
        dtype=dtype,
    )
    if method == "B0":
        return EngineConfig(**common, tail_recompute_every=0, full_refresh_every=0)
    if method == "B1":
        return EngineConfig(**common, tail_recompute_every=0,
                            full_refresh_every=args.full_refresh_every)
    return EngineConfig(**common, tail_recompute_every=args.tail_recompute_every,
                        full_refresh_every=args.full_refresh_every)


def main(args):
    df = pd.read_csv(args.csv)
    if args.column not in df:
        raise ValueError(f"Column {args.column!r} is absent; available: {list(df.columns)}")
    series = pd.to_numeric(df[args.column], errors="raise").to_numpy(dtype=np.float32)
    if not np.isfinite(series).all():
        raise ValueError("Series contains NaN or infinite values.")
    required = args.start_index + args.steps + args.prediction_length
    if args.start_index < args.context_length or required > len(series):
        raise ValueError(
            f"Need context before start and {required} values, but series has {len(series)}."
        )

    device, dtype = get_device_dtype()
    print(f"device={device}, dtype={dtype}, series={args.column}, n={len(series)}")
    print(f"evaluation: observed x[{args.start_index}]..x[{args.start_index + args.steps - 1}], "
          f"context={args.context_length}, horizon={args.prediction_length}")
    naive_scale = naive_mae(series[:args.start_index], period=args.naive_period)
    model = load_model(args.model, device, dtype)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if not methods or any(m not in {"B0", "B1", "B2"} for m in methods):
        raise ValueError("--methods must be a non-empty comma-separated subset of B0,B1,B2")
    if args.cuda_graph and (device != "cuda" or methods != ["B0"]):
        raise ValueError("--cuda-graph requires CUDA and --methods B0")

    print("\n[full recompute]")
    full, full_preds = evaluate_full(
        model, series, args.start_index, args.steps, args.context_length,
        args.prediction_length, device, dtype, naive_scale
    )
    results = {"full_recompute": full}

    for method in methods:
        cfg = make_config(args, method, device, dtype)
        print(f"[{method}] {asdict(cfg)}")
        metrics, preds = evaluate_rolling(
            model, series, args.start_index, args.steps, cfg, device, naive_scale,
            use_cuda_graph=args.cuda_graph,
        )
        metrics["prediction_gap_mae_vs_full"] = compute_mae(preds, full_preds)
        metrics["mae_delta_vs_full"] = metrics["mae"] - full["mae"]
        results[method] = metrics

    print("\nmethod                 latency(ms)      MAE       RMSE      sMAPE    MASE   gap/full")
    for name, metrics in results.items():
        gap = metrics.get("prediction_gap_mae_vs_full", 0.0)
        print(f"{name:<22} {metrics['mean_latency_ms']:>10.3f} "
              f"{metrics['mae']:>9.5f} {metrics['rmse']:>9.5f} "
              f"{metrics['smape']:>9.3f} {metrics['mase']:>7.3f} {gap:>10.5f}")

    output = {
        "dataset": {"csv": os.path.abspath(args.csv), "column": args.column,
                    "series_length": int(len(series)), "start_index": args.start_index,
                    "naive_period": args.naive_period, "naive_mae": naive_scale},
        "protocol": "observe x[t], then forecast x[t+1:t+1+H] for every method",
        "context_length": args.context_length,
        "prediction_length": args.prediction_length,
        "rolling_execution": "cuda_graph" if args.cuda_graph else "eager",
        "methods": results,
    }
    if args.output:
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nSaved {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--column", required=True)
    parser.add_argument("--start-index", type=int, required=True)
    parser.add_argument("--steps", type=int, default=96)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--prediction-length", type=int, default=64)
    parser.add_argument("--methods", default="B0,B1,B2")
    parser.add_argument("--full-refresh-every", type=int, default=64)
    parser.add_argument("--tail-recompute-every", type=int, default=16)
    parser.add_argument("--tail-length", type=int, default=128)
    parser.add_argument("--naive-period", type=int, default=1)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--output")
    main(parser.parse_args())
