"""Graph-only TimeMoE rolling/full-refresh evaluation on one real series.

Both update branches are CUDA Graph replays using the graph-safe fixed-shape
MoE dispatcher.  ``refresh=1`` is full recomputation at every scalar update;
``refresh=0`` is rolling cache without refresh.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from time_moe.models.modeling_time_moe import TimeMoeForPrediction
from time_moe.online import (
    CudaGraphFullTimeMoEStep,
    CudaGraphRollingTimeMoEStep,
    RollingTimeMoEEngine,
)
from time_moe.online.rolling_engine import EngineConfig


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
            (np.abs(err) / ((np.abs(pred) + np.abs(target)) / 2 + 1e-8)).mean()
            * 100
        ),
        "mase": float(mae / max(naive_scale, 1e-8)),
    }


@torch.no_grad()
def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("this benchmark is CUDA-Graph-only and requires CUDA")
    if args.context_length < 2:
        raise ValueError("--context-length must be >=2")

    df = pd.read_csv(args.csv)
    if args.column not in df:
        raise ValueError(f"column {args.column!r} absent; available={list(df.columns)}")
    series = pd.to_numeric(df[args.column], errors="raise").to_numpy(np.float32)
    if not np.isfinite(series).all():
        raise ValueError("series contains NaN or infinite values")
    required = args.start_index + args.steps + args.horizon
    if args.start_index < args.context_length or required > len(series):
        raise ValueError(
            f"need start>={args.context_length} and {required} values; series has {len(series)}"
        )

    refresh_lengths = parse_refresh_lengths(args.refresh_lengths)
    dtype = torch.bfloat16
    model = TimeMoeForPrediction.from_pretrained(
        args.ckpt, device_map="cuda", torch_dtype=dtype
    )
    model.eval()
    cfg = EngineConfig(
        context_length=args.context_length,
        prediction_length=args.horizon,
        tail_length=min(128, args.context_length),
        tail_recompute_every=0,
        full_refresh_every=0,
        batch_size=1,
        device="cuda",
        dtype=dtype,
    )
    initial = torch.as_tensor(
        series[args.start_index - args.context_length:args.start_index][None, :],
        device="cuda",
    )

    rolling_engine = RollingTimeMoEEngine(model, cfg)
    rolling_engine.full_refresh(initial)
    rolling = CudaGraphRollingTimeMoEStep(rolling_engine)
    rolling.capture(preserve_state=True)
    full_engine = RollingTimeMoEEngine(model, cfg)
    full_engine.full_refresh(initial)
    full = CudaGraphFullTimeMoEStep(full_engine, rolling_target=rolling)
    full.capture(preserve_target=True)
    torch.cuda.synchronize()

    naive_scale = float(
        np.abs(
            series[args.naive_period:args.start_index]
            - series[:args.start_index - args.naive_period]
        ).mean()
    )
    targets = np.stack([
        series[t + 1:t + 1 + args.horizon]
        for t in range(args.start_index, args.start_index + args.steps)
    ])
    all_predictions = {}
    methods = {}

    for refresh in refresh_lengths:
        # Reset all rolling graph state via a full graph replay.  This setup
        # replay is deliberately excluded from the evaluated updates.
        full.step(initial)
        torch.cuda.synchronize()
        window = initial.clone()
        predictions, latencies = [], []
        full_calls = 0
        for i, t in enumerate(range(args.start_index, args.start_index + args.steps)):
            new_value = torch.as_tensor([series[t]], device="cuda")
            window = torch.cat((window[:, 1:], new_value.view(1, 1)), dim=1)
            use_full = refresh > 0 and (i + 1) % refresh == 0
            t0 = time.perf_counter()
            pred = full.step(window) if use_full else rolling.step(new_value)
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
            "refresh_unit": "scalar updates",
            "full_graph_replays": full_calls,
            "rolling_graph_replays": args.steps - full_calls,
        })
        methods[key] = metrics
        print(
            f"K={refresh:<4} full={full_calls:<4} "
            f"latency={metrics['mean_latency_ms']:.3f} ms MAE={metrics['mae']:.6f}"
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
        "model": "TimeMoE-50M",
        "execution": "cuda_graph_only",
        "dtype": str(dtype),
        "dataset": {
            "csv": os.path.abspath(args.csv),
            "column": args.column,
            "series_length": int(len(series)),
            "start_index": args.start_index,
            "naive_period": args.naive_period,
            "naive_mae": naive_scale,
        },
        "protocol": (
            "observe x[t], then forecast H points; K=1 is full graph every update, "
            "K=0 is rolling graph without refresh"
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
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--refresh-lengths", default="1,4,16,64,0")
    parser.add_argument("--naive-period", type=int, default=1)
    parser.add_argument("--output")
    main(parser.parse_args())
