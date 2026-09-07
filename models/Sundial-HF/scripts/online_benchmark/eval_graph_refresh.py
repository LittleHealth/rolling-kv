"""Graph-only Sundial refresh evaluation on one real univariate series."""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from sundial_online import (
    CudaGraphFullSundialStep,
    CudaGraphRollingSundialStep,
    RollingSundialEngine,
    SundialRollingConfig,
)


def summarize(prediction, target, latency, naive_scale):
    prediction, target, latency = map(np.asarray, (prediction, target, latency))
    error = prediction - target
    mae = float(np.abs(error).mean())
    mse = float(np.square(error).mean())
    return {
        "steps": int(len(latency)),
        "mean_latency_ms": float(latency.mean()),
        "p50_latency_ms": float(np.percentile(latency, 50)),
        "p95_latency_ms": float(np.percentile(latency, 95)),
        "mae": mae,
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mase": float(mae / max(naive_scale, 1e-8)),
    }


@torch.no_grad()
def main(args):
    refreshes = [int(value) for value in args.refresh_lengths.split(",")]
    if 1 not in refreshes or any(value < 0 for value in refreshes):
        raise ValueError("refresh lengths must be non-negative and include K=1")
    frame = pd.read_csv(args.csv)
    series = pd.to_numeric(frame[args.column], errors="raise").to_numpy(np.float32)
    if not np.isfinite(series).all():
        raise ValueError("series contains NaN or infinite values")
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt, trust_remote_code=True, torch_dtype=torch.float32
    ).cuda().eval()
    patch = int(model.config.input_token_len)
    if args.context_length % patch:
        raise ValueError(f"context length must be divisible by {patch}")
    required = args.start_index + args.steps * patch + args.horizon
    if args.start_index < args.context_length or required > len(series):
        raise ValueError(
            f"need start>={args.context_length} and {required} points; series has {len(series)}"
        )
    cfg = SundialRollingConfig(
        context_length=args.context_length,
        horizon=args.horizon,
        num_samples=args.num_samples,
        sampling_steps=args.sampling_steps,
        seed=args.seed,
        noise_mode=args.noise_mode,
        rope_rebase=args.rope_rebase,
    )
    initial = torch.as_tensor(
        series[args.start_index - args.context_length : args.start_index][None],
        device="cuda",
    )
    rolling_engine = RollingSundialEngine(model, cfg)
    rolling_engine.full_refresh(initial)
    rolling = CudaGraphRollingSundialStep(
        rolling_engine, track_normalization_drift=bool(args.adaptive_thresholds)
    )
    rolling.capture(preserve_state=True)
    full_engine = RollingSundialEngine(model, cfg)
    full_engine.full_refresh(initial)
    full = CudaGraphFullSundialStep(full_engine, rolling)
    full.capture(preserve_target=True)
    torch.cuda.synchronize()

    naive_scale = float(
        np.abs(
            series[args.naive_period : args.start_index]
            - series[: args.start_index - args.naive_period]
        ).mean()
    )
    targets = np.stack(
        [
            series[start + patch : start + patch + args.horizon]
            for start in range(
                args.start_index, args.start_index + args.steps * patch, patch
            )
        ]
    )
    predictions_by_method, methods = {}, {}
    for refresh in refreshes:
        full.step(initial)
        torch.cuda.synchronize()
        window = initial.clone()
        predictions, latencies = [], []
        full_calls = 0
        for index, start in enumerate(
            range(args.start_index, args.start_index + args.steps * patch, patch)
        ):
            new_patch = torch.as_tensor(
                series[start : start + patch][None], device="cuda"
            )
            window = torch.cat((window[:, patch:], new_patch), dim=-1)
            use_full = refresh > 0 and (index + 1) % refresh == 0
            begin = time.perf_counter()
            prediction = full.step(window) if use_full else rolling.step(new_patch)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - begin) * 1000)
            predictions.append(prediction[0].cpu().numpy().copy())
            full_calls += int(use_full)
        predictions = np.stack(predictions)
        key = str(refresh)
        predictions_by_method[key] = predictions
        metrics = summarize(predictions, targets, latencies, naive_scale)
        metrics.update(
            {
                "refresh_length": refresh,
                "refresh_unit": "16-point patch updates",
                "full_graph_replays": full_calls,
                "rolling_graph_replays": args.steps - full_calls,
            }
        )
        methods[key] = metrics
        print(
            f"K={refresh:<4} full={full_calls:<4} "
            f"latency={metrics['mean_latency_ms']:.3f} ms MAE={metrics['mae']:.6f}"
        )

    adaptive_thresholds = [
        float(value) for value in args.adaptive_thresholds.split(",") if value
    ]
    if any(value <= 0 for value in adaptive_thresholds):
        raise ValueError("adaptive thresholds must be positive")
    for threshold in adaptive_thresholds:
        full.step(initial)
        torch.cuda.synchronize()
        window = initial.clone()
        predictions, latencies, drifts = [], [], []
        full_calls = 0
        since_refresh = 0
        refresh_pending = False
        for start in range(
            args.start_index, args.start_index + args.steps * patch, patch
        ):
            new_patch = torch.as_tensor(
                series[start : start + patch][None], device="cuda"
            )
            window = torch.cat((window[:, patch:], new_patch), dim=-1)
            since_refresh += 1
            use_full = since_refresh >= args.adaptive_max_refresh or refresh_pending
            begin = time.perf_counter()
            prediction = full.step(window) if use_full else rolling.step(new_patch)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - begin) * 1000)
            predictions.append(prediction[0].cpu().numpy().copy())
            drift = 0.0 if use_full else rolling.normalization_drift.item()
            drifts.append(drift)
            full_calls += int(use_full)
            if use_full:
                since_refresh = 0
                refresh_pending = False
            elif since_refresh >= args.adaptive_min_refresh and drift >= threshold:
                refresh_pending = True
        predictions = np.stack(predictions)
        key = f"adaptive_{threshold:g}"
        predictions_by_method[key] = predictions
        metrics = summarize(predictions, targets, latencies, naive_scale)
        metrics.update(
            {
                "refresh_policy": "normalization_drift",
                "drift_threshold": threshold,
                "min_refresh_length": args.adaptive_min_refresh,
                "max_refresh_length": args.adaptive_max_refresh,
                "full_graph_replays": full_calls,
                "rolling_graph_replays": args.steps - full_calls,
                "mean_observed_drift": float(np.mean(drifts)),
            }
        )
        methods[key] = metrics
        print(
            f"adaptive={threshold:g} full={full_calls:<4} "
            f"latency={metrics['mean_latency_ms']:.3f} ms MAE={metrics['mae']:.6f}"
        )

    full_predictions = predictions_by_method["1"]
    full_mae = methods["1"]["mae"]
    for key, predictions in predictions_by_method.items():
        methods[key]["prediction_gap_mae_vs_full_k1"] = float(
            np.abs(predictions - full_predictions).mean()
        )
        methods[key]["forecast_mae_delta_vs_full_k1"] = (
            methods[key]["mae"] - full_mae
        )
    output = {
        "model": "Sundial-base-128m",
        "execution": "cuda_graph_only",
        "deterministic_common_noise": True,
        "rope_rebase": args.rope_rebase,
        "num_samples": args.num_samples,
        "sampling_steps": args.sampling_steps,
        "noise_mode": args.noise_mode,
        "dataset": {
            "csv": os.path.abspath(args.csv),
            "column": args.column,
            "series_length": int(len(series)),
            "start_index": args.start_index,
            "naive_period": args.naive_period,
            "naive_mae": naive_scale,
        },
        "context_length": args.context_length,
        "context_tokens": args.context_length // patch,
        "patch_length": patch,
        "prediction_length": args.horizon,
        "methods": methods,
    }
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as handle:
            json.dump(output, handle, indent=2)
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
    parser.add_argument("--context-length", type=int, default=2880)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--refresh-lengths", default="1,2,4,8,16,64,0")
    parser.add_argument("--adaptive-thresholds", default="")
    parser.add_argument("--adaptive-min-refresh", type=int, default=4)
    parser.add_argument("--adaptive-max-refresh", type=int, default=32)
    parser.add_argument("--naive-period", type=int, default=1)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--noise-mode", choices=("antithetic", "random"), default="antithetic")
    parser.add_argument(
        "--rope-rebase", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--output")
    main(parser.parse_args())
