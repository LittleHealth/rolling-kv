"""Validate and benchmark Timer eager/graph rolling and full recomputation."""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from timer_online import (
    CudaGraphFullTimerStep,
    CudaGraphRollingTimerStep,
    RollingTimerEngine,
    TimerRollingConfig,
)


def latency(fn, inputs, warmup, runs):
    for i in range(warmup):
        fn(inputs[i % len(inputs)])
    torch.cuda.synchronize()
    values = []
    for i in range(runs):
        t0 = time.perf_counter()
        fn(inputs[i % len(inputs)])
        torch.cuda.synchronize()
        values.append((time.perf_counter() - t0) * 1000)
    return float(np.median(values)), float(np.percentile(values, 95))


@torch.no_grad()
def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt, trust_remote_code=True, torch_dtype=torch.float32
    ).cuda().eval()
    cfg = TimerRollingConfig(
        context_length=args.context_length,
        horizon=args.horizon,
        full_refresh_every=0,
        batch_size=1,
        device="cuda",
        dtype=torch.float32,
        rope_rebase=args.rope_rebase,
    )
    patch = model.config.input_token_len
    series = torch.randn(
        1, args.context_length + patch * (args.validation_steps + args.runs + 8),
        device="cuda",
    )
    initial = series[:, :args.context_length]

    eager = RollingTimerEngine(model, cfg)
    eager.full_refresh(initial)
    graphed_engine = RollingTimerEngine(model, cfg)
    graphed_engine.full_refresh(initial)
    rolling_graph = CudaGraphRollingTimerStep(
        graphed_engine, track_normalization_drift=args.track_normalization_drift
    )
    rolling_graph.capture(preserve_state=True)
    full_engine = RollingTimerEngine(model, cfg)
    full_engine.full_refresh(initial)
    full_graph = CudaGraphFullTimerStep(full_engine, rolling_graph)
    full_graph.capture(preserve_target=True)

    # Full graph must reproduce the eager full-window path exactly.
    full_window = series[:, patch:args.context_length + patch]
    graph_full_output = full_graph.step(full_window).clone()
    eager_full = RollingTimerEngine(model, cfg)
    eager_full_output = eager_full.full_refresh(full_window).clone()
    full_error = (graph_full_output - eager_full_output).abs().max().item()

    # Reset and compare consecutive rolling replays with eager rolling.
    full_graph.step(initial)
    eager.full_refresh(initial)
    rolling_error = 0.0
    drift_error = 0.0
    validation_window = initial.clone()
    for i in range(args.validation_steps):
        start = args.context_length + i * patch
        new_patch = series[:, start:start + patch]
        got = rolling_graph.step(new_patch).clone()
        expected = eager.fast_update(new_patch).clone()
        rolling_error = max(rolling_error, (got - expected).abs().max().item())
        validation_window = torch.cat(
            (validation_window[:, patch:], new_patch), dim=-1
        )
        if args.track_normalization_drift:
            current_mean = validation_window.mean(dim=-1, keepdim=True)
            current_std = validation_window.std(dim=-1, keepdim=True).clamp_min(1e-8)
            expected_drift = torch.maximum(
                (current_mean - rolling_graph.mean).abs() / rolling_graph.std,
                (current_std - rolling_graph.std).abs() / rolling_graph.std,
            ).amax()
            drift_error = max(
                drift_error,
                abs(rolling_graph.normalization_drift.item() - expected_drift.item()),
            )
    torch.cuda.synchronize()
    if full_error > args.atol or rolling_error > args.atol or drift_error > args.atol:
        raise RuntimeError(
            f"graph mismatch: full={full_error:.3e}, rolling={rolling_error:.3e}, "
            f"drift={drift_error:.3e}"
        )

    timing_patches = [
        series[:, args.context_length + i * patch:args.context_length + (i + 1) * patch]
        for i in range(max(args.runs, 8))
    ]
    timing_windows = [
        series[:, i * patch:i * patch + args.context_length]
        for i in range(max(args.runs, 8))
    ]
    eager_roll_engine = RollingTimerEngine(model, cfg)
    eager_roll_engine.full_refresh(initial)
    eager_full_engine = RollingTimerEngine(model, cfg)
    eager_rolling_ms, eager_rolling_p95 = latency(
        eager_roll_engine.fast_update, timing_patches, 3, args.runs
    )
    graph_rolling_ms, graph_rolling_p95 = latency(
        rolling_graph.step, timing_patches, 3, args.runs
    )
    eager_full_ms, eager_full_p95 = latency(
        eager_full_engine.full_refresh, timing_windows, 3, args.runs
    )
    graph_full_ms, graph_full_p95 = latency(
        full_graph.step, timing_windows, 3, args.runs
    )
    result = {
        "context_length": args.context_length,
        "context_tokens": args.context_length // patch,
        "patch_length": patch,
        "horizon": args.horizon,
        "rope_rebase": args.rope_rebase,
        "validation_steps": args.validation_steps,
        "full_graph_max_error": full_error,
        "rolling_graph_max_error": rolling_error,
        "normalization_drift_max_error": drift_error,
        "eager_rolling_median_ms": eager_rolling_ms,
        "eager_rolling_p95_ms": eager_rolling_p95,
        "graph_rolling_median_ms": graph_rolling_ms,
        "graph_rolling_p95_ms": graph_rolling_p95,
        "rolling_graph_speedup": eager_rolling_ms / graph_rolling_ms,
        "eager_full_median_ms": eager_full_ms,
        "eager_full_p95_ms": eager_full_p95,
        "graph_full_median_ms": graph_full_ms,
        "graph_full_p95_ms": graph_full_p95,
        "full_graph_speedup": eager_full_ms / graph_full_ms,
        "graph_rolling_vs_graph_full_speedup": graph_full_ms / graph_rolling_ms,
    }
    print(json.dumps(result, indent=2))
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--context-length", type=int, default=2880)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--validation-steps", type=int, default=8)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument(
        "--rope-rebase", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--track-normalization-drift", action="store_true")
    parser.add_argument("--output")
    main(parser.parse_args())
