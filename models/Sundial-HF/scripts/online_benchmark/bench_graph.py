"""Validate and benchmark Sundial eager/graph rolling and full paths."""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from sundial_online import (
    CudaGraphFullSundialStep,
    CudaGraphRollingSundialStep,
    RollingSundialEngine,
    SundialRollingConfig,
)


def latency(fn, inputs, warmup, runs):
    for i in range(warmup):
        fn(inputs[i % len(inputs)])
    torch.cuda.synchronize()
    values = []
    for i in range(runs):
        begin = time.perf_counter()
        fn(inputs[i % len(inputs)])
        torch.cuda.synchronize()
        values.append((time.perf_counter() - begin) * 1000)
    return float(np.median(values)), float(np.percentile(values, 95))


@torch.no_grad()
def main(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt, trust_remote_code=True, torch_dtype=torch.float32
    ).cuda().eval()
    cfg = SundialRollingConfig(
        context_length=args.context_length,
        horizon=args.horizon,
        num_samples=args.num_samples,
        sampling_steps=args.sampling_steps,
        seed=args.seed,
        noise_mode=args.noise_mode,
        rope_rebase=args.rope_rebase,
    )
    patch = int(model.config.input_token_len)
    series = torch.randn(
        1,
        args.context_length + patch * (args.validation_steps + args.runs + 8),
        device="cuda",
    )
    initial = series[:, : args.context_length]

    eager = RollingSundialEngine(model, cfg)
    eager.full_refresh(initial)
    graphed_engine = RollingSundialEngine(model, cfg)
    graphed_engine.full_refresh(initial)
    rolling = CudaGraphRollingSundialStep(
        graphed_engine, track_normalization_drift=args.track_normalization_drift
    )
    rolling.capture(preserve_state=True)
    full_engine = RollingSundialEngine(model, cfg)
    full_engine.full_refresh(initial)
    full = CudaGraphFullSundialStep(full_engine, rolling)
    full.capture(preserve_target=True)

    full_window = series[:, patch : args.context_length + patch]
    graph_full = full.step(full_window).clone()
    eager_full = RollingSundialEngine(model, cfg).full_refresh(full_window).clone()
    full_error = (graph_full - eager_full).abs().max().item()

    full.step(initial)
    eager.full_refresh(initial)
    rolling_error = 0.0
    drift_error = 0.0
    validation_window = initial.clone()
    for index in range(args.validation_steps):
        start = args.context_length + index * patch
        new_patch = series[:, start : start + patch]
        got = rolling.step(new_patch).clone()
        expected = eager.fast_update(new_patch).clone()
        rolling_error = max(rolling_error, (got - expected).abs().max().item())
        validation_window = torch.cat(
            (validation_window[:, patch:], new_patch), dim=-1
        )
        if args.track_normalization_drift:
            current_mean = validation_window.mean(dim=-1, keepdim=True)
            current_std = (
                validation_window.std(dim=-1, keepdim=True, unbiased=False) + 1e-5
            )
            expected_drift = torch.maximum(
                (current_mean - rolling.mean).abs() / rolling.std,
                (current_std - rolling.std).abs() / rolling.std,
            ).amax()
            drift_error = max(
                drift_error,
                abs(rolling.normalization_drift.item() - expected_drift.item()),
            )
    torch.cuda.synchronize()
    if full_error > args.atol or rolling_error > args.atol or drift_error > args.atol:
        raise RuntimeError(
            f"graph mismatch: full={full_error:.3e}, rolling={rolling_error:.3e}, "
            f"drift={drift_error:.3e}"
        )

    patches = [
        series[:, args.context_length + i * patch : args.context_length + (i + 1) * patch]
        for i in range(max(args.runs, 8))
    ]
    windows = [
        series[:, i * patch : i * patch + args.context_length]
        for i in range(max(args.runs, 8))
    ]
    eager_roll = RollingSundialEngine(model, cfg)
    eager_roll.full_refresh(initial)
    eager_full_engine = RollingSundialEngine(model, cfg)
    eager_rolling_ms, eager_rolling_p95 = latency(
        eager_roll.fast_update, patches, 2, args.runs
    )
    graph_rolling_ms, graph_rolling_p95 = latency(
        rolling.step, patches, 2, args.runs
    )
    eager_full_ms, eager_full_p95 = latency(
        eager_full_engine.full_refresh, windows, 2, args.runs
    )
    graph_full_ms, graph_full_p95 = latency(full.step, windows, 2, args.runs)
    result = {
        "context_length": args.context_length,
        "context_tokens": args.context_length // patch,
        "patch_length": patch,
        "horizon": args.horizon,
        "num_samples": args.num_samples,
        "sampling_steps": args.sampling_steps,
        "noise_mode": args.noise_mode,
        "rope_rebase": args.rope_rebase,
        "full_graph_max_error": full_error,
        "rolling_graph_max_error": rolling_error,
        "normalization_drift_max_error": drift_error,
        "eager_rolling_median_ms": eager_rolling_ms,
        "eager_rolling_p95_ms": eager_rolling_p95,
        "graph_rolling_median_ms": graph_rolling_ms,
        "graph_rolling_p95_ms": graph_rolling_p95,
        "eager_full_median_ms": eager_full_ms,
        "eager_full_p95_ms": eager_full_p95,
        "graph_full_median_ms": graph_full_ms,
        "graph_full_p95_ms": graph_full_p95,
        "rolling_graph_vs_eager_speedup": eager_rolling_ms / graph_rolling_ms,
        "full_graph_vs_eager_speedup": eager_full_ms / graph_full_ms,
        "graph_rolling_vs_graph_full_speedup": graph_full_ms / graph_rolling_ms,
    }
    print(json.dumps(result, indent=2))
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as handle:
            json.dump(result, handle, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--context-length", type=int, default=2880)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--sampling-steps", type=int, default=50)
    parser.add_argument("--validation-steps", type=int, default=8)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--noise-mode", choices=("antithetic", "random"), default="antithetic")
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument(
        "--rope-rebase", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--output")
    parser.add_argument("--track-normalization-drift", action="store_true")
    main(parser.parse_args())
