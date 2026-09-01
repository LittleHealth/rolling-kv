"""Validate and benchmark TimeMoE's rolling CUDA Graph fast path.

Every replay is compared with the original dynamic-dispatch eager engine before
timing.  Capture is intentionally limited to B0 (no refresh/tail recompute).
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from time_moe.models.modeling_time_moe import TimeMoeForPrediction
from time_moe.online import (
    CudaGraphRollingTimeMoEStep,
    RollingTimeMoEEngine,
    set_static_moe_dispatch,
)
from time_moe.online.rolling_engine import EngineConfig


def make_series(n, seed=7):
    rng = np.random.RandomState(seed)
    t = np.arange(n, dtype=np.float32)
    return (
        np.sin(2 * np.pi * t / 24)
        + 0.5 * np.sin(2 * np.pi * t / 168)
        + 0.3 * rng.randn(n).astype(np.float32)
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
    values = np.asarray(values)
    return float(np.median(values)), float(np.percentile(values, 95))


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device, dtype = "cuda", torch.bfloat16
    model = TimeMoeForPrediction.from_pretrained(
        args.ckpt, device_map=device, torch_dtype=dtype
    )
    model.eval()
    cfg = EngineConfig(
        context_length=args.context_length,
        prediction_length=args.horizon,
        full_refresh_every=0,
        tail_recompute_every=0,
        batch_size=1,
        device=device,
        dtype=dtype,
    )
    series = make_series(args.context_length + args.validation_steps + args.runs + 32)
    initial = torch.tensor(series[: args.context_length], dtype=torch.float32)

    eager = RollingTimeMoEEngine(model, cfg)
    eager.full_refresh(initial)
    graphed = RollingTimeMoEEngine(model, cfg)
    graphed.full_refresh(initial)
    runner = CudaGraphRollingTimeMoEStep(graphed)
    runner.capture(preserve_state=True)

    # The eager reference must exercise the original where/nonzero dispatcher.
    # The graph has already recorded the static branch, so toggling this Python
    # flag cannot change its replay.
    set_static_moe_dispatch(model, False)
    max_error = 0.0
    scale = 1.0
    validation_inputs = series[
        args.context_length : args.context_length + args.validation_steps
    ]
    for value in validation_inputs:
        expected = eager.step(float(value))
        got = runner.step(float(value)).clone().cpu()[0]
        torch.cuda.synchronize()
        max_error = max(max_error, (got - expected).abs().max().item())
        scale = max(scale, expected.abs().max().item())
    tolerance = args.rtol * scale + args.atol
    if max_error > tolerance:
        raise RuntimeError(
            f"CUDA Graph mismatch: max_error={max_error:.3e}, tolerance={tolerance:.3e}"
        )

    timing_inputs = series[
        args.context_length + args.validation_steps :
        args.context_length + args.validation_steps + max(args.runs, 8)
    ]
    eager_ms, eager_p95 = latency(lambda x: eager.step(float(x)), timing_inputs, 4, args.runs)
    graph_ms, graph_p95 = latency(lambda x: runner.step(float(x)), timing_inputs, 4, args.runs)
    result = {
        "context_length": args.context_length,
        "horizon": args.horizon,
        "dtype": str(dtype),
        "validation_steps": args.validation_steps,
        "max_error": max_error,
        "tolerance": tolerance,
        "eager_median_ms": eager_ms,
        "eager_p95_ms": eager_p95,
        "graph_median_ms": graph_ms,
        "graph_p95_ms": graph_p95,
        "speedup": eager_ms / graph_ms,
    }
    print(json.dumps(result, indent=2))
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--validation-steps", type=int, default=8)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--rtol", type=float, default=2e-3)
    parser.add_argument("--atol", type=float, default=2e-3)
    parser.add_argument("--output")
    main(parser.parse_args())
