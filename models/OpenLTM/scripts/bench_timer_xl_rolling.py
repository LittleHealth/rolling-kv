"""Correctness and CUDA Graph latency benchmark for Timer-XL rolling cache."""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from timer_xl_online import (
    CudaGraphFullTimerXLStep,
    CudaGraphRollingTimerXLStep,
    RollingTimerXLEngine,
    TimerXLRollingConfig,
    load_pretrained_timer_xl,
)


def median_ms(fn, runs):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        values.append((time.perf_counter() - start) * 1000)
    return float(torch.tensor(values).median())


def main(args):
    torch.manual_seed(args.seed)
    model = load_pretrained_timer_xl(args.checkpoint, args.device)
    cfg = TimerXLRollingConfig(context_length=args.context_length, device=args.device)
    window = torch.randn(1, args.context_length, 1, device=args.device).cumsum(1) * 0.03
    patches = torch.randn(args.steps, 1, 96, 1, device=args.device)

    eager = RollingTimerXLEngine(model, cfg)
    initial = eager.full_refresh(window)
    official = model(window, None, None)[:, -96:, 0]
    official_error = (initial - official).abs().max().item()

    graph_engine = RollingTimerXLEngine(model, cfg)
    graph_engine.full_refresh(window)
    rolling = CudaGraphRollingTimerXLStep(graph_engine)
    rolling.capture()
    full = CudaGraphFullTimerXLStep(graph_engine)
    full.capture()
    full_error = (full.step(window) - initial).abs().max().item()

    eager_out, graph_out = [], []
    for patch in patches:
        eager_out.append(eager.fast_update(patch).clone())
        graph_out.append(rolling.step(patch).clone())
    torch.cuda.synchronize()
    rolling_error = (torch.stack(eager_out) - torch.stack(graph_out)).abs().max().item()
    patch = patches[-1]
    rolling_ms = median_ms(lambda: rolling.step(patch), args.runs)
    full_ms = median_ms(lambda: full.step(window), args.runs)
    result = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "context_length": args.context_length,
        "context_tokens": eager.num_tokens,
        "official_full_max_error": official_error,
        "full_graph_max_error": full_error,
        "rolling_graph_max_error": rolling_error,
        "graph_full_median_ms": full_ms,
        "graph_rolling_median_ms": rolling_ms,
        "rolling_vs_full_graph_speedup": full_ms / rolling_ms,
    }
    print(json.dumps(result, indent=2))
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as handle:
            json.dump(result, handle, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--context-length", type=int, default=12288)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output")
    main(parser.parse_args())
