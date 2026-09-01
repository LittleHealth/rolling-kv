"""Correctness and latency smoke benchmark for Lag-Llama online cache."""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lag_llama.online import (
    CudaGraphFullLagLlamaStep,
    CudaGraphRollingLagLlamaStep,
    LagLlamaRollingConfig,
    RollingLagLlamaEngine,
    load_pretrained_lag_llama,
)


def median_ms(fn, runs):
    for _ in range(5): fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(runs):
        start = time.perf_counter(); fn(); torch.cuda.synchronize()
        values.append((time.perf_counter() - start) * 1000)
    return float(torch.tensor(values).median())


def main(args):
    torch.manual_seed(args.seed)
    model = load_pretrained_lag_llama(args.checkpoint, args.context_length, args.device)
    cfg = LagLlamaRollingConfig(context_length=args.context_length, device=args.device)
    engine = RollingLagLlamaEngine(model, cfg)
    raw = torch.randn(1, engine.total_length, device=args.device).cumsum(-1) * 0.03
    times = torch.zeros(1, engine.total_length, 6, device=args.device)
    initial = engine.full_refresh(raw, times)

    official_params, loc, scale = model(
        raw, torch.ones_like(raw), times,
        torch.zeros(1, 1, 6, device=args.device), use_kv_cache=False,
    )
    official = official_params[1][..., -1] * scale.squeeze(-1) + loc.squeeze(-1)
    official_error = (initial - official).abs().max().item()

    graph_engine = RollingLagLlamaEngine(model, cfg)
    graph_engine.full_refresh(raw, times)
    rolling = CudaGraphRollingLagLlamaStep(graph_engine); rolling.capture()
    full = CudaGraphFullLagLlamaStep(graph_engine); full.capture()
    full_error = (full.step(raw, times) - initial).abs().max().item()
    values = torch.randn(args.steps, 1, device=args.device)
    features = torch.zeros(args.steps, 1, 6, device=args.device)
    eager_out, graph_out = [], []
    for value, feature in zip(values, features):
        eager_out.append(engine.fast_update(value, feature).clone())
        graph_out.append(rolling.step(value, feature).clone())
    torch.cuda.synchronize()
    rolling_error = (torch.stack(eager_out) - torch.stack(graph_out)).abs().max().item()
    rolling_ms = median_ms(lambda: rolling.step(values[-1], features[-1]), args.runs)
    full_ms = median_ms(lambda: full.step(raw, times), args.runs)
    result = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "context_length": args.context_length,
        "max_lag": engine.max_lag,
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
        with open(args.output, "w") as handle: json.dump(result, handle, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output")
    main(parser.parse_args())
