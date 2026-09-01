"""Correctness and latency smoke test for Toto 2.0 online rolling cache."""

import argparse
import json
import os
import sys
import time

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "toto2"))
sys.path.insert(0, os.path.join(ROOT, "dd_unit_scaling"))

from toto2 import Toto2Model
from toto2.online_rolling import (
    CudaGraphFullToto2Step,
    CudaGraphRollingToto2Step,
    RollingToto2Engine,
    Toto2RollingConfig,
)


def timed(fn, runs):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(runs):
        begin = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - begin) * 1000)
    return float(torch.tensor(samples).median())


def main(args):
    torch.manual_seed(args.seed)
    model = Toto2Model.from_pretrained(args.checkpoint, map_location="cpu")
    model = model.to(args.device).eval()
    cfg = Toto2RollingConfig(
        context_length=args.context_length,
        horizon=args.horizon,
        device=args.device,
    )
    base = torch.randn(1, 1, args.context_length, device=args.device).cumsum(-1) * 0.03
    patches = torch.randn(args.steps, 1, 1, model.config.patch_size, device=args.device)

    eager = RollingToto2Engine(model, cfg)
    initial = eager.full_refresh(base)
    official = model.forecast(
        {
            "target": base,
            "target_mask": torch.ones_like(base, dtype=torch.bool),
            "series_ids": torch.zeros(1, 1, dtype=torch.long, device=args.device),
        },
        horizon=args.horizon,
        decode_block_size=None,
        has_missing_values=False,
    )[4]
    official_error = (initial - official).abs().max().item()

    graph_engine = RollingToto2Engine(model, cfg)
    graph_engine.full_refresh(base)
    runner = CudaGraphRollingToto2Step(graph_engine)
    runner.capture()
    full_runner = CudaGraphFullToto2Step(graph_engine)
    full_runner.capture()
    full_graph_error = (full_runner.step(base) - initial).abs().max().item()

    eager_outputs, graph_outputs = [], []
    for patch in patches:
        eager_outputs.append(eager.fast_update(patch).clone())
        graph_outputs.append(runner.step(patch).clone())
    torch.cuda.synchronize()
    graph_error = (torch.stack(eager_outputs) - torch.stack(graph_outputs)).abs().max().item()

    eager_speed_engine = RollingToto2Engine(model, cfg)
    eager_speed_engine.full_refresh(base)
    patch = patches[-1]
    eager_ms = timed(lambda: eager_speed_engine.fast_update(patch), args.runs)
    graph_ms = timed(lambda: runner.step(patch), args.runs)
    full_ms = timed(lambda: full_runner.step(base), args.runs)

    result = {
        "checkpoint": args.checkpoint,
        "parameters": sum(p.numel() for p in model.parameters()),
        "context_length": args.context_length,
        "context_patches": eager.num_patches,
        "patch_size": model.config.patch_size,
        "horizon": args.horizon,
        "official_full_max_error": official_error,
        "rolling_graph_max_error": graph_error,
        "full_graph_max_error": full_graph_error,
        "eager_rolling_median_ms": eager_ms,
        "graph_rolling_median_ms": graph_ms,
        "graph_full_median_ms": full_ms,
        "graph_vs_eager_speedup": eager_ms / graph_ms,
        "rolling_vs_full_graph_speedup": full_ms / graph_ms,
    }
    print(json.dumps(result, indent=2))
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as handle:
            json.dump(result, handle, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output")
    main(parser.parse_args())
