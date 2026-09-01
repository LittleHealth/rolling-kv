"""Profile EXP-2 model stages and appendix kernel classes for one grid cell."""

from __future__ import annotations

import argparse
import gc
import time
import traceback
from collections import defaultdict
from contextlib import ExitStack
from typing import Any

import numpy as np
import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile, record_function

from adapters import GraphPair
from common import MODELS, RESULTS, append_jsonl, base_record, classify_failure, read_jsonl


STAGES = ("S1_norm", "S2_embed", "S3_attn", "S4_ffn", "S5_head", "S6_cache")


def synthetic_batch(batch: int, length: int, step: int, seed: int = 19) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    x = np.arange(length + step, dtype=np.float32)
    base = np.sin(2 * np.pi * x / 96) + 0.3 * np.sin(2 * np.pi * x / 336)
    values = np.stack(
        [base + 0.1 * rng.randn(length + step).astype(np.float32) for _ in range(batch)]
    )
    return values[:, :length].astype(np.float32), values[:, length:].astype(np.float32)


def stage_for_module(name: str, module: torch.nn.Module) -> str | None:
    text = f"{name}.{type(module).__name__}".lower()
    if any(token in text for token in ("output_head", "param_proj", ".head", "flow_sampler")):
        return "S5_head"
    if any(token in text for token in ("attention", "self_attn", ".attn")):
        return "S3_attn"
    if any(token in text for token in ("feedforward", "feed_forward", ".ffn", ".mlp")):
        return "S4_ffn"
    if any(token in text for token in ("embedding", ".embed", ".wte")):
        return "S2_embed"
    return None


def install_stage_hooks(model: torch.nn.Module) -> tuple[list[Any], dict[str, str]]:
    candidates: list[tuple[str, torch.nn.Module, str]] = []
    for name, module in model.named_modules():
        if not name:
            continue
        stage = stage_for_module(name, module)
        if stage:
            candidates.append((name, module, stage))
    # Keep top-most modules for each stage to prevent nested ranges from
    # double-counting their CUDA children.
    selected: list[tuple[str, torch.nn.Module, str]] = []
    for name, module, stage in sorted(candidates, key=lambda item: item[0].count(".")):
        if any(stage == old_stage and (name == old or name.startswith(old + ".")) for old, _, old_stage in selected):
            continue
        selected.append((name, module, stage))

    handles: list[Any] = []
    stage_map: dict[str, str] = {}
    active: dict[int, ExitStack] = {}

    def pre(module: torch.nn.Module, _inputs: Any, stage: str) -> None:
        stack = ExitStack()
        stack.enter_context(record_function(stage))
        active[id(module)] = stack

    def post(module: torch.nn.Module, _inputs: Any, _output: Any) -> None:
        stack = active.pop(id(module), None)
        if stack is not None:
            stack.close()

    for name, module, stage in selected:
        handles.append(module.register_forward_pre_hook(lambda m, i, s=stage: pre(m, i, s)))
        handles.append(module.register_forward_hook(post))
        stage_map[name] = stage
    return handles, stage_map


def kernel_class(name: str) -> str:
    text = name.lower()
    if any(x in text for x in ("gemm", "matmul", "mm", "linear")):
        return "matmul"
    if any(x in text for x in ("attention", "flash", "scaled_dot")):
        return "attention"
    if any(x in text for x in ("norm", "layer_norm", "rms")):
        return "norm"
    if any(x in text for x in ("copy", "index", "slice", "cat", "gather", "scatter")):
        return "index_mem"
    if any(x in text for x in ("elementwise", "vectorized", "relu", "gelu", "silu", "mul", "add")):
        return "elementwise"
    return "other"


def graph_wall(pair: GraphPair, path: str, initial: np.ndarray, update: np.ndarray) -> dict[str, float]:
    pair.reset(initial)
    fn = (lambda: pair.full_step(initial)) if path == "full" else (lambda: pair.rolling_step(update))
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    values = []
    runs = 20 if path == "full" else 50
    for _ in range(runs):
        begin = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        values.append((time.perf_counter() - begin) * 1000.0)
    return {
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "n": len(values),
    }


def completed(model: str, length: int, path: str, batch: int) -> bool:
    latest = {}
    result_path = RESULTS / "EXP2_stages" / model / "stages.jsonl"
    for row in read_jsonl(result_path):
        latest[(row.get("L"), row.get("path"), row.get("batch"), row.get("method"))] = row
    row = latest.get((length, path, batch, "profiler_eager"))
    return bool(row and row.get("status") in {"ok", "oom", "unsupported"})


def run_path(model: str, length: int, path_name: str, batch: int) -> None:
    spec = MODELS[model]
    initial, update = synthetic_batch(batch, length, spec.s)
    pair = GraphPair(model, length, initial, pos_remap=spec.pos_remap, batch_size=batch)
    wall = graph_wall(pair, path_name, initial, update)
    eager = pair.new_eager_engine()
    pair.eager_full(eager, initial)
    handles, module_map = install_stage_hooks(pair.module)
    try:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            if path_name == "full":
                pair.eager_full(eager, initial)
            else:
                pair.eager_roll(eager, update)
            torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()

    stage_us: dict[str, float] = defaultdict(float)
    for event in prof.events():
        # Profiler emits both the host record_function range and a synthetic
        # CUDA annotation with the same name.  Only the host range owns the
        # descendant kernels; summing both double-counts every stage.
        if event.name in STAGES and event.device_type == DeviceType.CPU:
            stage_us[event.name] += float(event.device_time_total)
    busy_us = float(
        sum(
            event.self_device_time_total
            for event in prof.events()
            if event.device_type == DeviceType.CUDA and event.name not in STAGES
        )
    )
    mapped_us = sum(stage_us.values())
    if mapped_us > busy_us > 0:
        scale = busy_us / mapped_us
        stage_us = defaultdict(float, {key: value * scale for key, value in stage_us.items()})
        mapped_us = busy_us
    stage_ms = {stage: stage_us[stage] / 1000.0 for stage in STAGES}
    stage_ms["S7_other"] = max(0.0, busy_us - mapped_us) / 1000.0
    stage_sum = sum(stage_ms.values())
    consistency = abs(stage_sum - wall["median"]) / max(wall["median"], 1e-12) * 100.0
    stage_map = {
        "S1_norm": "input statistics and normalization; unhooked kernels are conservatively included in S7_other",
        "S2_embed": "embedding/patch embedding modules",
        "S3_attn": "attention modules, including QKV projection, score and aggregation",
        "S4_ffn": "MLP/FFN/MoE modules",
        "S5_head": "forecast/output/flow head modules",
        "S6_cache": "cache eviction, position repair and KV writes; unhooked kernels are conservatively included in S7_other",
        "S7_other": "all profiler CUDA busy time not owned by a non-overlapping module range",
        "module_ranges": module_map,
    }
    row = base_record("D", "EXP2", model)
    row.update(
        {
            "L": length,
            "N_tokens": length // spec.s,
            "batch": batch,
            "path": path_name,
            "method": "profiler_eager",
            "pos_remap": spec.pos_remap,
            "stage_map": stage_map,
            "stage_busy_ms": stage_ms,
            "stage_sum_ms": stage_sum,
            "graph_wall_ms": wall["median"],
            "graph_wall_summary_ms": wall,
            "consistency_pct": consistency,
            "flag": "inconsistent" if consistency > 15.0 else None,
            "n_runs": 1,
        }
    )
    append_jsonl(RESULTS / "EXP2_stages" / model / "stages.jsonl", row)

    classes: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for event in prof.events():
        if (
            event.device_type != DeviceType.CUDA
            or event.name in STAGES
            or event.self_device_time_total <= 0
        ):
            continue
        category = kernel_class(event.name)
        classes[category] += float(event.self_device_time_total) / 1000.0
        counts[category] += 1
    kernel_row = base_record("D", "EXP2", model)
    kernel_row.update(
        {
            "L": length,
            "batch": batch,
            "path": path_name,
            "method": "profiler_eager_kernel_classes",
            "kernel_busy_ms": dict(classes),
            "kernel_counts": dict(counts),
            "graph_wall_ms": wall["median"],
        }
    )
    append_jsonl(RESULTS / "EXP2_stages" / model / "kernels.jsonl", kernel_row)
    print(
        f"EXP2 {model} L={length} batch={batch} {path_name}: "
        f"stage={stage_sum:.4f}ms graph={wall['median']:.4f}ms "
        f"consistency={consistency:.2f}%",
        flush=True,
    )


def record_failure(model: str, length: int, path_name: str, batch: int, exc: BaseException) -> None:
    row = base_record("D", "EXP2", model)
    row.update(
        {
            "status": classify_failure(exc),
            "reason": f"{type(exc).__name__}: {str(exc)[:1200]}",
            "L": length,
            "N_tokens": length // MODELS[model].s,
            "batch": batch,
            "path": path_name,
            "method": "profiler_eager",
            "stage_map": {},
            "stage_busy_ms": None,
            "stage_sum_ms": None,
            "graph_wall_ms": None,
            "consistency_pct": None,
            "flag": None,
            "n_runs": 0,
        }
    )
    append_jsonl(RESULTS / "EXP2_stages" / model / "stages.jsonl", row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--L", type=int, required=True)
    parser.add_argument("--path", choices=("full", "rolling", "both"), default="both")
    parser.add_argument("--batch", type=int, default=1)
    # Mirrors exp1_w1.py: supplementary contexts run without touching
    # spec.lengths, because launch_all.py builds its task list from that tuple
    # and only resumes while task_count matches, so widening the grid would
    # restart the main queue at index 0.
    parser.add_argument(
        "--allow-off-grid",
        action="store_true",
        help="permit an --L outside the model spec (supplementary sweeps)",
    )
    args = parser.parse_args()
    if args.L not in MODELS[args.model].lengths and not args.allow_off_grid:
        parser.error(
            f"L={args.L} is not in the {args.model} grid; "
            "pass --allow-off-grid to run a supplementary context"
        )
    paths = ("full", "rolling") if args.path == "both" else (args.path,)
    failed = False
    for path_name in paths:
        if completed(args.model, args.L, path_name, args.batch):
            print(f"EXP2 already complete: {args.model} L={args.L} {path_name} b={args.batch}")
            continue
        try:
            run_path(args.model, args.L, path_name, args.batch)
        except Exception as exc:
            traceback.print_exc()
            record_failure(args.model, args.L, path_name, args.batch, exc)
            failed = True
        gc.collect()
        torch.cuda.empty_cache()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
