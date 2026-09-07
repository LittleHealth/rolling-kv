"""Run EXP-5 eager/graph, batch scaling, and Sundial TimeFlow appendices."""

from __future__ import annotations

import argparse
import time
import traceback
from typing import Any, Callable

import numpy as np
import torch

from adapters import GraphPair
from common import (
    MODELS,
    RESULTS,
    append_jsonl,
    base_record,
    classify_failure,
    load_manifest,
    load_series,
    metric_summary,
    quality_metrics,
    read_jsonl,
    stable_policy_id,
)
from exp1_w1 import execute_policy, make_initial_and_updates


def synthetic_batch(batch: int, length: int, step: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(71)
    values = rng.randn(batch, length + step).astype(np.float32).cumsum(-1) * 0.03
    return values[:, :length], values[:, length:]


def measure(fn: Callable[[], torch.Tensor], warmup: int, runs: int) -> dict[str, Any]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(runs):
        begin = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - begin) * 1000.0)
    return metric_summary(samples)


def appendix_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row.get("L"), row.get("path"), row.get("exec"), row.get("batch"))


def run_eager_graph(model: str, batch: int) -> None:
    spec = MODELS[model]
    length = spec.main_length
    path = RESULTS / "EXP5_appendix" / model / "eager_graph.jsonl"
    latest = {appendix_key(row): row for row in read_jsonl(path)}
    initial, update = synthetic_batch(batch, length, spec.s)
    try:
        pair = GraphPair(
            model, length, initial, pos_remap=spec.pos_remap, batch_size=batch
        )
    except Exception as exc:
        status = classify_failure(exc)
        reason = f"{type(exc).__name__}: {str(exc)[:1200]}"
        for path_name in ("full", "rolling"):
            for execution in ("eager", "graph"):
                key = (length, path_name, execution, batch)
                if latest.get(key, {}).get("status") in {
                    "ok", "oom", "unsupported"
                }:
                    continue
                row = base_record("C", "EXP5", model)
                row.update(
                    {
                        "status": status,
                        "reason": reason,
                        "L": length,
                        "path": path_name,
                        "exec": execution,
                        "batch": batch,
                        "peak_mem_mb": (
                            torch.cuda.max_memory_allocated() / (1024**2)
                            if torch.cuda.is_available()
                            else None
                        ),
                    }
                )
                append_jsonl(path, row)
                print(
                    f"EXP5 {model} L={length} batch={batch} "
                    f"{path_name}/{execution}: {status}",
                    flush=True,
                )
        return
    for path_name in ("full", "rolling"):
        for execution in ("eager", "graph"):
            key = (length, path_name, execution, batch)
            if latest.get(key, {}).get("status") in {"ok", "oom", "unsupported"}:
                continue
            try:
                if execution == "graph":
                    pair.reset(initial)
                    fn = (
                        (lambda: pair.full_step(initial))
                        if path_name == "full"
                        else (lambda: pair.rolling_step(update))
                    )
                else:
                    eager = pair.new_eager_engine()
                    pair.eager_full(eager, initial)
                    fn = (
                        (lambda: pair.eager_full(eager, initial))
                        if path_name == "full"
                        else (lambda: pair.eager_roll(eager, update))
                    )
                summary = measure(fn, 10, 20 if path_name == "full" else 50)
                row = base_record("C", "EXP5", model)
                row.update(
                    {
                        "timing_id": f"EXP5-{model}-L{length}-{path_name}-{execution}-b{batch}",
                        "L": length,
                        "s": spec.s,
                        "U": spec.updates,
                        "policy": path_name,
                        "K": 1 if path_name == "full" else 0,
                        "policy_id": None,
                        "pos_remap": spec.pos_remap,
                        "path": path_name,
                        "exec": execution,
                        "batch": batch,
                        "latency_ms": summary,
                        "t_full_ms": summary if path_name == "full" else None,
                        "t_roll_ms": summary if path_name == "rolling" else None,
                        "t_update_computed_ms": None,
                        "t_update_measured_ms": {
                            "mean": summary["mean"], "p95": summary["p95"],
                            "p99": summary["p99"], "n": summary["n"],
                        },
                        "us_per_point": summary["mean"] / spec.s * 1000.0,
                        "timing_flag": None,
                        "graph_capture_ok": execution == "graph",
                        "n_kernels_full": None,
                        "n_kernels_roll": None,
                        "kv_cache_mb": pair.cache_mb(),
                        "peak_mem_mb": pair.peak_mem_mb,
                        "trigger_overhead_ms": None,
                    }
                )
            except Exception as exc:
                row = base_record("C", "EXP5", model)
                row.update(
                    {
                        "status": classify_failure(exc),
                        "reason": f"{type(exc).__name__}: {str(exc)[:1200]}",
                        "L": length,
                        "path": path_name,
                        "exec": execution,
                        "batch": batch,
                    }
                )
            append_jsonl(path, row)
            print(
                f"EXP5 {model} L={length} batch={batch} {path_name}/{execution}: "
                f"{row['status']}",
                flush=True,
            )


def timeflow_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row.get("sampling_steps"), row.get("K"), row.get("window"))


def run_timeflow() -> None:
    model = "sundial"
    spec = MODELS[model]
    path = RESULTS / "EXP5_appendix" / model / "timeflow.jsonl"
    latest = {timeflow_key(row): row for row in read_jsonl(path)}
    manifest = load_manifest()
    dataset, window_index, length = "ETTh1", 0, spec.main_length
    start = int(manifest["windows"][dataset][window_index])
    series = load_series(dataset)
    initial, updates, targets, _ = make_initial_and_updates(
        series, start, spec, length
    )
    for sampling_steps in (1, 2, 5, 10, 20, 50):
        pending = [k for k in (1, 4, 16) if latest.get((sampling_steps, k, 0), {}).get("status") not in {"ok", "oom", "unsupported"}]
        if not pending:
            continue
        try:
            pair = GraphPair(
                model,
                length,
                initial,
                pos_remap="on",
                sundial_steps=sampling_steps,
                sundial_samples=5,
            )
            predictions: dict[int, np.ndarray] = {}
            counts: dict[int, tuple[int, int]] = {}
            for k in (1, 4, 16):
                pred, _, n_full, n_roll = execute_policy(
                    pair, initial, updates, k, collect=True
                )
                predictions[k], counts[k] = pred, (n_full, n_roll)
            reference = predictions[1]
            for k in pending:
                payload = {
                    "exp": "EXP5", "model": model, "sampling_steps": sampling_steps,
                    "num_samples": 5, "K": k, "dataset": dataset, "L": length,
                }
                row = base_record("B", "EXP5", model)
                row.update(
                    {
                        "exp": "EXP5",
                        "dataset": dataset,
                        "window": window_index,
                        "window_start": start,
                        "L": length,
                        "s": spec.s,
                        "H": spec.horizon,
                        "U": spec.updates,
                        "policy": "fixed",
                        "K": k,
                        "tau_pts": k * spec.s,
                        "pos_remap": "on",
                        "norm_mode": "frozen_at_refresh",
                        "adaptive_params": None,
                        "policy_id": stable_policy_id(payload),
                        "sampling_steps": sampling_steps,
                        "num_samples": 5,
                        "noise_mode": "antithetic",
                        "n_full": counts[k][0],
                        "n_roll": counts[k][1],
                        **quality_metrics(predictions[k], targets, reference),
                    }
                )
                if k == 1:
                    row.update(
                        {
                            "gap_pct": 0.0, "mae_delta_pct": 0.0,
                            "mae_delta_h32_pct": 0.0, "mse_delta_pct": 0.0,
                        }
                    )
                append_jsonl(path, row)
        except Exception as exc:
            traceback.print_exc()
            for k in pending:
                row = base_record("B", "EXP5", model)
                row.update(
                    {
                        "status": classify_failure(exc),
                        "reason": f"{type(exc).__name__}: {str(exc)[:1200]}",
                        "dataset": dataset, "window": window_index, "L": length,
                        "sampling_steps": sampling_steps, "num_samples": 5, "K": k,
                    }
                )
                append_jsonl(path, row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("eager-graph", "timeflow"), required=True)
    parser.add_argument("--model", choices=tuple(MODELS))
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()
    if args.mode == "timeflow":
        run_timeflow()
    else:
        if not args.model:
            parser.error("eager-graph mode requires --model")
        run_eager_graph(args.model, args.batch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
