"""Run one resumable W1 EXP-1 timing or quality cell using CUDA Graphs."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from adapters import GraphPair
from common import (
    CHECKPOINTS,
    DATASETS,
    MODELS,
    MODELS_ROOT,
    RESULTS,
    append_jsonl,
    base_record,
    classify_failure,
    completed_keys,
    exp1_key,
    load_manifest,
    load_series,
    metric_summary,
    quality_metrics,
    read_jsonl,
    stable_policy_id,
    timing_key,
)


def sync() -> None:
    torch.cuda.synchronize()


def synthetic_series(length: int, seed: int = 7) -> np.ndarray:
    rng = np.random.RandomState(seed)
    x = np.arange(length, dtype=np.float32)
    return (
        np.sin(2 * np.pi * x / 96)
        + 0.5 * np.sin(2 * np.pi * x / 336)
        + 0.2 * rng.randn(length).astype(np.float32)
    ).astype(np.float32)




def policy_payload(
    model: str, context_length: int, k: int, pos_remap: str | None = None
) -> dict[str, Any]:
    spec = MODELS[model]
    return {
        "exp": "EXP1",
        "model": model,
        "L": context_length,
        "policy": "naive" if k == 0 else "fixed",
        "K": k,
        "pos_remap": pos_remap or spec.pos_remap,
    }


def timing_id(
    model: str, context_length: int, k: int, pos_remap: str | None = None
) -> str:
    payload = policy_payload(model, context_length, k, pos_remap)
    return f"EXP1-{model}-L{context_length}-{stable_policy_id(payload)}"


def wanted_k(spec, k_set: str) -> list[int]:
    if k_set == "baseline":
        return [1]
    if k_set == "rest":
        return [k for k in spec.k_values if k != 1]
    return list(spec.k_values)


def requested_k_values(args: argparse.Namespace) -> list[int]:
    """Return an optional explicit K subset, otherwise the regular K group."""
    if args.k_values is not None:
        return list(args.k_values)
    return wanted_k(MODELS[args.model], args.k_set)


def make_initial_and_updates(
    series: np.ndarray, start: int, spec, context_length: int
) -> tuple[torch.Tensor, list[torch.Tensor], np.ndarray, np.ndarray]:
    initial = torch.as_tensor(
        series[start - context_length : start][None, :], device="cuda", dtype=torch.float32
    )
    updates = [
        torch.as_tensor(
            series[start + i * spec.s : start + (i + 1) * spec.s][None, :],
            device="cuda",
            dtype=torch.float32,
        )
        for i in range(spec.updates)
    ]
    targets = np.stack(
        [
            series[
                start + (i + 1) * spec.s :
                start + (i + 1) * spec.s + spec.horizon
            ]
            for i in range(spec.updates)
        ]
    ).astype(np.float32)
    t_index = np.arange(
        start + spec.s,
        start + (spec.updates + 1) * spec.s,
        spec.s,
        dtype=np.int64,
    )
    return initial, updates, targets, t_index


def advance_window(window: torch.Tensor, update: torch.Tensor, step: int) -> torch.Tensor:
    return torch.cat((window[:, step:], update), dim=1)


def execute_policy(
    pair: GraphPair,
    initial: torch.Tensor,
    updates: list[torch.Tensor],
    k: int,
    collect: bool,
) -> tuple[np.ndarray | None, list[float], int, int]:
    pair.reset(initial)
    window = initial.clone()
    predictions = [] if collect else None
    latencies: list[float] = []
    n_full = 0
    for index, update in enumerate(updates):
        window = advance_window(window, update, pair.spec.s)
        use_full = k > 0 and (index + 1) % k == 0
        begin = time.perf_counter()
        prediction = pair.full_step(window) if use_full else pair.rolling_step(update)
        sync()
        latencies.append((time.perf_counter() - begin) * 1000.0)
        if collect:
            predictions.append(pair.prediction_numpy(prediction)[0].copy())
        n_full += int(use_full)
    stacked = np.stack(predictions).astype(np.float32) if collect else None
    return stacked, latencies, n_full, len(updates) - n_full


def time_replays(
    reset: Callable[[], None], fn: Callable[[int], torch.Tensor], warmup: int, runs: int
) -> dict[str, Any]:
    reset()
    for index in range(warmup):
        fn(index)
    sync()
    values = []
    for index in range(runs):
        begin = time.perf_counter()
        fn(index)
        sync()
        values.append((time.perf_counter() - begin) * 1000.0)
    return metric_summary(values)


def kernel_count(fn: Callable[[], torch.Tensor]) -> int | None:
    try:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            fn()
            sync()
        return int(sum(event.count for event in prof.key_averages() if event.self_device_time_total > 0))
    except Exception:
        return None


def run_timing(args: argparse.Namespace) -> None:
    spec = MODELS[args.model]
    requested = requested_k_values(args)
    path = RESULTS / "EXP1_sweep" / args.model / "timing.jsonl"
    completed = completed_keys(path, timing_key)
    pending = [
        k
        for k in requested
        if (args.L, k, args.pos_remap, "graph", 1) not in completed
    ]
    if not pending:
        print(f"timing already complete: {args.model} L={args.L}", flush=True)
        return

    needed = args.L + spec.updates * spec.s + spec.horizon + 8 * spec.s
    series = synthetic_series(needed)
    start = args.L
    initial, updates, _, _ = make_initial_and_updates(series, start, spec, args.L)
    pair = GraphPair(args.model, args.L, initial, pos_remap=args.pos_remap)
    repeated_updates = updates if len(updates) >= 64 else (updates * 64)[:64]

    full_stats = time_replays(
        lambda: pair.reset(initial),
        lambda _: pair.full_step(initial),
        warmup=10,
        runs=20,
    )
    rolling_stats = time_replays(
        lambda: pair.reset(initial),
        lambda i: pair.rolling_step(repeated_updates[i % len(repeated_updates)]),
        warmup=10,
        runs=50,
    )
    pair.reset(initial)
    n_kernels_full = kernel_count(lambda: pair.full_step(initial))
    pair.reset(initial)
    n_kernels_roll = kernel_count(lambda: pair.rolling_step(repeated_updates[0]))

    policy_rows = []
    baseline_mean = None
    for k in requested:
        if (args.L, k, args.pos_remap, "graph", 1) in completed:
            continue
        _, latencies, n_full, n_roll = execute_policy(
            pair, initial, updates, k, collect=False
        )
        measured = metric_summary(latencies)
        computed = (
            n_full * full_stats["median"] + n_roll * rolling_stats["median"]
        ) / spec.updates
        if k == 1:
            baseline_mean = measured["mean"]
        policy_rows.append((k, measured, computed, n_full, n_roll))

    if baseline_mean is None:
        old = [
            row
            for row in __import__("common").read_jsonl(path)
            if row.get("L") == args.L
            and row.get("K") == 1
            and row.get("pos_remap") == args.pos_remap
            and row.get("status") == "ok"
        ]
        if old:
            baseline_mean = old[-1]["t_update_measured_ms"]["mean"]
    if baseline_mean is None:
        raise RuntimeError("K=1 timing baseline is unavailable")

    for k, measured, computed, n_full, n_roll in policy_rows:
        inconsistency = abs(computed - measured["mean"]) / max(measured["mean"], 1e-12)
        payload = policy_payload(args.model, args.L, k, args.pos_remap)
        row = base_record("C", "EXP1", args.model)
        row.update(
            {
                "timing_id": timing_id(args.model, args.L, k, args.pos_remap),
                "L": args.L,
                "s": spec.s,
                "U": spec.updates,
                "policy": payload["policy"],
                "K": k,
                "policy_id": stable_policy_id(payload),
                "pos_remap": args.pos_remap,
                "exec": "graph",
                "batch": 1,
                "t_full_ms": full_stats,
                "t_roll_ms": rolling_stats,
                "t_update_computed_ms": computed,
                "t_update_measured_ms": {
                    "mean": measured["mean"],
                    "p95": measured["p95"],
                    "p99": measured["p99"],
                    "n": measured["n"],
                },
                "speedup": 1.0 if k == 1 else baseline_mean / measured["mean"],
                "us_per_point": measured["mean"] / spec.s * 1000.0,
                "timing_flag": "inconsistent" if inconsistency > 0.10 else None,
                "graph_capture_ok": True,
                "n_kernels_full": n_kernels_full,
                "n_kernels_roll": n_kernels_roll,
                "kv_cache_mb": pair.cache_mb(),
                "peak_mem_mb": pair.peak_mem_mb,
                "trigger_overhead_ms": None,
                "n_full": n_full,
                "n_roll": n_roll,
            }
        )
        append_jsonl(path, row)
        print(
            f"timing {args.model} L={args.L} K={k}: "
            f"{measured['mean']:.4f} ms, {row['speedup']:.3f}x",
            flush=True,
        )


def latest_timing(
    model: str, context_length: int, pos_remap: str
) -> dict[int, dict[str, Any]]:
    from common import read_jsonl

    path = RESULTS / "EXP1_sweep" / model / "timing.jsonl"
    result = {}
    for row in read_jsonl(path):
        if (
            row.get("L") == context_length
            and row.get("pos_remap") == pos_remap
            and row.get("status") == "ok"
        ):
            result[int(row["K"])] = row
    return result


def prediction_relative_path(
    model: str,
    dataset: str,
    window: int,
    context_length: int,
    k: int,
    pos_remap: str | None = None,
) -> Path:
    remap = pos_remap or MODELS[model].pos_remap
    tag = "na" if remap == "n/a" else remap
    return Path("EXP1_sweep") / model / "preds" / (
        f"{dataset}_w{window}_L{context_length}_K{k}_remap{tag}.npz"
    )


def quality_record(
    model: str,
    dataset: str,
    window_index: int,
    window_start: int,
    context_length: int,
    k: int,
    pos_remap: str | None = None,
) -> dict[str, Any]:
    spec = MODELS[model]
    remap = pos_remap or spec.pos_remap
    payload = policy_payload(model, context_length, k, remap)
    row = base_record("B", "EXP1", model)
    row.update(
        {
            "exp": "EXP1",
            "dataset": dataset,
            "window": window_index,
            "window_start": window_start,
            "L": context_length,
            "s": spec.s,
            "H": spec.horizon,
            "U": spec.updates,
            "policy": payload["policy"],
            "K": k,
            "tau_pts": k * spec.s if k > 0 else None,
            "pos_remap": remap,
            "norm_mode": "running_prefix_frozen" if model == "timesfm" else "frozen_at_refresh",
            "adaptive_params": None,
            "policy_id": stable_policy_id(payload),
            "n_full": None,
            "n_roll": None,
            "mae_native": None,
            "mse_native": None,
            "mae_h32": None,
            "mse_h32": None,
            "mae_ref_native": None,
            "mae_ref_h32": None,
            "gap_pct": None,
            "mae_delta_pct": None,
            "mae_delta_h32_pct": None,
            "mse_delta_pct": None,
            "speedup": None,
            "preds_file": str(
                prediction_relative_path(
                    model, dataset, window_index, context_length, k, remap
                )
            ),
            "timing_ref": timing_id(model, context_length, k, remap),
        }
    )
    return row


def save_predictions(path: Path, yhat: np.ndarray, y: np.ndarray, t_index: np.ndarray, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}.npz")
    np.savez_compressed(
        temp,
        yhat=np.asarray(yhat, dtype=np.float32),
        y=np.asarray(y, dtype=np.float32),
        t_index=np.asarray(t_index, dtype=np.int64),
        meta=np.asarray(json.dumps(meta, ensure_ascii=False, sort_keys=True)),
    )
    os.replace(temp, path)


def record_quality_failure(args: argparse.Namespace, ks: list[int], exc: BaseException) -> None:
    spec = MODELS[args.model]
    try:
        manifest = load_manifest()
        start = int(manifest["windows"][args.dataset][args.window])
    except Exception:
        start = -1
    status = classify_failure(exc)
    reason = f"{type(exc).__name__}: {str(exc)[:1200]}"
    path = RESULTS / "EXP1_sweep" / args.model / "records.jsonl"
    for k in ks:
        row = quality_record(
            args.model, args.dataset, args.window, start, args.L, k, args.pos_remap
        )
        row.update({"status": status, "reason": reason})
        if status == "oom":
            row["peak_mem_mb"] = torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else None
        append_jsonl(path, row)


def run_quality(args: argparse.Namespace) -> None:
    spec = MODELS[args.model]
    ks = requested_k_values(args)
    path = RESULTS / "EXP1_sweep" / args.model / "records.jsonl"
    completed = completed_keys(path, exp1_key)
    ks = [
        k
        for k in ks
        if (args.dataset, args.window, args.L, k, args.pos_remap) not in completed
    ]
    if not ks:
        print(
            f"quality already complete: {args.model} {args.dataset} "
            f"w={args.window} L={args.L} {args.k_set}",
            flush=True,
        )
        return

    manifest = load_manifest()
    start = int(manifest["windows"][args.dataset][args.window])
    series = load_series(args.dataset)
    end = start + spec.updates * spec.s + spec.horizon
    if start < args.L or end > len(series):
        reason = (
            f"dataset bounds: require history start-L >= 0 and future end <= {len(series)}; "
            f"got start={start}, L={args.L}, end={end}"
        )
        for k in ks:
            row = quality_record(
                args.model, args.dataset, args.window, start, args.L, k, args.pos_remap
            )
            row.update({"status": "unsupported", "reason": reason})
            append_jsonl(path, row)
        print(f"unsupported cell: {reason}", flush=True)
        return

    timings = latest_timing(args.model, args.L, args.pos_remap)
    missing_timings = [k for k in ks if k not in timings]
    if missing_timings:
        terminal: dict[int, dict[str, Any]] = {}
        for timing_row in read_jsonl(
            RESULTS / "EXP1_sweep" / args.model / "timing.jsonl"
        ):
            if (
                timing_row.get("L") == args.L
                and timing_row.get("pos_remap") == args.pos_remap
                and timing_row.get("K") in missing_timings
            ):
                terminal[int(timing_row["K"])] = timing_row
        if all(
            k in terminal
            and terminal[k].get("status")
            in {"oom", "unsupported", "capture_failed", "failed"}
            for k in missing_timings
        ):
            for k in ks:
                source = terminal.get(k)
                if source is None:
                    continue
                row = quality_record(
                    args.model,
                    args.dataset,
                    args.window,
                    start,
                    args.L,
                    k,
                    args.pos_remap,
                )
                row.update(
                    {
                        "status": source["status"],
                        "reason": "dependent timing unavailable: "
                        + str(source.get("reason", "unknown timing failure")),
                    }
                )
                append_jsonl(path, row)
            return
        raise RuntimeError(f"missing timing rows for K={missing_timings}")

    initial, updates, targets, t_index = make_initial_and_updates(
        series, start, spec, args.L
    )
    pair = GraphPair(args.model, args.L, initial, pos_remap=args.pos_remap)
    baseline_path = RESULTS / prediction_relative_path(
        args.model, args.dataset, args.window, args.L, 1, args.pos_remap
    )
    ref_yhat = None
    if baseline_path.exists():
        with np.load(baseline_path, allow_pickle=False) as archive:
            ref_yhat = archive["yhat"].copy()
    if 1 not in ks and ref_yhat is None:
        raise RuntimeError(f"missing mandatory K=1 prediction file: {baseline_path}")

    for k in ks:
        predictions, _, n_full, n_roll = execute_policy(
            pair, initial, updates, k, collect=True
        )
        if k == 1:
            ref_yhat = predictions
        if ref_yhat is None:
            raise RuntimeError("K=1 reference predictions are unavailable")
        row = quality_record(
            args.model, args.dataset, args.window, start, args.L, k, args.pos_remap
        )
        row.update(quality_metrics(predictions, targets, ref_yhat))
        row.update(
            {
                "n_full": n_full,
                "n_roll": n_roll,
                "speedup": timings[k]["speedup"],
            }
        )
        if k == 1:
            row.update(
                {
                    "gap_pct": 0.0,
                    "mae_delta_pct": 0.0,
                    "mae_delta_h32_pct": 0.0 if spec.horizon >= 32 else None,
                    "mse_delta_pct": 0.0,
                    "speedup": 1.0,
                }
            )
        output = RESULTS / row["preds_file"]
        save_predictions(output, predictions, targets, t_index, row)
        append_jsonl(path, row)
        print(
            f"quality {args.model} {args.dataset} w={args.window} "
            f"L={args.L} K={k}: MAE={row['mae_native']:.6f}",
            flush=True,
        )


def record_timing_failure(args: argparse.Namespace, exc: BaseException) -> None:
    spec = MODELS[args.model]
    path = RESULTS / "EXP1_sweep" / args.model / "timing.jsonl"
    status = classify_failure(exc)
    reason = f"{type(exc).__name__}: {str(exc)[:1200]}"
    for k in requested_k_values(args):
        row = base_record("C", "EXP1", args.model)
        row.update(
            {
                "status": status,
                "reason": reason,
                "timing_id": timing_id(args.model, args.L, k, args.pos_remap),
                "L": args.L,
                "s": spec.s,
                "U": spec.updates,
                "policy": "naive" if k == 0 else "fixed",
                "K": k,
                "policy_id": stable_policy_id(
                    policy_payload(args.model, args.L, k, args.pos_remap)
                ),
                "pos_remap": args.pos_remap,
                "exec": "graph",
                "batch": 1,
                "t_full_ms": None,
                "t_roll_ms": None,
                "t_update_computed_ms": None,
                "t_update_measured_ms": None,
                "speedup": None,
                "us_per_point": None,
                "timing_flag": None,
                "graph_capture_ok": False,
                "n_kernels_full": None,
                "n_kernels_roll": None,
                "kv_cache_mb": None,
                "peak_mem_mb": torch.cuda.max_memory_allocated() / (1024**2) if torch.cuda.is_available() else None,
                "trigger_overhead_ms": None,
            }
        )
        append_jsonl(path, row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("timing", "quality"), required=True)
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--L", type=int, required=True)
    parser.add_argument("--dataset", choices=tuple(DATASETS))
    parser.add_argument("--window", type=int)
    parser.add_argument("--k-set", choices=("baseline", "rest", "all"), default="all")
    parser.add_argument(
        "--k-values",
        help="optional comma-separated explicit K subset (for resumable supplements)",
    )
    parser.add_argument("--pos-remap", choices=("on", "off", "n/a"))
    # Supplementary contexts run without touching spec.lengths: launch_all.py
    # builds its task list from that tuple and only resumes while task_count
    # matches, so widening the grid would restart the main queue at index 0.
    parser.add_argument(
        "--allow-off-grid",
        action="store_true",
        help="permit an --L outside the model spec (supplementary sweeps)",
    )
    args = parser.parse_args()
    spec = MODELS[args.model]
    if args.k_values is not None:
        try:
            requested = [int(value) for value in args.k_values.split(",") if value]
        except ValueError:
            parser.error("--k-values must be a comma-separated list of integers")
        if not requested:
            parser.error("--k-values must not be empty")
        if len(set(requested)) != len(requested):
            parser.error("--k-values must not contain duplicates")
        invalid = [value for value in requested if value not in spec.k_values]
        if invalid:
            parser.error(f"K values {invalid} are not in the {args.model} grid")
        args.k_values = tuple(requested)
    if args.L not in spec.lengths and not args.allow_off_grid:
        parser.error(
            f"L={args.L} is not in the {args.model} grid; "
            "pass --allow-off-grid to run a supplementary context"
        )
    args.pos_remap = args.pos_remap or spec.pos_remap
    if args.pos_remap not in spec.remap_values:
        parser.error(
            f"pos_remap={args.pos_remap!r} invalid for {args.model}; "
            f"expected one of {spec.remap_values}"
        )
    if args.mode == "quality" and (args.dataset is None or args.window is None):
        parser.error("quality mode requires --dataset and --window")
    return args


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("EXP-1 requires CUDA")
    try:
        if args.mode == "timing":
            run_timing(args)
        else:
            run_quality(args)
        return 0
    except Exception as exc:
        traceback.print_exc()
        if args.mode == "timing":
            record_timing_failure(args, exc)
        else:
            record_quality_failure(args, requested_k_values(args), exc)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
