"""Run a resumable EXP-3 adaptive-policy block for one data/window/L cell."""

from __future__ import annotations

import argparse
import functools
import math
import time
import traceback
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from adapters import GraphPair
from common import (
    DATASETS,
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
    write_jsonl_create_once,
)
from exp1_w1 import latest_timing, make_initial_and_updates, save_predictions


THETAS = (0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
MAX_AGES = (None, 256, 1024, 4096)
BUDGETS = (1.0, 2.0, 5.0, 10.0)
CALIBRATIONS = ("none", "per_model")


def policy_payload(
    model: str,
    length: int,
    theta: float,
    max_age_pts: int | None,
    budget_pct: float,
    calib: str,
) -> dict[str, Any]:
    spec = MODELS[model]
    return {
        "exp": "EXP3",
        "model": model,
        "L": length,
        "theta": theta,
        "min_age_pts": spec.s,
        "max_age_pts": max_age_pts,
        "budget_pct": budget_pct,
        "signal": "norm_drift",
        "calib": calib,
        "pos_remap": spec.pos_remap,
    }


def policies(model: str, length: int) -> Iterable[tuple[str, dict[str, Any]]]:
    for theta in THETAS:
        for max_age in MAX_AGES:
            for budget in BUDGETS:
                for calib in CALIBRATIONS:
                    payload = policy_payload(model, length, theta, max_age, budget, calib)
                    yield stable_policy_id(payload), payload


def trace_relative_path(model: str, dataset: str, window: int, length: int, policy_id: str) -> Path:
    return Path("EXP3_adaptive") / model / "trace" / (
        f"{dataset}_w{window}_L{length}_{policy_id}.jsonl"
    )


def preds_relative_path(model: str, dataset: str, window: int, length: int, policy_id: str) -> Path:
    return Path("EXP3_adaptive") / model / "preds" / (
        f"{dataset}_w{window}_L{length}_{policy_id}.npz"
    )


def latest_policy_rows(path: Path) -> dict[tuple[Any, ...], dict[str, Any]]:
    result = {}
    for row in read_jsonl(path):
        result[(row.get("dataset"), row.get("window"), row.get("L"), row.get("policy_id"))] = row
    return result


@functools.lru_cache(maxsize=None)
def calibrated_max_age(model: str, length: int, budget: float) -> int:
    """Offline per-model cap from EXP-1 worst-over-dataset/window MAE delta."""
    latest: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in read_jsonl(RESULTS / "EXP1_sweep" / model / "records.jsonl"):
        key = (
            row.get("dataset"), row.get("window"), row.get("L"),
            row.get("K"), row.get("pos_remap"),
        )
        latest[key] = row
    worst: dict[int, float] = {}
    for row in latest.values():
        if (
            row.get("status") == "ok"
            and row.get("L") == length
            and row.get("pos_remap") == MODELS[model].pos_remap
            and isinstance(row.get("K"), int)
            and row.get("K", 0) > 0
            and row.get("mae_delta_pct") is not None
        ):
            k = int(row["K"])
            worst[k] = max(worst.get(k, -math.inf), float(row["mae_delta_pct"]))
    safe = [k for k, value in worst.items() if value <= budget]
    return max(safe, default=1) * MODELS[model].s


def realized_cap(payload: dict[str, Any]) -> int | None:
    configured = payload["max_age_pts"]
    if payload["calib"] == "none":
        return configured
    calibrated = calibrated_max_age(
        payload["model"], payload["L"], payload["budget_pct"]
    )
    return calibrated if configured is None else min(configured, calibrated)


def drift(window: np.ndarray, reference_mean: float, reference_std: float) -> float:
    current_mean = float(window.mean())
    current_std = float(window.std()) + 1e-6
    return max(
        abs(current_mean - reference_mean) / max(reference_std, 1e-6),
        abs(current_std - reference_std) / max(reference_std, 1e-6),
    )


def adaptive_record(
    model: str,
    dataset: str,
    window: int,
    start: int,
    length: int,
    policy_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    spec = MODELS[model]
    row = base_record("B", "EXP3", model)
    row.update(
        {
            "exp": "EXP3",
            "dataset": dataset,
            "window": window,
            "window_start": start,
            "L": length,
            "s": spec.s,
            "H": spec.horizon,
            "U": spec.updates,
            "policy": "adaptive",
            "K": None,
            "tau_pts": None,
            "pos_remap": spec.pos_remap,
            "norm_mode": "frozen_at_refresh",
            "adaptive_params": {
                key: payload[key]
                for key in (
                    "theta", "min_age_pts", "max_age_pts", "budget_pct", "signal", "calib"
                )
            },
            "effective_max_age_pts": realized_cap(payload),
            "policy_id": policy_id,
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
            "preds_file": str(preds_relative_path(model, dataset, window, length, policy_id)),
            "trace_file": str(trace_relative_path(model, dataset, window, length, policy_id)),
            "timing_ref": f"EXP3-{model}-L{length}-{policy_id}",
        }
    )
    return row


def decisions_and_predictions(
    pair: GraphPair,
    initial: torch.Tensor,
    updates: list[torch.Tensor],
    payload: dict[str, Any],
) -> tuple[np.ndarray, list[dict[str, Any]], int, int]:
    spec = pair.spec
    initial_np = initial.detach().float().cpu().numpy().reshape(-1)
    window = initial_np.copy()
    pair.reset(initial)
    ref_mean = float(window.mean())
    ref_std = float(window.std()) + 1e-6
    since_pts = 0
    cap = realized_cap(payload)
    predictions: list[np.ndarray] = []
    trace: list[dict[str, Any]] = []
    n_full = 0
    for step, update in enumerate(updates):
        update_np = update.detach().float().cpu().numpy().reshape(-1)
        window = np.concatenate((window[spec.s :], update_np))
        since_pts += spec.s
        value = drift(window, ref_mean, ref_std)
        max_hit = cap is not None and since_pts >= cap
        signal_hit = since_pts >= payload["min_age_pts"] and value >= payload["theta"]
        use_full = bool(max_hit or signal_hit)
        prediction = pair.full_step(window) if use_full else pair.rolling_step(update)
        torch.cuda.synchronize()
        predictions.append(pair.prediction_numpy(prediction)[0].copy())
        reason = "max_age" if max_hit else ("signal" if signal_hit else None)
        trace.append(
            {
                "step": step,
                "cache_age_pts": since_pts,
                "signal_value": value,
                "theta": payload["theta"],
                "refreshed": use_full,
                "refresh_reason": reason,
                "t_step_ms": None,
            }
        )
        n_full += int(use_full)
        if use_full:
            ref_mean = float(window.mean())
            ref_std = float(window.std()) + 1e-6
            since_pts = 0
    return np.stack(predictions).astype(np.float32), trace, n_full, len(updates) - n_full


def run_traced(args: argparse.Namespace, pair: GraphPair, initial: torch.Tensor, updates: list[torch.Tensor], targets: np.ndarray, t_index: np.ndarray, ref_yhat: np.ndarray, start: int) -> None:
    records_path = RESULTS / "EXP3_adaptive" / args.model / "records.jsonl"
    done = latest_policy_rows(records_path)
    exp1_timing = latest_timing(args.model, args.L, MODELS[args.model].pos_remap)[1]
    base_update_ms = float(exp1_timing["t_update_measured_ms"]["mean"])
    full_ms = float(exp1_timing["t_full_ms"]["median"])
    roll_ms = float(exp1_timing["t_roll_ms"]["median"])
    for policy_id, payload in policies(args.model, args.L):
        key = (args.dataset, args.window, args.L, policy_id)
        if done.get(key, {}).get("status") in {"ok", "unsupported", "oom"}:
            continue
        predictions, trace, n_full, n_roll = decisions_and_predictions(
            pair, initial, updates, payload
        )
        row = adaptive_record(
            args.model, args.dataset, args.window, start, args.L, policy_id, payload
        )
        row.update(quality_metrics(predictions, targets, ref_yhat))
        computed_ms = (n_full * full_ms + n_roll * roll_ms) / MODELS[args.model].updates
        row.update(
            {
                "n_full": n_full,
                "n_roll": n_roll,
                "speedup": base_update_ms / max(computed_ms, 1e-12),
            }
        )
        output = RESULTS / row["preds_file"]
        save_predictions(output, predictions, targets, t_index, row)
        trace_base = base_record("E", "EXP3", args.model)
        trace_rows = []
        for item in trace:
            trace_row = dict(trace_base)
            trace_row.update(
                {
                    "dataset": args.dataset,
                    "window": args.window,
                    "L": args.L,
                    "policy_id": policy_id,
                    "gap_step_pct": float(
                        np.abs(predictions[item["step"]] - ref_yhat[item["step"]]).mean()
                        / max(np.abs(ref_yhat[item["step"]] - targets[item["step"]]).mean(), 1e-12)
                        * 100.0
                    ),
                    "mae_step": float(
                        np.abs(predictions[item["step"]] - targets[item["step"]]).mean()
                    ),
                    **item,
                }
            )
            trace_rows.append(trace_row)
        write_jsonl_create_once(RESULTS / row["trace_file"], trace_rows)
        append_jsonl(records_path, row)
        print(
            f"EXP3 traced {args.model} {args.dataset} w={args.window} L={args.L} "
            f"policy={policy_id} full={n_full} MAE={row['mae_native']:.6f}",
            flush=True,
        )


def load_decisions(path: Path, updates: int) -> list[bool]:
    latest = {}
    for row in read_jsonl(path):
        latest[int(row["step"])] = bool(row["refreshed"])
    if set(latest) != set(range(updates)):
        raise RuntimeError(f"trace is incomplete: {path} has {len(latest)}/{updates} steps")
    return [latest[index] for index in range(updates)]


def timed_replay(
    pair: GraphPair,
    initial: torch.Tensor,
    updates: list[torch.Tensor],
    decisions: list[bool],
    include_signal: bool,
) -> list[float]:
    spec = pair.spec
    pair.reset(initial)
    window = initial.detach().float().cpu().numpy().reshape(-1).copy()
    ref_mean = float(window.mean())
    ref_std = float(window.std()) + 1e-6
    values = []
    for update, use_full in zip(updates, decisions):
        update_np = update.detach().float().cpu().numpy().reshape(-1)
        window = np.concatenate((window[spec.s :], update_np))
        begin = time.perf_counter()
        if include_signal:
            drift(window, ref_mean, ref_std)
        if use_full:
            pair.full_step(window)
        else:
            pair.rolling_step(update)
        torch.cuda.synchronize()
        values.append((time.perf_counter() - begin) * 1000.0)
        if use_full:
            ref_mean = float(window.mean())
            ref_std = float(window.std()) + 1e-6
    return values


def run_timed(args: argparse.Namespace, pair: GraphPair, initial: torch.Tensor, updates: list[torch.Tensor]) -> None:
    timing_path = RESULTS / "EXP3_adaptive" / args.model / "timing.jsonl"
    done = latest_policy_rows(timing_path)
    exp1_timing = latest_timing(args.model, args.L, MODELS[args.model].pos_remap)[1]
    baseline_ms = float(exp1_timing["t_update_measured_ms"]["mean"])
    full_ms = float(exp1_timing["t_full_ms"]["median"])
    roll_ms = float(exp1_timing["t_roll_ms"]["median"])
    for policy_id, payload in policies(args.model, args.L):
        key = (args.dataset, args.window, args.L, policy_id)
        if done.get(key, {}).get("status") in {"ok", "unsupported", "oom"}:
            continue
        trace_path = RESULTS / trace_relative_path(
            args.model, args.dataset, args.window, args.L, policy_id
        )
        decisions = load_decisions(trace_path, MODELS[args.model].updates)
        measured_values = timed_replay(pair, initial, updates, decisions, True)
        replay_values = timed_replay(pair, initial, updates, decisions, False)
        measured = metric_summary(measured_values)
        no_signal = metric_summary(replay_values)
        n_full = sum(decisions)
        n_roll = len(decisions) - n_full
        computed = (n_full * full_ms + n_roll * roll_ms) / len(decisions)
        inconsistency = abs(computed - measured["mean"]) / max(measured["mean"], 1e-12)
        row = base_record("C", "EXP3", args.model)
        row.update(
            {
                "dataset": args.dataset,
                "window": args.window,
                "L": args.L,
                "s": MODELS[args.model].s,
                "U": MODELS[args.model].updates,
                "policy": "adaptive",
                "K": None,
                "policy_id": policy_id,
                "adaptive_params": {
                    key: payload[key]
                    for key in (
                        "theta", "min_age_pts", "max_age_pts", "budget_pct", "signal", "calib"
                    )
                },
                "pos_remap": MODELS[args.model].pos_remap,
                "exec": "graph",
                "batch": 1,
                "timing_id": f"EXP3-{args.model}-L{args.L}-{policy_id}",
                "t_full_ms": exp1_timing["t_full_ms"],
                "t_roll_ms": exp1_timing["t_roll_ms"],
                "t_update_computed_ms": computed,
                "t_update_measured_ms": {
                    "mean": measured["mean"], "p95": measured["p95"],
                    "p99": measured["p99"], "n": measured["n"],
                },
                "speedup": baseline_ms / max(measured["mean"], 1e-12),
                "us_per_point": measured["mean"] / MODELS[args.model].s * 1000.0,
                "timing_flag": "inconsistent" if inconsistency > 0.10 else None,
                "graph_capture_ok": True,
                "n_kernels_full": exp1_timing.get("n_kernels_full"),
                "n_kernels_roll": exp1_timing.get("n_kernels_roll"),
                "kv_cache_mb": pair.cache_mb(),
                "peak_mem_mb": pair.peak_mem_mb,
                "trigger_overhead_ms": measured["mean"] - no_signal["mean"],
                "signal_none_replay_ms": no_signal,
                "n_full": n_full,
                "n_roll": n_roll,
            }
        )
        append_jsonl(timing_path, row)
        print(
            f"EXP3 timed {args.model} {args.dataset} w={args.window} L={args.L} "
            f"policy={policy_id} {measured['mean']:.4f}ms",
            flush=True,
        )


def write_unsupported(args: argparse.Namespace, start: int, reason: str) -> None:
    records_path = RESULTS / "EXP3_adaptive" / args.model / "records.jsonl"
    timing_path = RESULTS / "EXP3_adaptive" / args.model / "timing.jsonl"
    for policy_id, payload in policies(args.model, args.L):
        if args.mode in {"traced", "all"}:
            row = adaptive_record(
                args.model, args.dataset, args.window, start, args.L, policy_id, payload
            )
            row.update({"status": "unsupported", "reason": reason})
            append_jsonl(records_path, row)
        if args.mode in {"timed", "all"}:
            row = base_record("C", "EXP3", args.model)
            row.update(
                {
                    "status": "unsupported", "reason": reason,
                    "dataset": args.dataset, "window": args.window, "L": args.L,
                    "policy_id": policy_id, "policy": "adaptive", "K": None,
                }
            )
            append_jsonl(timing_path, row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--window", type=int, required=True)
    parser.add_argument("--L", type=int, required=True)
    parser.add_argument("--mode", choices=("traced", "timed", "all"), default="all")
    args = parser.parse_args()
    spec = MODELS[args.model]
    if args.L not in spec.lengths:
        parser.error("L is outside the model grid")
    try:
        manifest = load_manifest()
        start = int(manifest["windows"][args.dataset][args.window])
        series = load_series(args.dataset)
        end = start + spec.updates * spec.s + spec.horizon
        if start < args.L or end > len(series):
            reason = (
                f"dataset bounds: start={start}, L={args.L}, end={end}, "
                f"series_length={len(series)}"
            )
            write_unsupported(args, start, reason)
            return 0
        initial, updates, targets, t_index = make_initial_and_updates(
            series, start, spec, args.L
        )
        from exp1_w1 import prediction_relative_path

        baseline = RESULTS / prediction_relative_path(
            args.model, args.dataset, args.window, args.L, 1, spec.pos_remap
        )
        if not baseline.exists():
            raise FileNotFoundError(f"missing EXP1 K=1 predictions: {baseline}")
        with np.load(baseline, allow_pickle=False) as archive:
            ref_yhat = archive["yhat"].copy()
        pair = GraphPair(args.model, args.L, initial, pos_remap=spec.pos_remap)
        if args.mode in {"traced", "all"}:
            run_traced(args, pair, initial, updates, targets, t_index, ref_yhat, start)
        if args.mode in {"timed", "all"}:
            run_timed(args, pair, initial, updates)
        return 0
    except Exception as exc:
        traceback.print_exc()
        status = classify_failure(exc)
        reason = f"{type(exc).__name__}: {str(exc)[:1200]}"
        # One failure record per policy preserves the complete-grid contract.
        try:
            start
        except UnboundLocalError:
            start = -1
        for policy_id, payload in policies(args.model, args.L):
            if args.mode in {"traced", "all"}:
                row = adaptive_record(
                    args.model, args.dataset, args.window, start, args.L, policy_id, payload
                )
                row.update({"status": status, "reason": reason})
                append_jsonl(RESULTS / "EXP3_adaptive" / args.model / "records.jsonl", row)
            if args.mode in {"timed", "all"}:
                row = base_record("C", "EXP3", args.model)
                row.update(
                    {
                        "status": status, "reason": reason,
                        "dataset": args.dataset, "window": args.window, "L": args.L,
                        "policy_id": policy_id, "policy": "adaptive", "K": None,
                    }
                )
                append_jsonl(RESULTS / "EXP3_adaptive" / args.model / "timing.jsonl", row)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
