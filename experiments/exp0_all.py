"""Run resumable EXP-0 correctness gates for W2/W3 models."""

from __future__ import annotations

import argparse
import gc
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from adapters import GraphPair
from common import MODELS, RESULTS, append_jsonl, base_record, classify_failure, read_jsonl


def make_series(length: int, seed: int = 7) -> np.ndarray:
    rng = np.random.RandomState(seed)
    x = np.arange(length, dtype=np.float32)
    return (
        np.sin(2 * np.pi * x / 96)
        + 0.5 * np.sin(2 * np.pi * x / 336)
        + 0.2 * rng.randn(length).astype(np.float32)
    ).astype(np.float32)


def stats(got: np.ndarray, expected: np.ndarray) -> tuple[float, float, float]:
    got = np.asarray(got, dtype=np.float32)
    expected = np.asarray(expected, dtype=np.float32)
    max_abs = float(np.max(np.abs(got - expected)))
    scale = float(np.max(np.abs(expected)))
    return max_abs, max_abs / max(scale, 1e-12), scale


def gate_row(model: str, gate: str, length: int) -> dict[str, Any]:
    row = base_record("A", "EXP0", model)
    row.update(
        {
            "gate": gate,
            "L": length,
            "dtype": MODELS[model].dtype,
            "max_abs_err": None,
            "rel_err": None,
            "scale": None,
            "threshold": None,
            "passed": None,
            "cache_age": None,
            "gap_rel": None,
        }
    )
    return row


def completed_length(path: Path, length: int) -> bool:
    latest: dict[tuple[Any, Any], dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("L") == length:
            latest[(row.get("gate"), row.get("cache_age"))] = row
    return all(
        latest.get((gate, None), {}).get("status") == "ok"
        and latest[(gate, None)].get("passed") is True
        for gate in ("T1", "T2", "T3", "T4")
    ) and all(
        latest.get(("T5", age), {}).get("status") == "ok"
        for age in range(1, 9)
    )


def append(path: Path, row: dict[str, Any]) -> bool:
    append_jsonl(path, row)
    print(
        f"{row['model']} L={row['L']} {row['gate']}"
        f" age={row.get('cache_age')}: status={row['status']} "
        f"passed={row.get('passed')} rel={row.get('rel_err')}",
        flush=True,
    )
    return row.get("status") == "ok" and row.get("passed") is not False


@torch.no_grad()
def run_length(model: str, length: int, pos_remap: str, path: Path) -> bool:
    spec = MODELS[model]
    threshold = 2e-2 if spec.dtype == "bfloat16" else 1e-5
    series = make_series(length + 10 * spec.s)
    initial = series[:length]
    pair = GraphPair(model, length, initial, pos_remap=pos_remap)
    ok = True

    graph_full = pair.prediction_numpy(pair.full_step(initial))
    torch.cuda.synchronize()
    official = pair.prediction_numpy(pair.official_full(initial))
    max_abs, rel, scale = stats(graph_full, official)
    row = gate_row(model, "T1", length)
    row.update(
        {
            "max_abs_err": max_abs,
            "rel_err": rel,
            "scale": scale,
            "threshold": threshold,
            "passed": rel <= threshold,
            "comparison": "custom full graph path vs upstream full forward",
            "pos_remap": pos_remap,
        }
    )
    ok &= append(path, row)

    max_abs, rel = pair.append_only_error(initial)
    row = gate_row(model, "T2", length)
    row.update(
        {
            "max_abs_err": max_abs,
            "rel_err": rel,
            "scale": max_abs / max(rel, 1e-12) if rel else 0.0,
            "threshold": threshold,
            "passed": rel <= threshold,
            "comparison": (
                "append-only Q/K/V growth vs one full prefill; full outputs are "
                "covered by T1 and rolling outputs by T3"
                if model == "timerxl"
                else "cached prefix plus final token/patch vs one full prefill"
            ),
            "pos_remap": pos_remap,
        }
    )
    ok &= append(path, row)

    pair.reset(initial)
    eager = pair.new_eager_engine()
    pair.eager_full(eager, initial)
    graph_error = graph_scale = 0.0
    for age in range(1, 4):
        lo = length + (age - 1) * spec.s
        update = series[lo : lo + spec.s]
        got = pair.prediction_numpy(pair.rolling_step(update))
        expected = pair.prediction_numpy(pair.eager_roll(eager, update))
        torch.cuda.synchronize()
        current, _, current_scale = stats(got, expected)
        graph_error = max(graph_error, current)
        graph_scale = max(graph_scale, current_scale)
    row = gate_row(model, "T3", length)
    row.update(
        {
            "max_abs_err": graph_error,
            "rel_err": graph_error / max(graph_scale, 1e-12),
            "scale": graph_scale,
            "threshold": 1e-6,
            "passed": graph_error <= 1e-6,
            "comparison": "rolling CUDA Graph replay vs corresponding eager path",
            "pos_remap": pos_remap,
        }
    )
    ok &= append(path, row)

    algebra_error = pair.position_algebra_error()
    row = gate_row(model, "T4", length)
    row.update(
        {
            "max_abs_err": algebra_error,
            "rel_err": algebra_error,
            "scale": 1.0,
            "threshold": 1e-12,
            "passed": True if spec.pos_remap == "n/a" else algebra_error <= 1e-12,
            "comparison": "R(-1)K_p vs directly encoded K_(p-1) in FP64",
            "pos_remap": pos_remap,
            "note": "not applicable" if spec.pos_remap == "n/a" else None,
        }
    )
    ok &= append(path, row)

    pair.reset(initial)
    window = initial.copy()
    for age in range(1, 9):
        lo = length + (age - 1) * spec.s
        update = series[lo : lo + spec.s]
        window = np.concatenate((window[spec.s :], update))
        rolling = pair.prediction_numpy(pair.rolling_step(update))
        full_engine = pair.new_eager_engine()
        full = pair.prediction_numpy(pair.eager_full(full_engine, window))
        torch.cuda.synchronize()
        max_abs, rel, scale = stats(rolling, full)
        row = gate_row(model, "T5", length)
        row.update(
            {
                "max_abs_err": max_abs,
                "rel_err": rel,
                "scale": scale,
                "threshold": None,
                "passed": None,
                "cache_age": age,
                "gap_rel": float(np.mean(np.abs(rolling - full)))
                / max(float(np.mean(np.abs(full))), 1e-12),
                "pos_remap": pos_remap,
            }
        )
        ok &= append(path, row)
    return ok


def record_failure(model: str, length: int, path: Path, exc: BaseException) -> None:
    status = classify_failure(exc)
    reason = f"{type(exc).__name__}: {str(exc)[:1200]}"
    for gate in ("T1", "T2", "T3", "T4"):
        row = gate_row(model, gate, length)
        row.update({"status": status, "reason": reason})
        append_jsonl(path, row)
    for age in range(1, 9):
        row = gate_row(model, "T5", length)
        row.update({"status": status, "reason": reason, "cache_age": age})
        append_jsonl(path, row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    args = parser.parse_args()
    if args.model in {"timesfm", "timemoe"}:
        parser.error("W1 EXP-0 is handled by exp0_w1.py")
    if not torch.cuda.is_available():
        raise RuntimeError("EXP-0 requires CUDA")
    spec = MODELS[args.model]
    path = RESULTS / "EXP0_correctness" / args.model / "records.jsonl"
    failed = False
    for length in spec.lengths:
        if completed_length(path, length):
            print(f"EXP0 already complete: {args.model} L={length}", flush=True)
            continue
        try:
            failed |= not run_length(args.model, length, spec.pos_remap, path)
        except Exception as exc:
            traceback.print_exc()
            record_failure(args.model, length, path, exc)
            failed = True
        gc.collect()
        torch.cuda.empty_cache()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
