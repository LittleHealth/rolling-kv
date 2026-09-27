"""Disconnect-safe queue for the TimesFM-3.0 campaign under the locked v2 protocol.

Every TimesFM-2.5 result in the paper came from the v2 campaign (U=64, H=64,
s=32, FP32, batch 1, both paths as CUDA Graphs, the v2 manifest windows).
This queue produces the TimesFM-3.0 counterpart of each of them with the same
scripts and output schemas:

  P-E0       exp0_gates_native.py gates over L in {512,...,16384}; a failure aborts.
  P-timing   exp1_fixed_policy_sweep.py timing: 17 K at L in {512,2048,8192,16384} and
             K in {1,4,16,64} at the partial contexts L in {1024,4096}.
             Latency: requires an A100-SXM4 (the paper's hardware) and an
             otherwise idle GPU.
  P-quality  exp1_fixed_policy_sweep.py quality: 7 datasets x 5 windows x main L (17 K,
             infeasible cells recorded as status=unsupported) plus the
             partial contexts on ETTm1/ETTm2/Electricity/Weather (4 K).
  P-S2       exp2_time_attribution.py kernel profile: 6 L at batch 1, L=8192 at batch 8.
  P-validate validate_campaign.py.

``plan()`` is the single source of truth for the grid; validate_campaign.py
imports it.  The results root defaults to ``$ROLLKV_ROOT/results_tfm3`` and
deliberately ignores an inherited ``ROLLKV_RESULTS`` (env.sh points that at
the TimesFM-2.5 tree); pass ``--results`` to override.

Concurrency: ``--phases`` selects phases and ``--shard i --nshards n`` takes
every n-th task of each selected phase, each (phases, shard) with its own
lock and durable cursor, so e.g. timing and the kernel profile can run on one
SXM4 GPU while quality shards run on others.  Cross-launcher dependencies are
enforced by waiting on the artifacts themselves: nothing but P-E0 starts
before the EXP-0 gates have passed, and a quality task starts only once the
timing rows for its context exist (its speedup field is backfilled from
them).  A dropped session resumes with ``--resume`` and loses at most one
task; every underlying script is idempotent.

Examples (after ``source env.sh``)::

    python timesfm3/launch_campaign.py --gpu 1 --phases P-E0,P-timing,P-S2
    python timesfm3/launch_campaign.py --gpu 2 --phases P-quality --shard 0 --nshards 3
    python timesfm3/launch_campaign.py --gpu 5 --phases P-quality --shard 1 --nshards 3
    python timesfm3/launch_campaign.py --gpu 1 --phases P-validate
    python timesfm3/launch_campaign.py --print-plan
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # experiments/ root

import argparse
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

HERE = Path(__file__).resolve().parent


def default_results_root() -> Path:
    root = os.environ.get("ROLLKV_ROOT")
    base = Path(root) if root else HERE.parents[1]
    return base / "results_tfm3"


def bootstrap_results(argv: list[str] | None = None) -> Path:
    """Bind ROLLKV_RESULTS before ``common`` is imported (it reads it once)."""
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--results")
    known, _ = pre.parse_known_args(argv)
    results = Path(known.results).resolve() if known.results else default_results_root()
    os.environ["ROLLKV_RESULTS"] = str(results)
    return results


if __name__ == "__main__":
    bootstrap_results()

from shared.common import (  # noqa: E402
    DATASETS,
    MODELS,
    TFM3_ALL_LENGTHS,
    TFM3_EXP2_BATCH8_LENGTH,
    TFM3_MODEL,
    TFM3_PARTIAL_DATASETS,
    TFM3_PARTIAL_K_VALUES,
    TFM3_PARTIAL_LENGTHS,
    TFM3_V2_WINDOWS,
    create_manifest_tfm3,
    gpu_processes,
    gpu_snapshot,
    read_jsonl,
    tfm3_cell_feasible,
    utc_now,
    write_json_atomic,
)


MODEL = TFM3_MODEL
PHASES = ("P-E0", "P-timing", "P-quality", "P-S2", "P-validate")
# Latency and kernel-profile phases: paper hardware and an idle GPU.
LATENCY_PHASES = ("P-timing", "P-S2")
REQUIRED_GPU_TOKEN = "SXM4"
WINDOW_COUNT = 5
TERMINAL_TIMING = {"ok", "oom", "unsupported", "capture_failed", "failed"}


# ---------------------------------------------------------------------------
# The locked grid (pure; validate_campaign.py imports these).


def plan() -> dict[str, Any]:
    """Pure description of the locked TimesFM-3.0 grid (no I/O)."""
    spec = MODELS[MODEL]
    main_lengths = list(spec.lengths)
    main_k = list(spec.k_values)
    partial_k = list(TFM3_PARTIAL_K_VALUES)
    timing = [{"L": length, "k_values": main_k, "off_grid": False} for length in main_lengths]
    timing += [
        {"L": length, "k_values": partial_k, "off_grid": True}
        for length in TFM3_PARTIAL_LENGTHS
    ]
    timing.sort(key=lambda item: item["L"])
    quality = [
        {"dataset": dataset, "L": length, "k_values": main_k, "off_grid": False}
        for dataset in DATASETS
        for length in main_lengths
    ]
    quality += [
        {"dataset": dataset, "L": length, "k_values": partial_k, "off_grid": True}
        for dataset in TFM3_PARTIAL_DATASETS
        for length in TFM3_PARTIAL_LENGTHS
    ]
    exp2 = [
        {"L": length, "batch": 1, "off_grid": length not in spec.lengths}
        for length in TFM3_ALL_LENGTHS
    ]
    exp2.append(
        {
            "L": TFM3_EXP2_BATCH8_LENGTH,
            "batch": 8,
            "off_grid": TFM3_EXP2_BATCH8_LENGTH not in spec.lengths,
        }
    )
    return {
        "model": MODEL,
        "display_name": spec.display_name,
        "windows": WINDOW_COUNT,
        "manifest_windows": {name: list(values) for name, values in TFM3_V2_WINDOWS.items()},
        "s": spec.s,
        "H": spec.horizon,
        "U": spec.updates,
        "exp0_lengths": list(TFM3_ALL_LENGTHS),
        "timing": timing,
        "quality": quality,
        "exp2": exp2,
        # (dataset, window, L): series of ~17k points have no L=16384 history.
        "expected_unsupported": [
            [dataset, window, 16384]
            for dataset in ("ETTh1", "ETTh2", "Traffic")
            for window in range(WINDOW_COUNT)
        ],
        "phases": list(PHASES),
    }


def quality_cells(grid: dict[str, Any]) -> set[tuple[str, int, int, int]]:
    """Every planned (dataset, window, L, K) quality cell."""
    return {
        (item["dataset"], window, int(item["L"]), int(k))
        for item in grid["quality"]
        for window in range(grid["windows"])
        for k in item["k_values"]
    }


def timing_pairs(grid: dict[str, Any]) -> set[tuple[int, int]]:
    return {(int(item["L"]), int(k)) for item in grid["timing"] for k in item["k_values"]}


def kernel_cells(grid: dict[str, Any]) -> set[tuple[int, int, str]]:
    return {
        (int(item["L"]), int(item["batch"]), path)
        for item in grid["exp2"]
        for path in ("full", "rolling")
    }


def expected_unsupported(grid: dict[str, Any]) -> set[tuple[str, int, int]]:
    return {(str(d), int(w), int(length)) for d, w, length in grid["expected_unsupported"]}


# ---------------------------------------------------------------------------
# Tasks.


@dataclass(frozen=True)
class Task:
    phase: str
    command: tuple[str, ...]
    abort_queue_on_failure: bool = False
    is_validation: bool = False
    # Quality tasks: the context whose timing rows must exist first, the K
    # values it needs, and the cell (dataset, window) for feasibility.
    needs_timing: tuple[int, tuple[int, ...]] | None = None
    cell: tuple[str, int, int] | None = field(default=None)


def shared_command(script: str, *args: Any) -> tuple[str, ...]:
    return (sys.executable, str(HERE.parent / "shared" / script), *(str(value) for value in args))


def command(script: str, *args: Any) -> tuple[str, ...]:
    return (sys.executable, str(HERE / script), *(str(value) for value in args))


def k_arg(values: list[int]) -> str:
    return ",".join(str(k) for k in values)


def build_tasks(results: Path) -> list[Task]:
    grid = plan()
    spec = MODELS[MODEL]
    tasks: list[Task] = [
        Task("P-E0", shared_command("exp0_gates_native.py", "--model", MODEL), abort_queue_on_failure=True)
    ]
    for item in grid["timing"]:
        extra: tuple[str, ...] = ()
        if item["off_grid"]:
            extra = ("--k-values", k_arg(item["k_values"]), "--allow-off-grid")
        tasks.append(
            Task(
                "P-timing",
                shared_command("exp1_fixed_policy_sweep.py", "--mode", "timing", "--model", MODEL, "--L", item["L"], *extra),
            )
        )
    quality: list[tuple[int, str, int, dict[str, Any]]] = [
        (int(item["L"]), item["dataset"], window, item)
        for item in grid["quality"]
        for window in range(grid["windows"])
    ]
    order = {name: index for index, name in enumerate(DATASETS)}
    quality.sort(key=lambda entry: (entry[0], order[entry[1]], entry[2]))
    for length, dataset, window, item in quality:
        extra = ()
        if item["off_grid"]:
            extra = ("--k-values", k_arg(item["k_values"]), "--allow-off-grid")
        elif list(item["k_values"]) != list(spec.k_values):
            extra = ("--k-values", k_arg(item["k_values"]))
        tasks.append(
            Task(
                "P-quality",
                shared_command(
                    "exp1_fixed_policy_sweep.py", "--mode", "quality", "--model", MODEL,
                    "--dataset", dataset, "--window", window, "--L", length, *extra,
                ),
                needs_timing=(length, tuple(int(k) for k in item["k_values"])),
                cell=(dataset, window, length),
            )
        )
    for item in grid["exp2"]:
        extra = ("--allow-off-grid",) if item["off_grid"] else ()
        tasks.append(
            Task(
                "P-S2",
                shared_command(
                    "exp2_time_attribution.py", "--model", MODEL, "--L", item["L"],
                    "--path", "both", "--batch", item["batch"], "--dump-kernel-names", *extra,
                ),
            )
        )
    tasks.append(
        Task(
            "P-validate",
            command("validate_campaign.py", "--results", results),
            is_validation=True,
        )
    )
    return tasks


def select_tasks(
    tasks: list[Task], phases: tuple[str, ...], shard: int, nshards: int
) -> list[Task]:
    """Tasks of the selected phases; shard i keeps every n-th task per phase."""
    selected = []
    position: dict[str, int] = {}
    for task in tasks:
        if task.phase not in phases:
            continue
        index = position.get(task.phase, 0)
        position[task.phase] = index + 1
        if index % nshards == shard:
            selected.append(task)
    return selected


def task_counts(tasks: list[Task]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for task in tasks:
        counts[task.phase] = counts.get(task.phase, 0) + 1
    counts["total"] = len(tasks)
    return counts


def fingerprint(tasks: list[Task]) -> str:
    payload = [[task.phase, *task.command[1:]] for task in tasks]
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Manifest and cross-launcher dependencies.


@contextmanager
def manifest_lock(results: Path) -> Iterator[None]:
    results.mkdir(parents=True, exist_ok=True)
    with (results / "manifest.lock").open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def read_manifest(results: Path) -> dict[str, Any]:
    return json.loads((results / "manifest.json").read_text(encoding="utf-8"))


def ensure_manifest(results: Path, gpu: int) -> dict[str, Any]:
    with manifest_lock(results):
        path = results / "manifest.json"
        if not path.exists():
            print(f"manifest.json missing under {results}; creating it", flush=True)
            manifest = create_manifest_tfm3(gpu, WINDOW_COUNT)
        else:
            manifest = read_manifest(results)
        windows = {name: list(values) for name, values in manifest.get("windows", {}).items()}
        want = {name: list(values) for name, values in TFM3_V2_WINDOWS.items()}
        if windows != want:
            raise ValueError(f"{path} windows differ from the locked v2 windows")
        if not manifest.get("windows_redraw_matches_v2", False):
            print(
                "WARNING: choose_windows() does not redraw the v2 windows on this "
                f"server's data: {json.dumps(manifest.get('windows_redraw_mismatch'))}",
                flush=True,
            )
        return manifest


def update_phase(results: Path, name: str, state: str, detail: Any = None) -> None:
    with manifest_lock(results):
        manifest = read_manifest(results)
        manifest.setdefault("phases", {})[name] = {
            "state": state,
            "ts": utc_now(),
            "detail": detail,
        }
        write_json_atomic(results / "manifest.json", manifest)


def exp0_status(results: Path) -> str:
    """'passed' | 'failed' | 'pending' from the latest EXP-0 rows."""
    from shared.exp0_gates_native import completed_length, required_gates

    path = results / "EXP0_correctness" / MODEL / "records.jsonl"
    try:
        rows = read_jsonl(path)
        if all(completed_length(path, length, MODEL) for length in TFM3_ALL_LENGTHS):
            return "passed"
    except ValueError:
        # A concurrent appender's line may be caught half-written; poll again.
        return "pending"
    latest: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        latest[(row.get("L"), row.get("gate"), row.get("cache_age"))] = row
    gates = required_gates(MODEL)
    for (length, gate, age), row in latest.items():
        if gate in gates and age is None and (
            row.get("status") != "ok" or row.get("passed") is False
        ):
            return "failed"
    return "pending"


def wait_for_exp0(results: Path, timeout_s: float, poll_s: float = 30.0) -> None:
    begin = time.monotonic()
    announced = False
    while True:
        status = exp0_status(results)
        if status == "passed":
            if announced:
                print("EXP-0 gates passed; continuing", flush=True)
            return
        if status == "failed":
            raise RuntimeError(
                "EXP-0 gates failed for timesfm3; downstream measurements are "
                "not interpretable (fix and re-run P-E0)"
            )
        if time.monotonic() - begin > timeout_s:
            raise TimeoutError("timed out waiting for the EXP-0 gates to pass")
        if not announced:
            print("waiting for the EXP-0 gates (P-E0) to pass...", flush=True)
            announced = True
        time.sleep(poll_s)


def timing_ready(results: Path, length: int, ks: tuple[int, ...]) -> bool:
    path = results / "EXP1_sweep" / MODEL / "timing.jsonl"
    latest: dict[int, dict[str, Any]] = {}
    try:
        rows = read_jsonl(path)
    except ValueError:
        return False  # torn last line from a concurrent appender; poll again
    for row in rows:
        if (
            row.get("L") == length
            and row.get("exec") == "graph"
            and row.get("batch") == 1
            and row.get("pos_remap") == "n/a"
        ):
            latest[int(row["K"])] = row
    return all(k in latest and latest[k].get("status") in TERMINAL_TIMING for k in ks)


def wait_for_timing(
    results: Path, length: int, ks: tuple[int, ...], timeout_s: float, poll_s: float = 20.0
) -> None:
    begin = time.monotonic()
    announced = False
    while not timing_ready(results, length, ks):
        if time.monotonic() - begin > timeout_s:
            # Stop with the cursor on this task (--resume retries it) rather
            # than let exp1 record "missing timing rows" failures.
            raise TimeoutError(f"timed out waiting for P-timing rows at L={length}")
        if not announced:
            print(f"waiting for P-timing rows at L={length} K={list(ks)}...", flush=True)
            announced = True
        time.sleep(poll_s)


def wait_for_gpu(gpu: int, seconds: int = 60) -> None:
    announced = False
    while True:
        active = gpu_processes(gpu)
        if not active:
            if announced:
                print(f"GPU {gpu} became free", flush=True)
            return
        if not announced:
            print(f"GPU {gpu} busy; waiting for an idle GPU: {json.dumps(active)}", flush=True)
            announced = True
        time.sleep(seconds)


def run_task(
    task: Task, gpu: int, retries: int, results: Path, exclusive: bool
) -> tuple[bool, int, int]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # nvidia-smi numbering == CUDA numbering, independent of the driver's
    # default FASTEST_FIRST order on this mixed PCIe/SXM4 host.
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["PYTHONPATH"] = str(HERE.parent)
    env["ROLLKV_RESULTS"] = str(results)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    code = -1
    for attempt in range(1, retries + 2):
        if exclusive:
            wait_for_gpu(gpu)
        print(
            f"RUN phase={task.phase} attempt={attempt}: {' '.join(task.command)}",
            flush=True,
        )
        code = subprocess.run(task.command, env=env).returncode
        if code == 0:
            return True, attempt, code
        print(f"command failed exit={code}", flush=True)
        if attempt <= retries:
            time.sleep(min(60, attempt * 10))
    return False, retries + 1, code


# ---------------------------------------------------------------------------
# Main.


def parse_phases(text: str) -> tuple[str, ...]:
    names = tuple(item.strip() for item in text.split(",") if item.strip())
    unknown = [name for name in names if name not in PHASES]
    if unknown or not names:
        raise argparse.ArgumentTypeError(f"phases must be a subset of {PHASES}; got {text!r}")
    return tuple(name for name in PHASES if name in names)


def queue_tag(phases: tuple[str, ...], shard: int, nshards: int) -> str:
    if phases == PHASES and nshards == 1:
        return "all"
    short = "+".join(name.removeprefix("P-") for name in phases)
    return f"{short}.shard{shard}of{nshards}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gpu", type=int, help="nvidia-smi index of the GPU to use")
    parser.add_argument(
        "--results",
        help="results root (default: $ROLLKV_ROOT/results_tfm3; ROLLKV_RESULTS is ignored)",
    )
    parser.add_argument("--phases", type=parse_phases, default=PHASES)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--allow-non-sxm4",
        action="store_true",
        help="permit P-timing/P-S2 on a GPU that is not an A100-SXM4 (not paper hardware)",
    )
    parser.add_argument(
        "--wait-timeout-h",
        type=float,
        default=24.0,
        help="hours to wait for EXP-0 gates / timing rows produced by another launcher",
    )
    parser.add_argument("--print-plan", action="store_true", help="print task counts and exit")
    args = parser.parse_args()
    if not 0 <= args.shard < args.nshards:
        parser.error("--shard must satisfy 0 <= shard < nshards")

    results = Path(os.environ["ROLLKV_RESULTS"])
    tasks = select_tasks(build_tasks(results), args.phases, args.shard, args.nshards)
    if args.print_plan:
        full = build_tasks(results)
        grid = plan()
        print(json.dumps(
            {
                "results": str(results),
                "selected": task_counts(tasks),
                "all_tasks": task_counts(full),
                "quality_cells": len(quality_cells(grid)),
                "timing_pairs": len(timing_pairs(grid)),
                "kernel_records": len(kernel_cells(grid)),
                "expected_unsupported_cells": len(expected_unsupported(grid)),
            },
            indent=2,
        ))
        return 0
    if args.gpu is None:
        parser.error("--gpu is required")

    results.mkdir(parents=True, exist_ok=True)
    tag = queue_tag(args.phases, args.shard, args.nshards)
    lock_handle = (results / f"timesfm3_queue.{tag}.lock").open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError(f"another timesfm3/launch_campaign.py instance owns the {tag} queue lock")

    gpu_name = gpu_snapshot(args.gpu)["name"]
    if any(task.phase in LATENCY_PHASES for task in tasks):
        if REQUIRED_GPU_TOKEN not in gpu_name and not args.allow_non_sxm4:
            raise RuntimeError(
                f"GPU {args.gpu} is {gpu_name!r}; P-timing/P-S2 must run on an "
                f"A100-{REQUIRED_GPU_TOKEN} (paper hardware); pass --allow-non-sxm4 to override"
            )
    ensure_manifest(results, args.gpu)

    state_path = results / f"timesfm3_queue_state.{tag}.json"
    state: dict[str, Any] = {
        "schema": "timesfm3-v2-protocol-queue",
        "tag": tag,
        "phases": list(args.phases),
        "shard": args.shard,
        "nshards": args.nshards,
        "gpu": args.gpu,
        "gpu_name": gpu_name,
        "results": str(results),
        "task_count": len(tasks),
        "plan_fingerprint": fingerprint(tasks),
        "next_index": 0,
        "failed_commands": [],
        "validate_exit_code": None,
        "started_at": utc_now(),
        "updated_at": utc_now(),
    }
    if args.resume and state_path.exists():
        old = json.loads(state_path.read_text())
        if (
            old.get("task_count") == len(tasks)
            and old.get("plan_fingerprint") == state["plan_fingerprint"]
        ):
            for key in ("next_index", "failed_commands", "validate_exit_code", "started_at"):
                state[key] = old.get(key, state[key])
            state["resumed_at"] = utc_now()
        else:
            print("saved state describes a different task list; starting fresh", flush=True)
    write_json_atomic(state_path, state)
    print(
        f"timesfm3 queue {tag}: {len(tasks)} tasks {task_counts(tasks)} "
        f"from index {state['next_index']} on GPU {args.gpu} ({gpu_name}); results {results}",
        flush=True,
    )

    timeout_s = args.wait_timeout_h * 3600.0
    manifest = read_manifest(results)
    series_lengths = manifest.get("series_lengths", {})
    current_phase: str | None = None

    def finish_phase(phase: str) -> None:
        failures = [item for item in state["failed_commands"] if item["phase"] == phase]
        update_phase(
            results,
            f"{phase}[{tag}]",
            "complete_with_failures" if failures else "complete",
            {"failed_commands": len(failures), "gpu": args.gpu, "gpu_name": gpu_name},
        )

    for index in range(int(state.get("next_index", 0)), len(tasks)):
        task = tasks[index]
        if task.phase != current_phase:
            if current_phase is not None:
                finish_phase(current_phase)
            current_phase = task.phase
            update_phase(
                results, f"{current_phase}[{tag}]", "running",
                {"gpu": args.gpu, "gpu_name": gpu_name},
            )

        try:
            if task.phase not in ("P-E0", "P-validate"):
                wait_for_exp0(results, timeout_s)
            if task.needs_timing is not None and task.cell is not None:
                dataset, window, length = task.cell
                start = int(manifest["windows"][dataset][window])
                feasible = tfm3_cell_feasible(start, length, int(series_lengths.get(dataset, 0)))
                if feasible:
                    wait_for_timing(results, *task.needs_timing, timeout_s=timeout_s)
        except (RuntimeError, TimeoutError) as exc:
            print(f"ABORT {exc}", flush=True)
            update_phase(results, f"{task.phase}[{tag}]", "aborted", {"reason": str(exc)})
            state.update(
                {
                    "next_index": index,
                    "status": "aborted_dependency",
                    "finished_at": utc_now(),
                    "abort_reason": str(exc),
                }
            )
            write_json_atomic(state_path, state)
            return 1

        started = time.monotonic()
        success, attempts, last_code = run_task(
            task, args.gpu, args.retries, results, exclusive=task.phase in LATENCY_PHASES
        )
        elapsed = time.monotonic() - started
        print(
            f"DONE phase={task.phase} ok={success} attempts={attempts} "
            f"exit={last_code} elapsed={elapsed:.1f}s",
            flush=True,
        )
        if not success:
            state["failed_commands"].append(
                {
                    "index": index,
                    "phase": task.phase,
                    "command": list(task.command),
                    "attempts": attempts,
                    "exit_code": last_code,
                    "ts": utc_now(),
                }
            )
        if task.is_validation:
            state["validate_exit_code"] = last_code
        aborting = not success and task.abort_queue_on_failure
        state.update(
            {
                # An aborted gate keeps its own index so --resume re-runs it.
                "next_index": index if aborting else index + 1,
                "current_phase": task.phase,
                "last_command": list(task.command),
                "last_command_ok": success,
                "updated_at": utc_now(),
            }
        )
        write_json_atomic(state_path, state)
        if aborting:
            update_phase(results, f"{task.phase}[{tag}]", "aborted", {"exit_code": last_code})
            state["status"] = "aborted_correctness_gate"
            state["finished_at"] = utc_now()
            write_json_atomic(state_path, state)
            print(f"ABORT correctness gate failed (exit={last_code}); stopping the queue", flush=True)
            return 1

    if current_phase is not None:
        finish_phase(current_phase)

    if "P-validate" in args.phases and args.shard == 0:
        exit_code = state.get("validate_exit_code")
        if exit_code is None:
            validation_path = results / "validation_timesfm3.json"
            passed = False
            if validation_path.exists():
                passed = bool(json.loads(validation_path.read_text()).get("passed"))
            exit_code = 0 if passed else 2
    else:
        exit_code = 1 if state["failed_commands"] else 0
    state["finished_at"] = utc_now()
    state["status"] = "complete" if exit_code == 0 else "complete_with_failures"
    write_json_atomic(state_path, state)
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
