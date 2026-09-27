"""Disconnect-safe autonomous queue for the tiered TimesFM-2.5 campaign.

Runs the whole TimesFM plan in locked priority order (correctness gate,
B0 timing/quality, stage decomposition, KV diagnostics, S4 staleness
extension, S1 secondary datasets, validation) with a durable cursor so a
dropped SSH session resumes with ``--resume`` and loses at most one task.
Every underlying script is idempotent, so re-running a task is safe.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # experiments/ root

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from shared.common import (
    EXT_DATASETS,
    EXT_K_VALUES,
    EXT_LENGTHS,
    MODELS,
    RESULTS,
    create_manifest,
    gpu_processes,
    load_manifest,
    utc_now,
    write_json_atomic,
)


HERE = Path(__file__).resolve().parent
MODEL = "timesfm"
STATE_NAME = "timesfm_queue_state.json"
LOCK_NAME = "timesfm_queue.lock"

# B0 primary datasets; Weather first is a locked priority decision.
B0_DATASETS = ("Weather", "ETTm1", "ETTm2")


def plan(windows: int = 5) -> dict[str, Any]:
    """Pure description of every grid in the tiered campaign.

    The validator imports this to know what coverage to demand; keep it
    free of I/O and side effects.
    """
    spec = MODELS[MODEL]
    lengths = list(spec.lengths)
    # S1 short-series datasets stop at the main length (8192).
    short_lengths = [length for length in lengths if length <= spec.main_length]
    return {
        "b0_datasets": list(B0_DATASETS),
        "s1": {
            "ETTh1": list(short_lengths),
            "ETTh2": list(short_lengths),
            "Traffic": list(short_lengths),
            "Electricity": list(lengths),
        },
        "lengths": lengths,
        "k_values": list(spec.k_values),
        "windows": windows,
        "ext": {
            "datasets": list(EXT_DATASETS),
            "lengths": list(EXT_LENGTHS),
            "k_values": list(EXT_K_VALUES),
            "windows": windows,
        },
        "exp2_lengths": list(lengths),
        "exp2_batch8_L": spec.main_length,
    }


@dataclass(frozen=True)
class Task:
    phase: str
    command: tuple[str, ...]
    abort_queue_on_failure: bool = False
    is_validation: bool = False


def command(script: str, *args: Any) -> tuple[str, ...]:
    return (sys.executable, str(HERE / script), *(str(value) for value in args))


def shared_command(script: str, *args: Any) -> tuple[str, ...]:
    return (sys.executable, str(HERE.parent / "shared" / script), *(str(value) for value in args))


def quality_command(dataset: str, window: int, length: int, ext: bool = False) -> tuple[str, ...]:
    extra = ("--ext",) if ext else ()
    return shared_command(
        "exp1_fixed_policy_sweep.py", "--mode", "quality", "--model", MODEL,
        "--dataset", dataset, "--window", window, "--L", length, *extra,
    )


def build_tasks(windows: int) -> list[Task]:
    grids = plan(windows)
    tasks: list[Task] = []

    # P-E0: correctness gate; a failure here aborts the whole queue.
    tasks.append(
        Task("P-E0", shared_command("exp0_gates_native.py", "--model", MODEL), abort_queue_on_failure=True)
    )

    # P-B0-timing: latency sweep over every context length.
    for length in grids["lengths"]:
        tasks.append(
            Task(
                "P-B0-timing",
                shared_command("exp1_fixed_policy_sweep.py", "--mode", "timing", "--model", MODEL, "--L", length),
            )
        )

    # P-B0-quality: primary datasets, Weather first.
    for dataset in grids["b0_datasets"]:
        for length in grids["lengths"]:
            for window in range(windows):
                tasks.append(Task("P-B0-quality", quality_command(dataset, window, length)))

    # P-S2: stage decomposition, batch 1 at every length plus batch 8 at L_main.
    for length in grids["exp2_lengths"]:
        tasks.append(
            Task(
                "P-S2",
                shared_command(
                    "exp2_time_attribution.py", "--model", MODEL, "--L", length,
                    "--path", "both", "--batch", 1,
                ),
            )
        )
    tasks.append(
        Task(
            "P-S2",
            shared_command(
                "exp2_time_attribution.py", "--model", MODEL, "--L", grids["exp2_batch8_L"],
                "--path", "both", "--batch", 8,
            ),
        )
    )

    # P-S3: KV-cache diagnostics.
    tasks.append(Task("P-S3", command("exp4_kv_divergence.py")))

    # P-S4: staleness-extension arm (U=512) in its own results tree.
    for length in grids["ext"]["lengths"]:
        tasks.append(
            Task(
                "P-S4",
                shared_command(
                    "exp1_fixed_policy_sweep.py", "--mode", "timing", "--model", MODEL,
                    "--ext", "--L", length,
                ),
            )
        )
    for dataset in grids["ext"]["datasets"]:
        for length in grids["ext"]["lengths"]:
            for window in range(windows):
                tasks.append(Task("P-S4", quality_command(dataset, window, length, ext=True)))

    # P-S1 (last priority): secondary datasets.
    for dataset, s1_lengths in grids["s1"].items():
        for length in s1_lengths:
            for window in range(windows):
                tasks.append(Task("P-S1", quality_command(dataset, window, length)))

    # P-validate: its exit code becomes the queue's final status.
    tasks.append(Task("P-validate", command("validate_campaign.py"), is_validation=True))
    return tasks


def ensure_manifest(windows: int, gpu: int) -> None:
    path = RESULTS / "manifest.json"
    if not path.exists():
        print("manifest.json missing; creating it", flush=True)
        create_manifest(windows, gpu)
        return
    manifest = load_manifest()
    if "windows_ext" not in manifest:
        print("manifest.json predates the S4 extension arm; recreating it", flush=True)
        create_manifest(windows, gpu)
        return
    if any(len(values) != windows for values in manifest.get("windows", {}).values()):
        raise ValueError("--windows differs from the existing manifest")


def update_phase(name: str, state: str, detail: Any = None) -> None:
    manifest = load_manifest()
    manifest.setdefault("phases", {})[name] = {
        "state": state,
        "ts": utc_now(),
        "detail": detail,
    }
    write_json_atomic(RESULTS / "manifest.json", manifest)


def finish_phase(state: dict[str, Any], phase: str) -> None:
    failures = [item for item in state["failed_commands"] if item["phase"] == phase]
    update_phase(
        phase,
        "complete_with_failures" if failures else "complete",
        {"failed_commands": len(failures)},
    )


def wait_for_gpu(gpu: int, seconds: int = 60) -> None:
    announced = False
    while True:
        active = gpu_processes(gpu)
        if not active:
            if announced:
                print(f"GPU {gpu} became free", flush=True)
            return
        if not announced:
            print(f"GPU {gpu} busy; timesfm queue waiting: {json.dumps(active)}", flush=True)
            announced = True
        time.sleep(seconds)


def run_task(task: Task, gpu: int, retries: int) -> tuple[bool, int, int]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = str(HERE.parent)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    code = -1
    for attempt in range(1, retries + 2):
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--windows", type=int, default=5)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    lock_handle = (RESULTS / LOCK_NAME).open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("another timesfm/launch_campaign.py instance already owns the queue lock")

    ensure_manifest(args.windows, args.gpu)
    tasks = build_tasks(args.windows)
    state_path = RESULTS / STATE_NAME
    state: dict[str, Any] = {
        "schema": "timesfm-tiered-queue",
        "task_count": len(tasks),
        "next_index": 0,
        "failed_commands": [],
        "validate_exit_code": None,
        "started_at": utc_now(),
        "updated_at": utc_now(),
    }
    if args.resume and state_path.exists():
        old = json.loads(state_path.read_text())
        if old.get("task_count") == len(tasks):
            state.update(old)
            state["updated_at"] = utc_now()
        else:
            print("saved state has a different task count; starting fresh", flush=True)

    current_phase: str | None = None
    for index in range(int(state.get("next_index", 0)), len(tasks)):
        task = tasks[index]
        if task.phase != current_phase:
            if current_phase is not None:
                finish_phase(state, current_phase)
            current_phase = task.phase
            update_phase(current_phase, "running")

        started = time.monotonic()
        success, attempts, last_code = run_task(task, args.gpu, args.retries)
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
            update_phase(task.phase, "aborted", {"exit_code": last_code})
            state["status"] = "aborted_correctness_gate"
            state["finished_at"] = utc_now()
            write_json_atomic(state_path, state)
            print(
                f"ABORT correctness gate failed (exit={last_code}); stopping the queue",
                flush=True,
            )
            return 1

    if current_phase is not None:
        finish_phase(state, current_phase)

    exit_code = state.get("validate_exit_code")
    if exit_code is None:
        # Validation never ran in any session; fall back to its artifact.
        validation_path = RESULTS / "validation_timesfm.json"
        passed = False
        if validation_path.exists():
            passed = bool(json.loads(validation_path.read_text()).get("passed"))
        exit_code = 0 if passed else 2
    state["finished_at"] = utc_now()
    state["status"] = "complete" if exit_code == 0 else "complete_with_failures"
    write_json_atomic(state_path, state)
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
