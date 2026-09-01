"""Run one model wave as an autonomous resumable queue."""

from __future__ import annotations

import argparse
import fcntl
import json
from typing import Any

from common import RESULTS, gpu_processes, gpu_snapshot, utc_now, write_json_atomic
from launch_all import build_tasks, exp0_gate_ok, run_task


def update_phase(state: dict[str, Any], phase: str, status: str) -> None:
    failures = [item for item in state["failed_commands"] if item["phase"] == phase]
    state.setdefault("phases", {})[phase] = {
        "state": status,
        "failed_commands": len(failures),
        "ts": utc_now(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wave", choices=("W1", "W2", "W3"), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--windows", type=int, default=5)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    stem = args.wave.lower()
    lock_handle = (RESULTS / f"{stem}_queue.lock").open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError(f"another {args.wave} queue already owns the lock") from exc

    tasks = [
        task
        for task in build_tasks(args.windows)
        if task.phase.startswith(f"{args.wave}_")
    ]
    state_path = RESULTS / f"{stem}_queue_state.json"
    state: dict[str, Any] = {
        "schema": "rolling-kv-wave-queue",
        "wave": args.wave,
        "gpu_index": args.gpu,
        "task_count": len(tasks),
        "next_index": 0,
        "failed_commands": [],
        "disabled_models": [],
        "phases": {},
        "started_at": utc_now(),
        "updated_at": utc_now(),
    }
    if args.resume and state_path.exists():
        old = json.loads(state_path.read_text())
        if old.get("wave") != args.wave or old.get("task_count") != len(tasks):
            raise RuntimeError("existing wave state does not match this queue")
        state.update(old)
        state["gpu_index"] = args.gpu
        state["updated_at"] = utc_now()

    if int(state.get("next_index", 0)) == 0:
        active = gpu_processes(args.gpu)
        if active:
            raise RuntimeError(f"selected GPU {args.gpu} is not empty at launch: {active}")
        state["gpu_snapshot_at_launch"] = gpu_snapshot(args.gpu)
        write_json_atomic(state_path, state)

    disabled = set(state.get("disabled_models", []))
    current_phase = state.get("current_phase")
    for index in range(int(state.get("next_index", 0)), len(tasks)):
        task = tasks[index]
        if task.phase != current_phase:
            if current_phase:
                failures = [
                    item
                    for item in state["failed_commands"]
                    if item["phase"] == current_phase
                ]
                update_phase(
                    state,
                    current_phase,
                    "complete_with_failures" if failures else "complete",
                )
            current_phase = task.phase
            state["current_phase"] = current_phase
            update_phase(state, current_phase, "running")
            state["updated_at"] = utc_now()
            write_json_atomic(state_path, state)

        if task.model in disabled and task.gate_required:
            print(f"SKIP invalid model after EXP0 gate failure: {task.model}", flush=True)
            success, attempts = True, 0
        else:
            success, attempts = run_task(task, args.gpu, args.retries)

        if task.phase.endswith("EXP0") and task.model and not exp0_gate_ok(task.model):
            disabled.add(task.model)
            state["disabled_models"] = sorted(disabled)
        if not success:
            state["failed_commands"].append(
                {
                    "index": index,
                    "phase": task.phase,
                    "model": task.model,
                    "command": list(task.command),
                    "attempts": attempts,
                    "ts": utc_now(),
                }
            )
        state.update(
            {
                "next_index": index + 1,
                "current_phase": task.phase,
                "current_model": task.model,
                "last_command": list(task.command),
                "last_command_ok": success,
                "updated_at": utc_now(),
            }
        )
        write_json_atomic(state_path, state)

    if current_phase:
        failures = [
            item for item in state["failed_commands"] if item["phase"] == current_phase
        ]
        update_phase(
            state,
            current_phase,
            "complete_with_failures" if failures else "complete",
        )
    state["finished_at"] = utc_now()
    state["status"] = (
        "complete" if not state["failed_commands"] else "complete_with_failures"
    )
    state["updated_at"] = utc_now()
    write_json_atomic(state_path, state)
    return 0 if not state["failed_commands"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
