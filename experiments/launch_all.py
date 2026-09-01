"""Autonomously run the complete experiment queue, with resume."""

from __future__ import annotations

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

from common import (
    DATASETS,
    MODELS,
    MODEL_WAVES,
    RESULTS,
    create_manifest,
    gpu_processes,
    gpu_snapshot,
    load_manifest,
    read_jsonl,
    utc_now,
    write_json_atomic,
)


HERE = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Task:
    phase: str
    model: str | None
    command: tuple[str, ...]
    gate_required: bool = True


def command(script: str, *args: Any) -> tuple[str, ...]:
    return (sys.executable, str(HERE / script), *(str(value) for value in args))


def build_tasks(windows: int) -> list[Task]:
    tasks: list[Task] = []
    for wave in ("W1", "W2", "W3"):
        models = MODEL_WAVES[wave]
        for model in models:
            script = "exp0_w1.py" if wave == "W1" else "exp0_all.py"
            tasks.append(Task(f"{wave}_EXP0", model, command(script, "--model", model), False))

        for model in models:
            spec = MODELS[model]
            for remap in spec.remap_values:
                for length in spec.lengths:
                    tasks.append(
                        Task(
                            f"{wave}_EXP1_timing",
                            model,
                            command(
                                "exp1_w1.py", "--mode", "timing", "--model", model,
                                "--L", length, "--pos-remap", remap,
                            ),
                        )
                    )
            for remap in spec.remap_values:
                for dataset in DATASETS:
                    for window in range(windows):
                        for length in spec.lengths:
                            tasks.append(
                                Task(
                                    f"{wave}_EXP1_K1",
                                    model,
                                    command(
                                        "exp1_w1.py", "--mode", "quality", "--model", model,
                                        "--dataset", dataset, "--window", window, "--L", length,
                                        "--k-set", "baseline", "--pos-remap", remap,
                                    ),
                                )
                            )
            for remap in spec.remap_values:
                for dataset in DATASETS:
                    for window in range(windows):
                        for length in spec.lengths:
                            tasks.append(
                                Task(
                                    f"{wave}_EXP1_sweep",
                                    model,
                                    command(
                                        "exp1_w1.py", "--mode", "quality", "--model", model,
                                        "--dataset", dataset, "--window", window, "--L", length,
                                        "--k-set", "rest", "--pos-remap", remap,
                                    ),
                                )
                            )

        for model in models:
            spec = MODELS[model]
            for length in spec.lengths:
                tasks.append(
                    Task(
                        f"{wave}_EXP2",
                        model,
                        command("exp2_stages.py", "--model", model, "--L", length, "--path", "both", "--batch", 1),
                    )
                )
            tasks.append(
                Task(
                    f"{wave}_EXP2",
                    model,
                    command("exp2_stages.py", "--model", model, "--L", spec.main_length, "--path", "both", "--batch", 8),
                )
            )

        for model in models:
            spec = MODELS[model]
            lengths = spec.lengths if wave == "W1" else (spec.main_length,)
            for dataset in DATASETS:
                for window in range(windows):
                    for length in lengths:
                        tasks.append(
                            Task(
                                f"{wave}_EXP3",
                                model,
                                command(
                                    "exp3_adaptive.py", "--model", model, "--dataset", dataset,
                                    "--window", window, "--L", length, "--mode", "all",
                                ),
                            )
                        )

        for model in models:
            for batch in (1, 4, 8, 16, 32):
                tasks.append(
                    Task(
                        f"{wave}_EXP5",
                        model,
                        command("exp5_appendix.py", "--mode", "eager-graph", "--model", model, "--batch", batch),
                    )
                )
            if model == "sundial":
                tasks.append(
                    Task(
                        f"{wave}_EXP5",
                        model,
                        command("exp5_appendix.py", "--mode", "timeflow"),
                    )
                )

    tasks.append(
        Task("aggregate", None, command("aggregate.py", "--verify-determinism"), False)
    )
    tasks.append(Task("validation", None, command("validate_all.py"), False))
    return tasks


def exp0_gate_ok(model: str) -> bool:
    latest = {}
    path = RESULTS / "EXP0_correctness" / model / "records.jsonl"
    for row in read_jsonl(path):
        latest[(row.get("L"), row.get("gate"), row.get("cache_age"))] = row
    for length in MODELS[model].lengths:
        for gate in ("T1", "T2", "T3", "T4"):
            row = latest.get((length, gate, None))
            if not row or row.get("status") != "ok" or row.get("passed") is not True:
                return False
    return True


def update_manifest(**updates: Any) -> None:
    manifest = load_manifest()
    manifest.update(updates)
    write_json_atomic(RESULTS / "manifest.json", manifest)


def update_phase(name: str, state: str, detail: Any = None) -> None:
    manifest = load_manifest()
    manifest.setdefault("phases", {})[name] = {
        "state": state,
        "ts": utc_now(),
        "detail": detail,
    }
    write_json_atomic(RESULTS / "manifest.json", manifest)


def ensure_manifest(windows: int, gpu: int) -> dict[str, Any]:
    path = RESULTS / "manifest.json"
    manifest = load_manifest() if path.exists() else create_manifest(windows, gpu)
    if any(len(values) != windows for values in manifest.get("windows", {}).values()):
        raise ValueError("--windows differs from the existing manifest")
    active = gpu_processes(gpu)
    if active:
        raise RuntimeError(f"selected GPU {gpu} is not empty at launch: {active}")
    snapshot = gpu_snapshot(gpu)
    previous_gpu = manifest.get("gpu_index")
    if previous_gpu != gpu:
        manifest.setdefault("gpu_migrations", []).append(
            {
                "ts": utc_now(),
                "from_gpu_index": previous_gpu,
                "from_gpu_uuid": manifest.get("gpu_uuid"),
                "to_gpu_index": gpu,
                "to_gpu_uuid": snapshot["uuid"],
                "reason": "complete_plan_autonomous_queue",
            }
        )
    manifest.update(
        {
            "wave": "ALL",
            "waves": ["W1", "W2", "W3"],
            "gpu_index": gpu,
            "gpu_name": snapshot["name"],
            "gpu_uuid": snapshot["uuid"],
            "driver": snapshot["driver_version"],
            "other_processes_on_gpu": active,
            "models": {name: spec.__dict__ for name, spec in MODELS.items()},
            "all_experiments_started_at": manifest.get("all_experiments_started_at") or utc_now(),
            "all_experiments_finished_at": None,
            "all_validation_passed": None,
        }
    )
    write_json_atomic(path, manifest)
    return manifest


def wait_for_gpu(gpu: int, seconds: int = 60) -> None:
    announced = False
    while True:
        active = gpu_processes(gpu)
        if not active:
            if announced:
                print(f"GPU {gpu} became free", flush=True)
            return
        if not announced:
            print(f"GPU {gpu} busy; autonomous queue waiting: {json.dumps(active)}", flush=True)
            announced = True
        time.sleep(seconds)


def run_task(task: Task, gpu: int, retries: int) -> tuple[bool, int]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = str(HERE)
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    for attempt in range(1, retries + 2):
        wait_for_gpu(gpu)
        print(
            f"RUN phase={task.phase} model={task.model} attempt={attempt}: "
            f"{' '.join(task.command)}",
            flush=True,
        )
        code = subprocess.run(task.command, env=env).returncode
        if code == 0:
            return True, attempt
        print(f"command failed exit={code}", flush=True)
        if attempt <= retries:
            time.sleep(min(60, attempt * 10))
    return False, retries + 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--windows", type=int, default=5)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    lock_handle = (RESULTS / "all_queue.lock").open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("another launch_all.py instance already owns the queue lock")

    ensure_manifest(args.windows, args.gpu)
    tasks = build_tasks(args.windows)
    state_path = RESULTS / "all_queue_state.json"
    state = {
        "schema": "rolling-kv-queue",
        "task_count": len(tasks),
        "next_index": 0,
        "failed_commands": [],
        "disabled_models": [],
        "started_at": utc_now(),
        "updated_at": utc_now(),
    }
    if args.resume and state_path.exists():
        old = json.loads(state_path.read_text())
        if old.get("task_count") == len(tasks):
            state.update(old)
            state["updated_at"] = utc_now()
    disabled = set(state.get("disabled_models", []))
    current_phase = None

    for index in range(int(state.get("next_index", 0)), len(tasks)):
        task = tasks[index]
        if task.phase != current_phase:
            if current_phase is not None:
                failures = [item for item in state["failed_commands"] if item["phase"] == current_phase]
                update_phase(
                    current_phase,
                    "complete_with_failures" if failures else "complete",
                    {"failed_commands": len(failures)},
                )
            current_phase = task.phase
            update_phase(current_phase, "running")

        if task.model in disabled and task.gate_required:
            print(f"SKIP invalid model after EXP0 gate failure: {task.model}", flush=True)
            state["next_index"] = index + 1
            state["updated_at"] = utc_now()
            write_json_atomic(state_path, state)
            continue

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

    if current_phase is not None:
        failures = [item for item in state["failed_commands"] if item["phase"] == current_phase]
        update_phase(
            current_phase,
            "complete_with_failures" if failures else "complete",
            {"failed_commands": len(failures)},
        )
    validation_path = RESULTS / "validation_all.json"
    passed = False
    if validation_path.exists():
        passed = bool(json.loads(validation_path.read_text()).get("passed"))
    state["finished_at"] = utc_now()
    state["status"] = "complete" if passed else "complete_with_failures"
    write_json_atomic(state_path, state)
    update_manifest(
        all_experiments_finished_at=utc_now(),
        all_validation_passed=passed,
        finished_at=utc_now() if passed else None,
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
