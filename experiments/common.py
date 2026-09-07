"""Shared protocol and append-only I/O helpers for the experiments."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
# ``env.sh`` already uses ROLLKV_RESULTS for the repository-wide results root.
# Keep the results contract independent and offer a deliberately named override for
# isolated smoke tests.
RESULTS = Path(
    os.environ.get("ROLLKV_RESULTS", str(ROOT / "results"))
)


def _first_existing(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[-1]


# The artifact ships the seven model repositories under ``models/``; the
# original development tree kept them directly at the repository root. Resolve
# whichever is present so both layouts run the same harness unmodified.
MODELS_ROOT = Path(
    os.environ.get(
        "ROLLKV_MODELS", str(_first_existing(ROOT / "models", ROOT))
    )
)
# Checkpoints and datasets are too large to ship; the top-level ``README.md``
# says where to get them and what the directory names have to be.
CHECKPOINTS = Path(os.environ.get("ROLLKV_CKPT", str(ROOT / "checkpoints")))
DATASET_ROOT = Path(os.environ.get("ROLLKV_DATASETS", str(ROOT / "datasets")))
SEED_BASE = 20260819
W1_MAX_FUTURE = 64 * 32 + 64


@dataclass(frozen=True)
class DatasetSpec:
    path: str
    column: str


@dataclass(frozen=True)
class ModelSpec:
    name: str
    display_name: str
    wave: str
    repo: str
    checkpoint: str
    s: int
    horizon: int
    updates: int
    lengths: tuple[int, ...]
    main_length: int
    k_values: tuple[int, ...]
    dtype: str
    pos_remap: str
    remap_values: tuple[str, ...]


DATASETS: dict[str, DatasetSpec] = {
    "ETTh1": DatasetSpec("ETT-small/ETTh1.csv", "OT"),
    "ETTh2": DatasetSpec("ETT-small/ETTh2.csv", "OT"),
    "ETTm1": DatasetSpec("ETT-small/ETTm1.csv", "OT"),
    "ETTm2": DatasetSpec("ETT-small/ETTm2.csv", "OT"),
    "Electricity": DatasetSpec("electricity/electricity.csv", "0"),
    "Traffic": DatasetSpec("traffic/traffic.csv", "0"),
    "Weather": DatasetSpec("weather/weather.csv", "T (degC)"),
}


MODELS: dict[str, ModelSpec] = {
    "timesfm": ModelSpec(
        name="timesfm",
        display_name="TimesFM-2.5-200M",
        wave="W1",
        repo="TimesFM-2.5",
        checkpoint="TimesFM-2.5-200M/model.safetensors",
        s=32,
        horizon=64,
        updates=64,
        lengths=(512, 2048, 8192, 16384),
        main_length=8192,
        k_values=(
            1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64,
            128, 256, 512, 1024, 0,
        ),
        dtype="float32",
        pos_remap="n/a",
        remap_values=("n/a",),
    ),
    "timemoe": ModelSpec(
        name="timemoe",
        display_name="TimeMoE-50M",
        wave="W1",
        repo="Time-MoE",
        checkpoint="TimeMoE-50M",
        s=1,
        horizon=64,
        updates=1024,
        lengths=(512, 2048, 4096, 8192),
        main_length=8192,
        k_values=(1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128, 256, 512, 1024, 0),
        dtype="bfloat16",
        # The current implementation uses the model's window-relative default
        # positions but has no independently switchable survivor-key remapper.
        pos_remap="n/a",
        remap_values=("n/a",),
    ),
    "sundial": ModelSpec(
        name="sundial",
        display_name="Sundial-base-128m",
        wave="W2",
        repo="Sundial-HF",
        checkpoint="Sundial-base-128M",
        s=16,
        horizon=96,
        updates=64,
        lengths=(480, 960, 1920, 2880),
        main_length=2880,
        k_values=(1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 0),
        dtype="float32",
        pos_remap="on",
        remap_values=("on", "off"),
    ),
    "timer": ModelSpec(
        name="timer",
        display_name="Timer-base-84M",
        wave="W2",
        repo="Timer-HF",
        checkpoint="Timer-base-84M",
        s=96,
        horizon=96,
        updates=64,
        lengths=(480, 960, 1920, 2880),
        main_length=2880,
        k_values=(1, 2, 3, 4, 5, 6, 8, 11, 12, 16, 21, 24, 32, 43, 48, 64, 0),
        dtype="float32",
        pos_remap="on",
        remap_values=("on", "off"),
    ),
    "toto2": ModelSpec(
        name="toto2",
        display_name="Toto-2.0-313m",
        wave="W3",
        repo="Toto",
        checkpoint="Toto-2.0-313m",
        s=32,
        horizon=32,
        updates=64,
        lengths=(2048, 4096, 8192),
        main_length=8192,
        k_values=(1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 0),
        dtype="float32",
        pos_remap="on",
        remap_values=("on", "off"),
    ),
    "timerxl": ModelSpec(
        name="timerxl",
        display_name="Timer-XL-67M",
        wave="W3",
        repo="OpenLTM",
        checkpoint="Timer-XL-67M/checkpoint.pth",
        s=96,
        horizon=96,
        updates=64,
        lengths=(3072, 6144, 12288, 24576),
        main_length=12288,
        k_values=(1, 2, 3, 4, 5, 6, 8, 11, 12, 16, 21, 24, 32, 43, 48, 64, 0),
        dtype="float32",
        pos_remap="on",
        remap_values=("on", "off"),
    ),
    "lagllama": ModelSpec(
        name="lagllama",
        display_name="Lag-Llama",
        wave="W3",
        repo="Lag-Llama",
        checkpoint="Lag-Llama/lag-llama.ckpt",
        s=1,
        horizon=1,
        updates=1024,
        lengths=(1024, 2048, 4096),
        main_length=4096,
        k_values=(1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 128, 256, 512, 1024, 0),
        dtype="float32",
        pos_remap="n/a",
        remap_values=("n/a",),
    ),
}

MODEL_WAVES = {
    wave: tuple(name for name, spec in MODELS.items() if spec.wave == wave)
    for wave in ("W1", "W2", "W3")
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one durable UTF-8 record while holding an advisory file lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(json_safe(record), ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def write_jsonl_create_once(path: Path, records: Iterable[dict[str, Any]]) -> bool:
    """Atomically publish a new immutable JSONL artifact without overwriting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(json_safe(record), ensure_ascii=False, sort_keys=True) + "\n"
        for record in records
    ]
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temp.open("x", encoding="utf-8") as handle:
            handle.writelines(lines)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp, path)
            return True
        except FileExistsError:
            return False
    finally:
        try:
            temp.unlink()
        except FileNotFoundError:
            pass


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temp.write_text(
        json.dumps(json_safe(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def run_output(command: list[str]) -> str:
    return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()


def _pinned_upstream_sha(repo: str) -> str | None:
    """Upstream commit recorded in ``models/UPSTREAM.json``.

    The vendored artifact copies carry no ``.git``, so the live ``rev-parse``
    below cannot run there. The pin keeps the recorded provenance identical to
    the original campaign instead of degrading to ``deployment_sha()``.
    """
    manifest = MODELS_ROOT / "UPSTREAM.json"
    try:
        entry = json.loads(manifest.read_text(encoding="utf-8"))[repo]
    except (OSError, ValueError, KeyError):
        return None
    sha = entry.get("commit")
    return sha if isinstance(sha, str) and sha else None


def repo_sha(model: str) -> str:
    spec = MODELS[model]
    try:
        return run_output(
            ["git", "-C", str(MODELS_ROOT / spec.repo), "rev-parse", "HEAD"]
        )
    except (OSError, subprocess.CalledProcessError):
        return _pinned_upstream_sha(spec.repo) or deployment_sha()


def deployment_sha() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run_id(exp: str, model: str) -> str:
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    return f"{exp}-{model}-{day}-{repo_sha(model)[:7]}"


def base_record(schema: str, exp: str, model: str) -> dict[str, Any]:
    return {
        "schema": schema,
        "run_id": run_id(exp, model),
        "model": model,
        "git_sha": repo_sha(model),
        "ts": utc_now(),
        "status": "ok",
        "reason": None,
    }


def stable_policy_id(payload: dict[str, Any]) -> str:
    raw = json.dumps(json_safe(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def exp1_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("dataset"),
        row.get("window"),
        row.get("L"),
        row.get("K"),
        row.get("pos_remap"),
    )


def timing_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row.get("L"), row.get("K"), row.get("pos_remap"), row.get("exec"), row.get("batch"))


def completed_keys(path: Path, key_fn) -> set[tuple[Any, ...]]:
    latest: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in read_jsonl(path):
        latest[key_fn(row)] = row
    return {
        key
        for key, row in latest.items()
        if row.get("status") in {"ok", "unsupported", "oom"}
    }


def load_series(dataset: str) -> np.ndarray:
    import pandas as pd

    spec = DATASETS[dataset]
    frame = pd.read_csv(DATASET_ROOT / spec.path)
    if spec.column not in frame:
        raise ValueError(f"column {spec.column!r} absent from {dataset}")
    series = pd.to_numeric(frame[spec.column], errors="raise").to_numpy(np.float32)
    if not np.isfinite(series).all():
        raise ValueError(f"{dataset} contains NaN or infinite values")
    return series


def choose_windows(series_length: int, count: int = 5) -> list[int]:
    """Choose shared W1 forecast endpoints in the last 30% of a series.

    ``window_start`` is the first newly observed point (the history ends just
    before it).  The plan's literal L_max-inclusive inequality is infeasible
    for several supplied datasets, so L feasibility is checked per grid cell
    and recorded as ``unsupported`` instead of silently changing the windows.
    """
    low = int(math.ceil(series_length * 0.70))
    high = series_length - W1_MAX_FUTURE
    if high < low:
        raise ValueError(
            f"series length {series_length} has no W1 endpoint in its last 30% "
            f"with {W1_MAX_FUTURE} future points"
        )
    windows = []
    for index in range(count):
        rng = np.random.RandomState(SEED_BASE + index)
        windows.append(int(rng.randint(low, high + 1)))
    return windows


def gpu_snapshot(gpu: int) -> dict[str, Any]:
    query = "index,name,uuid,memory.total,memory.used,utilization.gpu,driver_version"
    text = run_output(
        [
            "nvidia-smi",
            f"--id={gpu}",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ]
    )
    values = [item.strip() for item in text.split(",")]
    return dict(zip(query.split(","), values))


def gpu_processes(gpu: int) -> list[dict[str, str]]:
    gpu_info = gpu_snapshot(gpu)
    try:
        text = run_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ]
        )
    except subprocess.CalledProcessError:
        return []
    rows = []
    for line in text.splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) == 4 and fields[0] == gpu_info["uuid"]:
            rows.append(dict(zip(("gpu_uuid", "pid", "process_name", "used_gpu_memory_mb"), fields)))
    return rows


def create_manifest(window_count: int, gpu: int) -> dict[str, Any]:
    import pandas as pd
    import torch
    import transformers

    windows = {}
    lengths = {}
    for dataset in DATASETS:
        series = load_series(dataset)
        lengths[dataset] = int(series.size)
        windows[dataset] = choose_windows(series.size, window_count)
    snapshot = gpu_snapshot(gpu)
    manifest = {
        "schema_version": "rolling-kv",
        "wave": "W1",
        "git_sha": {
            "deployment": deployment_sha(),
            **{model: repo_sha(model) for model in MODELS},
        },
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "driver": snapshot["driver_version"],
        "transformers": transformers.__version__,
        "gpu_index": gpu,
        "gpu_name": snapshot["name"],
        "gpu_uuid": snapshot["uuid"],
        "hostname": socket.gethostname(),
        "started_at": utc_now(),
        "finished_at": None,
        "seed_base": SEED_BASE,
        "windows": windows,
        "series_lengths": lengths,
        "window_start_semantics": "first newly observed point; history is [start-L,start)",
        "other_processes_on_gpu": gpu_processes(gpu),
        "protocol_notes": [
            "W1 windows are shared by TimesFM and TimeMoE.",
            "Per-cell history/future infeasibility is recorded as status=unsupported.",
            "TimeMoE position remapping is n/a because the current engine has no independently switchable remapper.",
        ],
        "models": {name: spec.__dict__ for name, spec in MODELS.items()},
        "datasets": {name: spec.__dict__ for name, spec in DATASETS.items()},
        "pandas": pd.__version__,
    }
    write_json_atomic(RESULTS / "manifest.json", manifest)
    return manifest


def load_manifest() -> dict[str, Any]:
    path = RESULTS / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run launch_all.py first")
    return json.loads(path.read_text(encoding="utf-8"))


def classify_failure(exc: BaseException) -> str:
    message = str(exc).lower()
    if "out of memory" in message or "cuda error: out of memory" in message:
        return "oom"
    if "capture" in message or "cuda graph" in message:
        return "capture_failed"
    if isinstance(exc, (ValueError, FileNotFoundError)):
        return "unsupported"
    return "failed"


def metric_summary(values: Iterable[float]) -> dict[str, Any]:
    array = np.asarray(list(values), dtype=np.float64)
    return {
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "std": float(np.std(array)),
        "n": int(array.size),
    }


def quality_metrics(
    yhat: np.ndarray, y: np.ndarray, ref_yhat: np.ndarray, ref_mae: float | None = None
) -> dict[str, Any]:
    yhat = np.asarray(yhat, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    ref_yhat = np.asarray(ref_yhat, dtype=np.float32)
    mae = float(np.abs(yhat - y).mean())
    mse = float(np.square(yhat - y).mean())
    ref_mae = float(np.abs(ref_yhat - y).mean()) if ref_mae is None else float(ref_mae)
    ref_mse = float(np.square(ref_yhat - y).mean())
    gap = float(np.abs(yhat - ref_yhat).mean() / max(ref_mae, 1e-12) * 100.0)
    result = {
        "mae_native": mae,
        "mse_native": mse,
        "mae_h32": float(np.abs(yhat[:, :32] - y[:, :32]).mean()) if yhat.shape[1] >= 32 else None,
        "mse_h32": float(np.square(yhat[:, :32] - y[:, :32]).mean()) if yhat.shape[1] >= 32 else None,
        "mae_ref_native": ref_mae,
        "mae_ref_h32": float(np.abs(ref_yhat[:, :32] - y[:, :32]).mean()) if yhat.shape[1] >= 32 else None,
        "gap_pct": gap,
        "mae_delta_pct": float((mae - ref_mae) / max(ref_mae, 1e-12) * 100.0),
        "mae_delta_h32_pct": None,
        "mse_delta_pct": float((mse - ref_mse) / max(ref_mse, 1e-12) * 100.0),
    }
    if yhat.shape[1] >= 32:
        mae32 = result["mae_h32"]
        ref32 = result["mae_ref_h32"]
        result["mae_delta_h32_pct"] = float((mae32 - ref32) / max(ref32, 1e-12) * 100.0)
    return result
