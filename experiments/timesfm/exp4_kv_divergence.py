"""EXP-4 KV-divergence probe + reducer for TimesFM rolling caches.

Wraps the vendored diagnostic of the selected model

* ``--model timesfm`` (default, TimesFM-2.5):
  ``models/TimesFM-2.5/scripts/online_benchmark/compare_kv_intermediates.py``
* ``--model timesfm3`` (TimesFM-3.0):
  ``models/TimesFM-3.0/scripts/online_benchmark/compare_kv_intermediates.py``

(each measures per-(cache age, layer) K/V divergence between rolling-cache
and full-compute inference on the same fixed synthetic series, with a RoPE
rebase self-check) and reduces its raw output into the paper's plot contract.

TimesFM-2.5 (unchanged layout) writes under ``EXP4_kvdiag/``:

* ``EXP4_kvdiag/raw/``              -- the probe's full JSON/CSV, untouched;
* ``EXP4_kvdiag/reduced_k12.csv``   -- ages {1..2048 powers of two} x layer
  x scope {all,newest,survivors} x tensor {k_direct,k_rebased,v};
* ``EXP4_kvdiag/summary_per_age.json`` -- provenance plus per-age forecast
  divergence (rel_l2 and mse for every age);
* ``EXP4_kvdiag/records.jsonl``     -- one schema-E provenance record.

TimesFM-3.0 writes the same four artifacts under ``EXP4_kvdiag/timesfm3/``,
shaped like the paper's TimesFM-2.5 inputs
(``plot/2026VLDB/sections/53-kv-divergence/data/raw/k2048/``):
``reduced_k12.csv`` has the ten paper columns with rows ordered
age, layer, scope {all,survivors,newest}, tensor; ``summary_per_age.json``
has exactly the paper file's top-level keys (model, checkpoint, config,
environment, protocol, metric_definitions, rebase_self_check_max_abs, steps)
and per-age steps with age, rolling_position_first/last, forecast (+ mse),
normalization and the global (all-layer) metrics.

``--smoke`` shrinks eviction to 8 steps and writes everything under
``<out>/smoke/`` instead, so a rehearsal never shadows the real run.
GPU required at runtime; ``--help`` and ``py_compile`` work anywhere.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # experiments/ root

import argparse
import csv
import json
import os
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from shared.common import (
    CHECKPOINTS,
    MODELS,
    MODELS_ROOT,
    RESULTS,
    append_jsonl,
    base_record,
    classify_failure,
    read_jsonl,
    write_json_atomic,
)

PROBE_PATH = (
    MODELS_ROOT
    / "TimesFM-2.5"
    / "scripts"
    / "online_benchmark"
    / "compare_kv_intermediates.py"
)
HORIZON = 128  # the probe's forecast-column horizon (its own upper bound)
SMOKE_EVICT_STEPS = 8
K12_AGES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)
SCOPES = ("all", "newest", "survivors")
TENSORS = ("k_direct", "k_rebased", "v")
METRIC_FIELDS = ("mae", "rmse", "max_abs", "rel_l2", "normalized_mae", "cosine_distance")
REDUCED_FIELDS = ("age", "layer", "scope", "tensor", *METRIC_FIELDS)
CONFIG_MATCH_KEYS = ("context_length", "evict_steps", "seed", "dtype")
EXPECTED_LAYERS = 20

#: Keys of one ``steps[]`` entry in the paper's TimesFM-2.5 summary_per_age.json.
PAPER_STEP_KEYS = (
    "age",
    "rolling_position_first",
    "rolling_position_last",
    "forecast",
    "normalization",
    "global",
)


@dataclass(frozen=True)
class ProbeSpec:
    """Per-model probe wiring; ``timesfm`` reproduces the original constants."""

    display: str
    repo: str
    module_name: str
    default_checkpoint: str  # relative to CHECKPOINTS
    out_subdir: str | None  # None == EXP4_kvdiag/ itself (TimesFM-2.5 layout)
    scopes: tuple[str, ...]  # reduced_k12.csv scope order
    paper_summary: bool  # summary_per_age.json shaped like the paper's file
    dtypes: tuple[str, ...]


PROBES: dict[str, ProbeSpec] = {
    "timesfm": ProbeSpec(
        display="TimesFM-2.5",
        repo="TimesFM-2.5",
        module_name="compare_kv_intermediates",
        default_checkpoint="TimesFM-2.5-200M/model.safetensors",
        out_subdir=None,
        scopes=SCOPES,
        paper_summary=False,
        dtypes=("float32", "bfloat16"),
    ),
    "timesfm3": ProbeSpec(
        display="TimesFM-3.0",
        repo="TimesFM-3.0",
        module_name="compare_kv_intermediates_timesfm3",
        default_checkpoint="TimesFM-3.0",
        out_subdir="timesfm3",
        # the paper's reduced_k12.csv lists scopes in the probe's own order
        scopes=("all", "survivors", "newest"),
        paper_summary=True,
        # the vendored TimesFM-3.0 forward is fp32-only
        dtypes=("float32",),
    ),
}


def probe_path(model: str) -> Path:
    return (
        MODELS_ROOT
        / PROBES[model].repo
        / "scripts"
        / "online_benchmark"
        / "compare_kv_intermediates.py"
    )


def checkpoint_path(model: str) -> Path:
    spec = MODELS.get(model)
    relative = spec.checkpoint if spec is not None else PROBES[model].default_checkpoint
    return CHECKPOINTS / relative


def expected_layers(model: str) -> int:
    """Layer count the reducer checks against (read from the checkpoint when possible)."""
    if model == "timesfm3":
        ckpt = checkpoint_path(model)
        config_path = (ckpt if ckpt.is_dir() else ckpt.parent) / "config.json"
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            return int(config["transformer_config"]["num_layers"])
        except (OSError, ValueError, KeyError, TypeError):
            return EXPECTED_LAYERS
    return EXPECTED_LAYERS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=tuple(PROBES),
        default="timesfm",
        help="timesfm = TimesFM-2.5 (default, original layout); timesfm3 = TimesFM-3.0",
    )
    parser.add_argument("--context-length", type=int, default=16384)
    parser.add_argument("--evict-steps", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=f"use evict-steps={SMOKE_EVICT_STEPS} and write under <out>/smoke/",
    )
    args = parser.parse_args(argv)
    if args.context_length < 32 or args.context_length % 32:
        parser.error("--context-length must be >=32 and divisible by 32")
    if args.evict_steps < 1:
        parser.error("--evict-steps must be >= 1")
    if args.dtype not in PROBES[args.model].dtypes:
        parser.error(
            f"--dtype {args.dtype} unsupported for --model {args.model} "
            f"(supported: {', '.join(PROBES[args.model].dtypes)})"
        )
    return args


def requested_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "context_length": args.context_length,
        "evict_steps": SMOKE_EVICT_STEPS if args.smoke else args.evict_steps,
        "seed": args.seed,
        "dtype": args.dtype,
        "horizon": HORIZON,
        "device": "cuda",
        "smoke": bool(args.smoke),
    }


def output_root(model: str, smoke: bool) -> Path:
    out_root = RESULTS / "EXP4_kvdiag"
    subdir = PROBES[model].out_subdir
    if subdir:
        out_root = out_root / subdir
    if smoke:
        out_root = out_root / "smoke"
    return out_root


def config_matches(existing: dict[str, Any] | None, wanted: dict[str, Any]) -> bool:
    if not isinstance(existing, dict):
        return False
    return all(existing.get(key) == wanted[key] for key in CONFIG_MATCH_KEYS)


def already_complete(out_root: Path, wanted: dict[str, Any]) -> bool:
    reduced = out_root / "reduced_k12.csv"
    summary = out_root / "summary_per_age.json"
    if not (reduced.exists() and summary.exists()):
        return False
    try:
        payload = json.loads(summary.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return config_matches(payload.get("config"), wanted)


def has_completion_record(records_path: Path, wanted: dict[str, Any]) -> bool:
    for row in read_jsonl(records_path):
        if row.get("status") == "ok" and config_matches(row.get("config"), wanted):
            return True
    return False


def _load_probe_module(model: str = "timesfm"):
    """Import the vendored probe so its functions and CLI driver are reusable."""
    import importlib.util

    path = probe_path(model)
    name = PROBES[model].module_name
    if not path.exists():
        raise FileNotFoundError(f"probe script missing: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run_probe(
    config: dict[str, Any], raw_json: Path, raw_csv: Path, model: str = "timesfm"
) -> None:
    """Run the probe end to end, writing its raw JSON/CSV exactly as it does."""
    ckpt = checkpoint_path(model)
    if not ckpt.exists():
        raise FileNotFoundError(f"missing {PROBES[model].display} checkpoint: {ckpt}")
    probe_args = [
        "--ckpt", str(ckpt),
        "--device", "cuda",
        "--context-length", str(config["context_length"]),
        "--horizon", str(config["horizon"]),
        "--evict-steps", str(config["evict_steps"]),
        "--seed", str(config["seed"]),
        "--dtype", config["dtype"],
        "--output-json", str(raw_json),
        "--output-csv", str(raw_csv),
    ]
    path = probe_path(model)
    try:
        probe = _load_probe_module(model)
    except Exception:
        traceback.print_exc()
        print("EXP4: probe import failed; falling back to subprocess CLI", flush=True)

        subprocess.run([sys.executable, str(path), *probe_args], check=True)
        return
    argv_backup = sys.argv
    sys.argv = [str(path), *probe_args]
    try:
        probe.main()
    finally:
        sys.argv = argv_backup


def reduce_rows(
    raw: dict[str, Any],
    scopes: tuple[str, ...] = SCOPES,
    n_layers: int = EXPECTED_LAYERS,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Filter the probe's per-step metrics down to the plot-contract rows."""
    warnings: list[str] = []
    step_by_age = {step.get("age"): step for step in raw.get("steps", [])}
    ages = [age for age in K12_AGES if age in step_by_age]
    absent = [age for age in K12_AGES if age not in step_by_age]
    if absent:
        print(
            f"EXP4 note: ages {absent} not in probe output "
            f"(evict_steps={raw.get('config', {}).get('evict_steps')}); "
            f"emitting {ages}",
            flush=True,
        )
    if not ages:
        warnings.append("no K12 grid age present in probe output")

    missing_scopes: set[str] = set()
    missing_tensors: set[str] = set()
    missing_fields: set[str] = set()
    rows: list[dict[str, Any]] = []
    for age in ages:
        layers = step_by_age[age].get("layers", [])
        if len(layers) != n_layers:
            warnings.append(
                f"age {age}: expected {n_layers} layers, probe emitted {len(layers)}"
            )
        for layer_entry in layers:
            layer_scopes = layer_entry.get("scopes", {})
            for scope in scopes:
                if scope not in layer_scopes:
                    missing_scopes.add(scope)
                    continue
                for tensor in TENSORS:
                    metrics = layer_scopes[scope].get(tensor)
                    if not isinstance(metrics, dict):
                        missing_tensors.add(tensor)
                        continue
                    row: dict[str, Any] = {
                        "age": age,
                        "layer": layer_entry.get("layer"),
                        "scope": scope,
                        "tensor": tensor,
                    }
                    for field in METRIC_FIELDS:
                        if field in metrics:
                            row[field] = metrics[field]
                        else:
                            # Not derivable from the probe's raw JSON (it stores
                            # metrics, not tensors); never fabricate a number.
                            missing_fields.add(field)
                    rows.append(row)
    if missing_scopes:
        warnings.append(
            "probe output lacks scope(s) "
            f"{sorted(missing_scopes)}; they are not recomputable from the raw "
            "JSON (tensors are not stored), so only the available scopes were emitted"
        )
    if missing_tensors:
        warnings.append(f"probe output lacks tensor(s) {sorted(missing_tensors)}")
    if missing_fields:
        warnings.append(
            f"probe output lacks metric column(s) {sorted(missing_fields)}; "
            "left blank in reduced_k12.csv"
        )
    return rows, warnings


def write_reduced_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REDUCED_FIELDS, restval="")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def build_summary(
    raw: dict[str, Any], config: dict[str, Any], warnings: list[str]
) -> dict[str, Any]:
    import platform

    import torch

    environment = dict(raw.get("environment", {}))
    environment.setdefault("torch", torch.__version__)
    environment["cuda"] = torch.version.cuda
    environment["gpu_name"] = (
        torch.cuda.get_device_name(0)
        if torch.cuda.is_available()
        else environment.get("device_name")
    )
    environment["platform"] = platform.platform()

    steps = []
    for step in raw.get("steps", []):
        forecast = dict(step.get("forecast", {}))
        if "mse" not in forecast and isinstance(forecast.get("rmse"), (int, float)):
            forecast["mse"] = float(forecast["rmse"]) ** 2
        steps.append(
            {
                "age": step.get("age"),
                "rolling_position_first": step.get("rolling_position_first"),
                "rolling_position_last": step.get("rolling_position_last"),
                "forecast": forecast,
                "normalization": step.get("normalization"),
            }
        )

    metric_definitions = dict(raw.get("metric_definitions", {}))
    metric_definitions.setdefault(
        "mse", "mean((rolling - full)^2), derived exactly as rmse**2"
    )
    return {
        "model": raw.get("model"),
        "checkpoint": raw.get("checkpoint"),
        "config": {**raw.get("config", {}), "smoke": config["smoke"]},
        "environment": environment,
        "protocol": raw.get("protocol"),
        "metric_definitions": metric_definitions,
        "rebase_self_check_max_abs": raw.get("rebase_self_check_max_abs"),
        "reduction": {
            "k12_ages": list(K12_AGES),
            "scopes": list(SCOPES),
            "tensors": list(TENSORS),
            "warnings": warnings,
        },
        "steps": steps,
    }


def build_paper_summary(raw: dict[str, Any]) -> dict[str, Any]:
    """summary_per_age.json with the paper file's exact top-level keys.

    The paper's TimesFM-2.5 ``summary_per_age.json`` is the probe's JSON minus
    the per-layer blocks: top-level model / checkpoint / config / environment /
    protocol / metric_definitions / rebase_self_check_max_abs / steps, and each
    step keeps age, rolling positions, forecast, normalization and the global
    (all-layer) metrics.  ``forecast`` additionally carries ``mse``, the value
    the heatmap prints (plot.py derives it as rmse**2 when absent).
    """
    steps = []
    for step in raw.get("steps", []):
        entry = {key: step.get(key) for key in PAPER_STEP_KEYS}
        forecast = dict(entry.get("forecast") or {})
        if "mse" not in forecast and isinstance(forecast.get("rmse"), (int, float)):
            forecast["mse"] = float(forecast["rmse"]) ** 2
        entry["forecast"] = forecast
        steps.append(entry)
    metric_definitions = dict(raw.get("metric_definitions", {}))
    metric_definitions.setdefault(
        "mse", "mean((rolling - full)^2), derived exactly as rmse**2"
    )
    return {
        "model": raw.get("model"),
        "checkpoint": raw.get("checkpoint"),
        "config": dict(raw.get("config", {})),
        "environment": dict(raw.get("environment", {})),
        "protocol": raw.get("protocol"),
        "metric_definitions": metric_definitions,
        "rebase_self_check_max_abs": raw.get("rebase_self_check_max_abs"),
        "steps": steps,
    }


def write_json_compact_atomic(path: Path, value: dict[str, Any]) -> None:
    """Single-line JSON with key order preserved (like the paper file), atomic."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(value), encoding="utf-8")
    os.replace(temp, path)


def probe_diagnostics(raw: dict[str, Any]) -> dict[str, Any]:
    """Provenance numbers for records.jsonl (k=0 agreement, reference checks)."""
    steps = raw.get("steps", [])
    out: dict[str, Any] = {}
    if steps:
        first = steps[0]
        out["k0_max_abs_kv"] = max(
            metrics["max_abs"]
            for scope in first.get("global", {}).values()
            for metrics in scope.values()
        )
        out["k0_forecast_rel_l2"] = first.get("forecast", {}).get("rel_l2")
        checks = [s.get("full_reference_check") for s in steps if s.get("full_reference_check")]
        if checks:
            worst: dict[str, float] = {}
            for check in checks:
                for key, value in check.items():
                    if isinstance(value, (int, float)):
                        worst[key] = max(worst.get(key, float("-inf")), float(value))
            out["full_reference_check_worst"] = worst
    for key in ("rebase_self_check_at_probe_shift", "runtime"):
        if key in raw:
            out[key] = raw[key]
    return out


def _repo_sha(repo: str) -> str:
    from shared.common import _pinned_upstream_sha, deployment_sha, run_output

    try:
        return run_output(["git", "-C", str(MODELS_ROOT / repo), "rev-parse", "HEAD"])
    except (OSError, subprocess.CalledProcessError):
        return _pinned_upstream_sha(repo) or deployment_sha()


def record_base(model: str) -> dict[str, Any]:
    """``base_record`` for registered models; same fields for a probe-only model."""
    if model in MODELS:
        return base_record("E", "EXP4", model)
    sha = _repo_sha(PROBES[model].repo)
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    return {
        "schema": "E",
        "run_id": f"EXP4-{model}-{day}-{sha[:7]}",
        "model": model,
        "git_sha": sha,
        "ts": datetime.now(timezone.utc).isoformat(),
        "status": "ok",
        "reason": None,
    }


def append_record(
    records_path: Path,
    config: dict[str, Any],
    status: str,
    reason: str | None,
    extra: dict[str, Any] | None = None,
    model: str = "timesfm",
) -> None:
    row = base_record("E", "EXP4", "timesfm") if model == "timesfm" else record_base(model)
    row.update({"status": status, "reason": reason, "config": config})
    if extra:
        row.update(extra)
    append_jsonl(records_path, row)


def main() -> int:
    args = parse_args()
    model = args.model
    probe_spec = PROBES[model]
    config = requested_config(args)
    out_root = output_root(model, args.smoke)
    records_path = out_root / "records.jsonl"
    reduced_path = out_root / "reduced_k12.csv"
    summary_path = out_root / "summary_per_age.json"
    raw_json = out_root / "raw" / "kv_intermediates.json"
    raw_csv = out_root / "raw" / "kv_intermediates.csv"

    if already_complete(out_root, config):
        if not has_completion_record(records_path, config):
            # Outputs can predate the queue (e.g. copied results); backfill the
            # completion record so launch_all can see this cell as done.
            append_record(
                records_path, config, "ok", "backfilled_on_idempotent_skip", model=model
            )
        print("EXP4 already complete")
        return 0

    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "EXP4 requires a CUDA GPU; torch.cuda.is_available() is false"
            )
        run_probe(config, raw_json, raw_csv, model)
        raw = json.loads(raw_json.read_text(encoding="utf-8"))
        n_layers = expected_layers(model)
        rows, warnings = reduce_rows(raw, probe_spec.scopes, n_layers)
        write_reduced_csv(reduced_path, rows)
        if probe_spec.paper_summary:
            write_json_compact_atomic(summary_path, build_paper_summary(raw))
        else:
            summary = build_summary(raw, config, warnings)
            write_json_atomic(summary_path, summary)
        for message in warnings:
            print(f"EXP4 WARNING: {message}", flush=True)
        extra: dict[str, Any] = {
            "rebase_self_check_max_abs": raw.get("rebase_self_check_max_abs"),
            "n_reduced_rows": len(rows),
            "ages_emitted": sorted({row["age"] for row in rows}),
            "warnings": warnings,
            "outputs": {
                "raw_json": str(raw_json),
                "raw_csv": str(raw_csv),
                "reduced_k12": str(reduced_path),
                "summary_per_age": str(summary_path),
            },
        }
        if model != "timesfm":
            extra["display_name"] = probe_spec.display
            extra["expected_layers"] = n_layers
            extra["reduction"] = {
                "k12_ages": list(K12_AGES),
                "scopes": list(probe_spec.scopes),
                "tensors": list(TENSORS),
            }
            extra.update(probe_diagnostics(raw))
        append_record(records_path, config, "ok", None, extra, model=model)
        print(
            f"EXP4 done: {len(rows)} reduced rows -> {reduced_path}",
            flush=True,
        )
        return 0
    except BaseException as exc:  # noqa: BLE001 - record SystemExit from the probe too
        if isinstance(exc, KeyboardInterrupt):
            raise
        traceback.print_exc()
        append_record(
            records_path,
            config,
            classify_failure(exc if isinstance(exc, Exception) else RuntimeError(str(exc))),
            f"{type(exc).__name__}: {str(exc)[:1200]}",
            model=model,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
