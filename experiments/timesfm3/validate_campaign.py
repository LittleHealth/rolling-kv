"""Acceptance checks for the TimesFM-3.0 campaign (locked v2 protocol).

The expected grid comes from ``launch_timesfm3.plan()``; this module only
enumerates and checks.  Results root: ``--results`` or, by default,
``$ROLLKV_ROOT/results_tfm3`` (an inherited ``ROLLKV_RESULTS`` is ignored,
because env.sh points it at the TimesFM-2.5 tree).

Hard failures (exit 1):
  * manifest windows differ from the v2 windows; unsupported cells differ
    from the feasibility rule / the 15 expected cells;
  * any EXP-0 gate not passed at any of the six contexts; T3 (graph vs eager)
    and T6 (refresh reset) not exactly 0; a T5 age missing;
  * any planned quality cell missing or not ``ok`` (``unsupported`` only for
    the 15 expected (dataset, window, L=16384) cells);
  * any planned timing (L, K) pair missing or not ``ok``;
  * record / timing / kernel rows whose key set differs from the v2 schema;
  * |D| > G on any ok record (the paper's corollary |D| <= G, checked with a
    1e-4 percentage-point float tolerance);
  * a prediction file missing, malformed, inconsistent with its record, or
    whose recomputed metrics disagree with the record;
  * EXP-2 kernel records not exactly the 14 planned ok rows;
  * latency phases recorded as run on a non-SXM4 GPU.
Warnings: timing_flag=inconsistent counts, K=1 baseline jumps between
adjacent contexts, quality/timing speedup drift, provenance.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # experiments/ root

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent


def _bootstrap_results(argv: list[str] | None = None) -> Path:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--results")
    known, _ = pre.parse_known_args(argv)
    if known.results:
        results = Path(known.results).resolve()
    else:
        root = os.environ.get("ROLLKV_ROOT")
        results = (Path(root) if root else HERE.parents[1]) / "results_tfm3"
    os.environ["ROLLKV_RESULTS"] = str(results)
    return results


if __name__ == "__main__":
    _bootstrap_results()

import numpy as np  # noqa: E402

from shared.common import (  # noqa: E402
    MODELS,
    RESULTS,
    TFM3_MODEL,
    TFM3_V2_WINDOWS,
    _pinned_upstream_sha,
    read_jsonl,
    tfm3_unsupported_cells,
    utc_now,
    write_json_atomic,
)
from timesfm3.launch_campaign import (  # noqa: E402
    expected_unsupported,
    kernel_cells,
    plan,
    quality_cells,
    timing_pairs,
)


MODEL = TFM3_MODEL
SPEC = MODELS[MODEL]
LIST_LIMIT = 20
D_LE_G_TOL_PP = 1e-4
METRIC_RTOL = 1e-5
METRIC_ATOL = 1e-7
T7_JUMP_FACTOR = 2.0
REQUIRED_GATES = ("T1", "T2", "T3", "T4", "T6")
EXACT_GATES = ("T3", "T6")

# Key sets of the v2 TimesFM-2.5 artifacts the paper was built from.
V2_RECORD_KEYS = frozenset(
    "H K L U adaptive_params dataset exp gap_pct git_sha mae_delta_h32_pct "
    "mae_delta_pct mae_h32 mae_native mae_ref_h32 mae_ref_native model "
    "mse_delta_pct mse_h32 mse_native n_full n_roll norm_mode policy policy_id "
    "pos_remap preds_file reason run_id s schema speedup status tau_pts "
    "timing_ref ts window window_start".split()
)
V2_TIMING_KEYS = frozenset(
    "K L U batch exec git_sha graph_capture_ok kv_cache_mb model n_full "
    "n_kernels_full n_kernels_roll n_roll peak_mem_mb policy policy_id pos_remap "
    "reason run_id s schema speedup status t_full_ms t_roll_ms "
    "t_update_computed_ms t_update_measured_ms timing_flag timing_id "
    "trigger_overhead_ms ts us_per_point".split()
)
V2_KERNEL_KEYS = frozenset(
    "L batch git_sha graph_wall_ms kernel_busy_ms kernel_counts method model "
    "path reason run_id schema status ts".split()
)
KERNEL_CLASSES = frozenset(
    ("matmul", "attention", "elementwise", "norm", "index_mem", "other")
)


class Report:
    def __init__(self) -> None:
        self.hard: list[str] = []
        self.warnings: list[str] = []
        self.counts: dict[str, Any] = {}

    def fail(self, message: str) -> None:
        self.hard.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def fail_list(self, what: str, items: list[str]) -> None:
        if items:
            shown = "; ".join(items[:LIST_LIMIT])
            self.fail(f"{what} ({len(items)}): {shown}")

    def warn_list(self, what: str, items: list[str]) -> None:
        if items:
            shown = "; ".join(items[:LIST_LIMIT])
            self.warn(f"{what} ({len(items)}): {shown}")


def _latest(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[tuple[Any, ...], dict[str, Any]]:
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        result[tuple(row.get(field) for field in fields)] = row
    return result


def _read(report: Report, path: Path, label: str) -> list[dict[str, Any]]:
    try:
        rows = read_jsonl(path)
    except ValueError as exc:
        report.fail(f"{label} unreadable: {exc}")
        return []
    if not rows:
        report.fail(f"{label} missing or empty: {path}")
    return rows


def _close(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return abs(float(a) - float(b)) <= METRIC_ATOL + METRIC_RTOL * abs(float(b))


# ---------------------------------------------------------------------------


def check_manifest(report: Report, grid: dict[str, Any]) -> dict[str, Any]:
    path = RESULTS / "manifest.json"
    if not path.exists():
        report.fail(f"manifest.json missing: {path}")
        return {}
    manifest = json.loads(path.read_text(encoding="utf-8"))
    windows = {name: list(values) for name, values in manifest.get("windows", {}).items()}
    want = {name: list(values) for name, values in TFM3_V2_WINDOWS.items()}
    if windows != want:
        report.fail("manifest windows differ from the locked v2 windows")
    if not manifest.get("windows_redraw_matches_v2", False):
        report.warn(
            "choose_windows() does not redraw the v2 windows on this data: "
            f"{json.dumps(manifest.get('windows_redraw_mismatch'))}"
        )
    lengths = manifest.get("series_lengths", {})
    try:
        pairs = {(item["dataset"], int(item["L"])) for item in grid["quality"]}
        rule = set(tfm3_unsupported_cells(want, lengths, sorted(pairs)))
    except (KeyError, TypeError, ValueError) as exc:
        report.fail(f"cannot apply the feasibility rule: {type(exc).__name__}: {exc}")
        rule = set()
    expected = expected_unsupported(grid)
    if rule != expected:
        report.fail(
            f"feasibility rule gives {len(rule)} unsupported cells, plan expects "
            f"{len(expected)}: rule-only={sorted(rule - expected)[:LIST_LIMIT]} "
            f"plan-only={sorted(expected - rule)[:LIST_LIMIT]}"
        )
    recorded = {
        (item["dataset"], int(item["window"]), int(item["L"]))
        for item in manifest.get("unsupported_cells", [])
    }
    if recorded != expected:
        report.fail("manifest unsupported_cells differ from the plan's expected cells")
    report.counts["expected_unsupported_cells"] = len(expected)
    return manifest


def check_exp0(report: Report, grid: dict[str, Any]) -> None:
    path = RESULTS / "EXP0_correctness" / MODEL / "records.jsonl"
    rows = _read(report, path, "EXP0 records")
    latest = _latest(rows, ("L", "gate", "cache_age"))
    problems: list[str] = []
    summary: dict[str, dict[str, Any]] = {}
    for length in grid["exp0_lengths"]:
        per_l: dict[str, Any] = {}
        for gate in REQUIRED_GATES:
            row = latest.get((length, gate, None))
            if row is None:
                problems.append(f"L={length} {gate} missing")
                continue
            if row.get("status") != "ok" or row.get("passed") is not True:
                problems.append(
                    f"L={length} {gate} status={row.get('status')} passed={row.get('passed')}"
                )
            if gate in EXACT_GATES and row.get("max_abs_err") != 0.0:
                problems.append(f"L={length} {gate} max_abs_err={row.get('max_abs_err')} != 0")
            per_l[gate] = {"max_abs_err": row.get("max_abs_err"), "rel_err": row.get("rel_err")}
        for age in range(1, 9):
            row = latest.get((length, "T5", age))
            if row is None or row.get("status") != "ok":
                problems.append(f"L={length} T5 age={age} missing/not ok")
        summary[str(length)] = per_l
    report.fail_list("EXP0 gate problems", problems)
    report.counts["exp0_rows"] = len(rows)
    report.counts["exp0_summary"] = summary


def check_quality(
    report: Report, grid: dict[str, Any], manifest: dict[str, Any]
) -> list[dict[str, Any]]:
    path = RESULTS / "EXP1_sweep" / MODEL / "records.jsonl"
    rows = _read(report, path, "EXP1 records")
    latest = _latest(rows, ("dataset", "window", "L", "K", "pos_remap"))
    cells = quality_cells(grid)
    unsupported = expected_unsupported(grid)
    missing: list[str] = []
    bad_status: list[str] = []
    wrong_unsupported: list[str] = []
    ok_rows: list[dict[str, Any]] = []
    unsupported_seen: set[tuple[str, int, int]] = set()
    for dataset, window, length, k in sorted(cells):
        row = latest.get((dataset, window, length, k, "n/a"))
        label = f"{dataset}/w{window}/L{length}/K{k}"
        if row is None:
            missing.append(label)
            continue
        status = row.get("status")
        if (dataset, window, length) in unsupported:
            if status != "unsupported":
                wrong_unsupported.append(f"{label} status={status} (expected unsupported)")
            else:
                unsupported_seen.add((dataset, window, length))
        elif status != "ok":
            bad_status.append(f"{label} status={status} reason={str(row.get('reason'))[:120]}")
        else:
            ok_rows.append(row)
    report.fail_list("EXP1 planned quality cells missing", missing)
    report.fail_list("EXP1 quality cells not ok", bad_status)
    report.fail_list("EXP1 unsupported cells mis-recorded", wrong_unsupported)
    extra = [
        f"{key[0]}/w{key[1]}/L{key[2]}/K{key[3]} status={row.get('status')}"
        for key, row in latest.items()
        if (key[0], key[1], key[2], key[3]) not in cells
    ]
    report.warn_list("EXP1 rows outside the plan", extra)

    schema_bad: list[str] = []
    protocol_bad: list[str] = []
    for key, row in latest.items():
        if row.get("status") not in ("ok", "unsupported"):
            continue
        keys = set(row)
        if keys != V2_RECORD_KEYS:
            schema_bad.append(
                f"{key}: extra={sorted(keys - V2_RECORD_KEYS)} missing={sorted(V2_RECORD_KEYS - keys)}"
            )
        if (
            row.get("model") != MODEL
            or row.get("U") != SPEC.updates
            or row.get("H") != SPEC.horizon
            or row.get("s") != SPEC.s
            or row.get("pos_remap") != "n/a"
        ):
            protocol_bad.append(f"{key}: model/U/H/s/pos_remap off protocol")
        windows = manifest.get("windows", {})
        try:
            if int(row["window_start"]) != int(windows[row["dataset"]][row["window"]]):
                protocol_bad.append(f"{key}: window_start != manifest window")
        except (KeyError, TypeError, ValueError, IndexError):
            protocol_bad.append(f"{key}: window_start not checkable")
    report.fail_list("EXP1 record key set differs from v2 schema", schema_bad)
    report.fail_list("EXP1 records off protocol", protocol_bad)

    # |D| <= G, the paper's corollary, on every ok record.
    violations: list[str] = []
    max_excess = -math.inf
    for row in ok_rows:
        delta, gap = row.get("mae_delta_pct"), row.get("gap_pct")
        if delta is None or gap is None:
            violations.append(f"{row['dataset']}/w{row['window']}/L{row['L']}/K{row['K']} D or G null")
            continue
        excess = abs(float(delta)) - float(gap)
        max_excess = max(max_excess, excess)
        if excess > D_LE_G_TOL_PP:
            violations.append(
                f"{row['dataset']}/w{row['window']}/L{row['L']}/K{row['K']} "
                f"|D|={abs(float(delta)):.6f} G={float(gap):.6f}"
            )
    report.fail_list(f"|D| <= G violated (tolerance {D_LE_G_TOL_PP} pp)", violations)
    report.counts["quality_expected"] = len(cells)
    report.counts["quality_ok"] = len(ok_rows)
    report.counts["quality_missing"] = len(missing)
    report.counts["unsupported_cells_seen"] = len(unsupported_seen)
    report.counts["d_minus_g_max_pp"] = None if max_excess == -math.inf else max_excess
    return ok_rows


def check_timing(report: Report, grid: dict[str, Any], ok_quality: list[dict[str, Any]]) -> None:
    path = RESULTS / "EXP1_sweep" / MODEL / "timing.jsonl"
    rows = _read(report, path, "EXP1 timing")
    latest: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        if row.get("exec") != "graph" or row.get("batch") != 1 or row.get("pos_remap") != "n/a":
            continue
        if row.get("L") is None or row.get("K") is None:
            continue
        latest[(int(row["L"]), int(row["K"]))] = row
    pairs = timing_pairs(grid)
    missing, bad_status, schema_bad = [], [], []
    inconsistent = 0
    for pair in sorted(pairs):
        row = latest.get(pair)
        if row is None:
            missing.append(f"L{pair[0]}/K{pair[1]}")
            continue
        if row.get("status") != "ok":
            bad_status.append(f"L{pair[0]}/K{pair[1]} status={row.get('status')}")
            continue
        if set(row) != V2_TIMING_KEYS:
            schema_bad.append(
                f"L{pair[0]}/K{pair[1]}: extra={sorted(set(row) - V2_TIMING_KEYS)} "
                f"missing={sorted(V2_TIMING_KEYS - set(row))}"
            )
        if row.get("timing_flag") == "inconsistent":
            inconsistent += 1
        if pair[1] == 1 and row.get("speedup") != 1.0:
            bad_status.append(f"L{pair[0]}/K1 speedup={row.get('speedup')} != 1")
    report.fail_list("timing (L, K) pairs missing", missing)
    report.fail_list("timing pairs not ok", bad_status)
    report.fail_list("timing key set differs from v2 schema", schema_bad)
    extra = [f"L{L}/K{K}" for (L, K) in latest if (L, K) not in pairs]
    report.warn_list("timing rows outside the plan", extra)
    report.counts["timing_expected"] = len(pairs)
    report.counts["timing_ok"] = sum(
        1 for pair in pairs if latest.get(pair, {}).get("status") == "ok"
    )
    report.counts["timing_inconsistent"] = inconsistent
    if inconsistent:
        report.warn(f"{inconsistent} timing rows flagged inconsistent (|computed-measured|>10%)")

    drift = []
    for row in ok_quality:
        timing = latest.get((int(row["L"]), int(row["K"])))
        if timing is None or row.get("K") == 1:
            continue
        if not _close(row.get("speedup"), timing.get("speedup")):
            drift.append(
                f"{row['dataset']}/w{row['window']}/L{row['L']}/K{row['K']} "
                f"record {row.get('speedup')} vs timing {timing.get('speedup')}"
            )
    report.warn_list("quality speedup differs from the latest timing row", drift)


def check_npz(report: Report, ok_rows: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    U, H, s = SPEC.updates, SPEC.horizon, SPEC.s
    reference: dict[tuple[str, int, int], np.ndarray] = {}
    problems: list[str] = []
    loaded: dict[tuple[str, int, int, int], dict[str, np.ndarray]] = {}
    for row in sorted(ok_rows, key=lambda item: (item["dataset"], item["window"], item["L"], item["K"])):
        key = (row["dataset"], row["window"], row["L"], row["K"])
        label = f"{row['dataset']}/w{row['window']}/L{row['L']}/K{row['K']}"
        path = RESULTS / str(row.get("preds_file"))
        try:
            with np.load(path, allow_pickle=False) as archive:
                data = {name: archive[name].copy() for name in ("yhat", "y", "t_index", "meta")}
        except Exception as exc:
            problems.append(f"{label}: unreadable {path} ({type(exc).__name__}: {exc})")
            continue
        yhat, y, t_index = data["yhat"], data["y"], data["t_index"]
        if yhat.shape != (U, H) or y.shape != (U, H) or t_index.shape != (U,):
            problems.append(f"{label}: shapes yhat{yhat.shape} y{y.shape} t{t_index.shape}")
            continue
        if yhat.dtype != np.float32 or y.dtype != np.float32 or t_index.dtype != np.int64:
            problems.append(f"{label}: dtypes {yhat.dtype}/{y.dtype}/{t_index.dtype}")
        if not (np.isfinite(yhat).all() and np.isfinite(y).all()):
            problems.append(f"{label}: non-finite values")
            continue
        start = int(row["window_start"])
        if not np.array_equal(t_index, start + s * np.arange(1, U + 1, dtype=np.int64)):
            problems.append(f"{label}: t_index does not match window_start")
        try:
            meta = json.loads(str(data["meta"]))
        except ValueError:
            problems.append(f"{label}: meta is not JSON")
            meta = {}
        for field in ("dataset", "window", "L", "K", "model"):
            if meta.get(field) != row.get(field):
                problems.append(f"{label}: meta {field}={meta.get(field)!r} != record {row.get(field)!r}")
                break
        mae = float(np.abs(yhat - y).mean())
        mse = float(np.square(yhat - y).mean())
        mse32 = float(np.square(yhat[:, :32] - y[:, :32]).mean())
        mae32 = float(np.abs(yhat[:, :32] - y[:, :32]).mean())
        for name, value in (("mae_native", mae), ("mse_native", mse), ("mse_h32", mse32), ("mae_h32", mae32)):
            if not _close(value, row.get(name)):
                problems.append(f"{label}: recomputed {name} {value:.8g} != record {row.get(name)}")
        loaded[key] = {"yhat": yhat, "y": y}
        if row["K"] == 1:
            reference[(row["dataset"], row["window"], row["L"])] = yhat
    by_key = {(row["dataset"], row["window"], row["L"], row["K"]): row for row in ok_rows}
    for key, data in loaded.items():
        dataset, window, length, k = key
        ref = reference.get((dataset, window, length))
        label = f"{dataset}/w{window}/L{length}/K{k}"
        if ref is None:
            problems.append(f"{label}: no K=1 reference prediction file")
            continue
        # Same float32 reductions as common.quality_metrics on the same arrays.
        ref_mae = float(np.abs(ref - data["y"]).mean())
        gap = float(np.abs(data["yhat"] - ref).mean() / max(ref_mae, 1e-12) * 100.0)
        if not _close(gap, by_key[key].get("gap_pct")):
            problems.append(f"{label}: recomputed gap_pct {gap:.6g} != record {by_key[key].get('gap_pct')}")
    report.fail_list("prediction file problems", problems)
    report.counts["npz_checked"] = len(loaded)


def check_kernels(report: Report, grid: dict[str, Any]) -> None:
    path = RESULTS / "EXP2_stages" / MODEL / "kernels.jsonl"
    rows = _read(report, path, "EXP2 kernels")
    latest = _latest(
        [row for row in rows if row.get("method") == "profiler_eager_kernel_classes"],
        ("L", "batch", "path"),
    )
    cells = kernel_cells(grid)
    problems: list[str] = []
    ok = 0
    for key in sorted(cells):
        row = latest.get(key)
        label = f"L={key[0]} batch={key[1]} {key[2]}"
        if row is None:
            problems.append(f"{label} missing")
            continue
        if row.get("status") != "ok":
            problems.append(f"{label} status={row.get('status')}")
            continue
        if set(row) != V2_KERNEL_KEYS:
            problems.append(
                f"{label} keys extra={sorted(set(row) - V2_KERNEL_KEYS)} "
                f"missing={sorted(V2_KERNEL_KEYS - set(row))}"
            )
        classes = set(row.get("kernel_busy_ms") or {}) | set(row.get("kernel_counts") or {})
        if not classes or not classes <= KERNEL_CLASSES:
            problems.append(f"{label} classes {sorted(classes)}")
        if not (row.get("graph_wall_ms") or 0) > 0:
            problems.append(f"{label} graph_wall_ms={row.get('graph_wall_ms')}")
        ok += 1
    extra = [f"L={k[0]} batch={k[1]} {k[2]}" for k in latest if k not in cells]
    report.fail_list("EXP2 kernel records", problems)
    report.warn_list("EXP2 kernel rows outside the plan", extra)
    report.counts["kernels_expected"] = len(cells)
    report.counts["kernels_ok"] = ok


def check_hardware(report: Report, manifest: dict[str, Any]) -> None:
    phases = manifest.get("phases", {}) or {}
    seen = False
    for name, entry in phases.items():
        if not (name.startswith("P-timing") or name.startswith("P-S2")):
            continue
        detail = (entry or {}).get("detail") or {}
        gpu_name = detail.get("gpu_name")
        if gpu_name is None:
            continue
        seen = True
        if "SXM4" not in str(gpu_name):
            report.fail(f"{name} ran on {gpu_name!r}, not an A100-SXM4 (paper hardware)")
    if not seen:
        report.warn("no latency phase GPU recorded in manifest phases")


def check_provenance(report: Report) -> None:
    pinned = _pinned_upstream_sha(SPEC.repo)
    shas: set[str] = set()
    for rel in (
        ("EXP0_correctness", MODEL, "records.jsonl"),
        ("EXP1_sweep", MODEL, "records.jsonl"),
        ("EXP1_sweep", MODEL, "timing.jsonl"),
        ("EXP2_stages", MODEL, "kernels.jsonl"),
    ):
        try:
            shas |= {str(row.get("git_sha")) for row in read_jsonl(RESULTS.joinpath(*rel))}
        except ValueError:
            continue
    report.counts["git_shas"] = sorted(shas)
    if pinned and shas and shas != {pinned}:
        report.warn(f"git_sha values {sorted(shas)} differ from the TimesFM-3.0 pin {pinned}")


def check_baseline(report: Report, ok_rows: list[dict[str, Any]]) -> None:
    per_cell: dict[tuple[str, int], list[float]] = {}
    for row in ok_rows:
        if row.get("K") == 1 and row.get("mse_h32") is not None:
            per_cell.setdefault((row["dataset"], int(row["L"])), []).append(float(row["mse_h32"]))
    medians: dict[str, dict[int, float]] = {}
    for (dataset, length), values in per_cell.items():
        medians.setdefault(dataset, {})[length] = float(np.median(values))
    offenders = []
    for dataset, by_l in sorted(medians.items()):
        lengths = sorted(by_l)
        for low, high in zip(lengths, lengths[1:]):
            a, b = by_l[low], by_l[high]
            floor = min(a, b)
            ratio = max(a, b) / floor if floor > 0 else float("inf")
            if ratio > T7_JUMP_FACTOR:
                offenders.append(f"{dataset} L{low}->L{high} median mse_h32 {a:.6g}->{b:.6g} ({ratio:.2f}x)")
    report.warn_list(f"K=1 median mse_h32 jumps >{T7_JUMP_FACTOR:g}x between adjacent contexts", offenders)


def validate() -> dict[str, Any]:
    report = Report()
    grid = plan()

    def guarded(name: str, fn):
        try:
            return fn()
        except Exception as exc:  # a checker bug must not hide the others
            report.fail(f"{name} check crashed: {type(exc).__name__}: {exc}")
            return None

    manifest = guarded("manifest", lambda: check_manifest(report, grid)) or {}
    guarded("EXP0", lambda: check_exp0(report, grid))
    ok_rows = guarded("EXP1 quality", lambda: check_quality(report, grid, manifest)) or []
    guarded("EXP1 timing", lambda: check_timing(report, grid, ok_rows))
    guarded("prediction files", lambda: check_npz(report, ok_rows, manifest))
    guarded("EXP2 kernels", lambda: check_kernels(report, grid))
    guarded("hardware", lambda: check_hardware(report, manifest))
    guarded("provenance", lambda: check_provenance(report))
    guarded("baseline", lambda: check_baseline(report, ok_rows))
    result = {
        "schema": "rolling-kv-validation-timesfm3",
        "model": MODEL,
        "results": str(RESULTS),
        "ts": utc_now(),
        "passed": not report.hard,
        "hard_failures": report.hard,
        "warnings": report.warnings,
        "counts": report.counts,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    write_json_atomic(RESULTS / "validation_timesfm3.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", help="results root (default: $ROLLKV_ROOT/results_tfm3)")
    parser.parse_args()
    result = validate()
    verdict = "PASS" if result["passed"] else "FAIL"
    print(
        f"validate_timesfm3: {verdict} ({len(result['hard_failures'])} hard failures, "
        f"{len(result['warnings'])} warnings) results={RESULTS}"
    )
    for line in result["hard_failures"]:
        print(f"  HARD: {line[:1500]}")
    for line in result["warnings"]:
        print(f"  WARN: {line[:1500]}")
    counts = {key: value for key, value in result["counts"].items() if key != "exp0_summary"}
    print("  counts: " + json.dumps(counts, sort_keys=True))
    print(f"  wrote {RESULTS / 'validation_timesfm3.json'}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
