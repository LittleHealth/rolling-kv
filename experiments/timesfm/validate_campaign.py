"""Acceptance checks over the finished TimesFM-2.5 campaign artifacts.

No CLI arguments: all paths are env-driven through ``common`` (``ROLLKV_RESULTS``).
The expected grid comes from ``launch_timesfm.plan()`` so the launcher stays the
single source of truth; this module only enumerates and checks.  Hard failures
(missing or failed cells, broken artifacts) exit 1; statistical sanity issues
are recorded as warnings and never change the exit code.
"""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # experiments/ root

import csv
import importlib
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from shared.common import (
    DATASETS,
    EXT_K_VALUES,
    EXT_UPDATES,
    MODELS,
    RESULTS,
    SEED_BASE,
    read_jsonl,
    utc_now,
    write_json_atomic,
)


SPEC = MODELS["timesfm"]
TREES = ("EXP1_sweep", "EXP1_ext")
TREE_ARMS = (("EXP1_sweep", "main"), ("EXP1_ext", "ext"))
EXP4_DIR = RESULTS / "EXP4_kvdiag"
EXP4_HEADER = (
    "age", "layer", "scope", "tensor", "mae", "rmse",
    "max_abs", "rel_l2", "normalized_mae", "cosine_distance",
)
EXP4_MIN_AGES = 12
EXP4_LAYERS = 20
GAP_TOLERANCE_PP = 1.0  # |D| <= G checked in percentage points with this slack
T7_JUMP_FACTOR = 2.0
NPZ_SAMPLE = 5
LIST_LIMIT = 20


# ---------------------------------------------------------------------------
# Expected-grid enumeration: pure functions over launch_timesfm.plan().

_MAIN_ARM_KEYS = ("main", "sweep", "exp1_sweep", "b0_s1", "base", "quality")
_EXT_ARM_KEYS = ("ext", "exp1_ext", "extension", "s4", "staleness_ext", "windows_ext")


def _is_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _int_seq(value: Any) -> tuple[int, ...] | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = list(value)
        if items and all(_is_int(item) for item in items):
            return tuple(int(item) for item in items)
    return None


def _first_key(mapping: Mapping[Any, Any], names: Sequence[str]) -> Any:
    lowered = {str(key).lower(): value for key, value in mapping.items()}
    for name in names:
        if name in lowered:
            return lowered[name]
    return None


def _dataset_length_pairs(arm: Any) -> list[tuple[str, int]]:
    """All (dataset, L) pairs described by one arm of the plan."""
    datasets = None
    if isinstance(arm, Mapping):
        datasets = _first_key(arm, ("datasets", "grid", "dataset_lengths"))
        if datasets is None:
            if arm and all(str(key) in DATASETS for key in arm):
                datasets = arm  # the arm itself maps dataset -> lengths
            else:
                raise ValueError(
                    "plan arm has no recognizable datasets: "
                    f"keys={sorted(map(str, arm))!r}"
                )
    else:
        datasets = arm
    if isinstance(datasets, Mapping):
        pairs: list[tuple[str, int]] = []
        for name, value in datasets.items():
            lengths = _int_seq(value)
            if lengths is None and isinstance(value, Mapping):
                lengths = _int_seq(_first_key(value, ("lengths", "l", "l_values")))
            if lengths is None:
                raise ValueError(f"plan arm has no lengths for dataset {name!r}")
            pairs.extend((str(name), length) for length in lengths)
        return pairs
    if isinstance(datasets, Sequence) and not isinstance(datasets, (str, bytes)):
        items = list(datasets)
        if items and all(isinstance(item, str) for item in items):
            lengths = None
            if isinstance(arm, Mapping):
                lengths = _int_seq(_first_key(arm, ("lengths", "l", "l_values")))
            if lengths is None:
                raise ValueError("plan arm lists dataset names but no lengths")
            return [(name, length) for name in items for length in lengths]
        pairs = []
        for item in items:
            if isinstance(item, Mapping):
                name = _first_key(item, ("dataset", "name"))
                raw = _first_key(item, ("l", "length", "context", "context_length"))
                lengths = _int_seq(raw) or (
                    (int(raw),) if _is_int(raw) else None
                )
                if name is None or lengths is None:
                    raise ValueError(f"plan arm cell not understood: {item!r}")
                pairs.extend((str(name), length) for length in lengths)
                continue
            if (
                isinstance(item, Sequence)
                and not isinstance(item, (str, bytes))
                and len(item) >= 2
                and isinstance(item[0], str)
                and _is_int(item[1])
            ):
                pairs.append((item[0], int(item[1])))
                continue
            raise ValueError(f"plan arm cell not understood: {item!r}")
        return pairs
    raise ValueError(f"plan arm shape not understood: {type(arm).__name__}")


def _indices_from_seq(seq: tuple[int, ...]) -> tuple[int, ...]:
    # Small values are window indices; large values are manifest-style window
    # start points (endpoints deep inside the series), so only their count counts.
    if all(0 <= item < 100 for item in seq):
        return seq
    return tuple(range(len(seq)))


def _window_indices(arm: Any, dataset: str, default_count: int = 5) -> tuple[int, ...]:
    spec = None
    if isinstance(arm, Mapping):
        spec = _first_key(arm, ("windows", "n_windows", "window_count", "window_indices"))
    if spec is None:
        return tuple(range(default_count))
    if isinstance(spec, Mapping):
        value = spec.get(dataset)
        if value is None:
            return tuple(range(default_count))
        if _is_int(value):
            return tuple(range(int(value)))
        seq = _int_seq(value)
        if seq is not None:
            return _indices_from_seq(seq)
        raise ValueError(f"plan windows for dataset {dataset!r} not understood")
    if _is_int(spec):
        return tuple(range(int(spec)))
    seq = _int_seq(spec)
    if seq is not None:
        return _indices_from_seq(seq)
    raise ValueError("plan windows not understood")


def _k_values(arm: Any, default: Sequence[int]) -> tuple[int, ...]:
    if isinstance(arm, Mapping):
        found = _int_seq(_first_key(arm, ("k_values", "ks", "k", "k_grid")))
        if found is not None:
            return found
    return tuple(int(k) for k in default)


def split_plan_arms(plan_result: Any) -> tuple[Any, Any]:
    """Return the (main, ext) arm specs of a launch_timesfm.plan() result."""
    if isinstance(plan_result, Mapping):
        main = _first_key(plan_result, _MAIN_ARM_KEYS)
        ext = _first_key(plan_result, _EXT_ARM_KEYS)
        if main is None and ext is not None:
            # Flat launch_timesfm.plan() shape: B0 datasets share the full
            # length grid, S1 carries its own dataset -> lengths mapping.
            b0 = _first_key(plan_result, ("b0_datasets",))
            s1 = _first_key(plan_result, ("s1",))
            lengths = _int_seq(_first_key(plan_result, ("lengths",)))
            if b0 is not None and lengths is not None:
                grid = {str(name): list(lengths) for name in b0}
                if isinstance(s1, Mapping):
                    for name, value in s1.items():
                        grid[str(name)] = list(_int_seq(value) or [])
                main = {
                    "datasets": grid,
                    "k_values": plan_result.get("k_values"),
                    "windows": plan_result.get("windows"),
                }
        if main is not None and ext is not None:
            return main, ext
        raise ValueError(
            "plan() mapping lacks recognizable main/ext arms: "
            f"keys={sorted(map(str, plan_result))!r}"
        )
    if (
        isinstance(plan_result, Sequence)
        and not isinstance(plan_result, (str, bytes))
        and len(plan_result) == 2
    ):
        return plan_result[0], plan_result[1]
    raise ValueError(
        f"plan() shape not understood: {type(plan_result).__name__}"
    )


def expected_quality_cells(
    arm: Any, default_k_values: Sequence[int], default_window_count: int = 5
) -> set[tuple[str, int, int, int]]:
    """Enumerate expected (dataset, L, K, window) cells for one plan arm."""
    cells: set[tuple[str, int, int, int]] = set()
    k_values = _k_values(arm, default_k_values)
    for dataset, length in _dataset_length_pairs(arm):
        for window in _window_indices(arm, dataset, default_window_count):
            for k in k_values:
                cells.add((dataset, int(length), int(k), int(window)))
    return cells


def expected_timing_pairs(
    cells: set[tuple[str, int, int, int]]
) -> set[tuple[int, int]]:
    return {(length, k) for _dataset, length, k, _window in cells}


def expected_cells(plan_result: Any) -> dict[str, dict[str, set]]:
    """Full expected grid derived from a launch_timesfm.plan() result."""
    main_arm, ext_arm = split_plan_arms(plan_result)
    main = expected_quality_cells(main_arm, SPEC.k_values)
    ext = expected_quality_cells(ext_arm, EXT_K_VALUES)
    return {
        "main": {"quality": main, "timing": expected_timing_pairs(main)},
        "ext": {"quality": ext, "timing": expected_timing_pairs(ext)},
    }


# ---------------------------------------------------------------------------
# Artifact readers.


def _latest(path: Path, fields: tuple[str, ...]) -> dict[tuple[Any, ...], dict[str, Any]]:
    result: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in read_jsonl(path):
        result[tuple(row.get(field) for field in fields)] = row
    return result


def _has_key(value: Any, key: str) -> bool:
    if isinstance(value, Mapping):
        if key in value:
            return True
        return any(_has_key(item, key) for item in value.values())
    if isinstance(value, list):
        return any(_has_key(item, key) for item in value)
    return False


def _find_exp4_file(name: str) -> Path | None:
    direct = EXP4_DIR / name
    if direct.exists():
        return direct
    if EXP4_DIR.exists():
        found = sorted(EXP4_DIR.rglob(name))
        if found:
            return found[0]
    return None


def _cell_text(cell: tuple[str, int, int, int]) -> str:
    dataset, length, k, window = cell
    return f"{dataset}/L{length}/K{k}/w{window}"


# ---------------------------------------------------------------------------
# Checks.


def validate(grids: dict[str, dict[str, set]] | None, plan_error: str | None) -> dict[str, Any]:
    hard: list[str] = []
    warnings: list[str] = []
    counts: dict[str, Any] = {}
    if plan_error:
        hard.append(plan_error)

    # 1. EXP-0 gates: every thresholded gate passed, all six lengths present.
    gate_path = RESULTS / "EXP0_correctness" / "timesfm" / "records.jsonl"
    try:
        gates = _latest(gate_path, ("L", "gate", "cache_age"))
    except ValueError as exc:
        hard.append(f"EXP0 records unreadable: {exc}")
        gates = {}
    if not gates:
        hard.append(f"EXP0 records missing or empty: {gate_path}")
    present_lengths = {key[0] for key in gates}
    for length in SPEC.lengths:
        if length not in present_lengths:
            hard.append(f"EXP0 gates missing for L={length}")
    gate_failures = 0
    for key, row in sorted(gates.items(), key=lambda item: repr(item[0])):
        if row.get("threshold") is None:
            continue
        if row.get("status") == "ok" and row.get("passed") is True:
            continue
        length, gate, age = key
        suffix = "" if age is None else f" age={age}"
        hard.append(
            f"EXP0 gate not passed: L={length} {gate}{suffix} "
            f"status={row.get('status')} passed={row.get('passed')}"
        )
        gate_failures += 1
    counts["exp0_rows"] = len(gates)
    counts["exp0_gate_failures"] = gate_failures

    # Quality rows for both trees (latest row per cell, append-only logs).
    quality_latest: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = {}
    ok_quality: dict[str, list[dict[str, Any]]] = {}
    for tree in TREES:
        path = RESULTS / tree / "timesfm" / "records.jsonl"
        try:
            quality_latest[tree] = _latest(path, ("dataset", "window", "L", "K"))
        except ValueError as exc:
            hard.append(f"{tree} records unreadable: {exc}")
            quality_latest[tree] = {}
        ok_quality[tree] = [
            row
            for row in quality_latest[tree].values()
            if row.get("status") == "ok" and row.get("policy") in ("fixed", "naive")
        ]
        counts[f"quality_rows_{tree}"] = len(quality_latest[tree])
        counts[f"quality_ok_{tree}"] = len(ok_quality[tree])

    # 2. Quality completeness against the plan.
    if grids is not None:
        for tree, arm in TREE_ARMS:
            cells = grids[arm]["quality"]
            have: set[tuple[str, int, int, int]] = set()
            for row in ok_quality[tree]:
                try:
                    have.add(
                        (
                            str(row["dataset"]),
                            int(row["L"]),
                            int(row["K"]),
                            int(row["window"]),
                        )
                    )
                except (KeyError, TypeError, ValueError):
                    continue
            missing = sorted(cells - have)
            counts[f"quality_expected_{tree}"] = len(cells)
            counts[f"quality_missing_{tree}"] = len(missing)
            if missing:
                shown = ", ".join(_cell_text(cell) for cell in missing[:LIST_LIMIT])
                hard.append(
                    f"{tree} quality missing {len(missing)} expected "
                    f"(dataset, L, K, window) cells; first "
                    f"{min(LIST_LIMIT, len(missing))}: {shown}"
                )

    # Timing rows: latest graph/batch-1 row per (L, K).
    timing_latest: dict[str, dict[tuple[int, int], dict[str, Any]]] = {}
    for tree in TREES:
        path = RESULTS / tree / "timesfm" / "timing.jsonl"
        try:
            rows = read_jsonl(path)
        except ValueError as exc:
            hard.append(f"{tree} timing unreadable: {exc}")
            rows = []
        latest: dict[tuple[int, int], dict[str, Any]] = {}
        for row in rows:
            if row.get("exec") != "graph" or row.get("batch") != 1:
                continue
            if row.get("L") is None or row.get("K") is None:
                continue
            latest[(int(row["L"]), int(row["K"]))] = row
        timing_latest[tree] = latest
        counts[f"timing_rows_{tree}"] = len(latest)

    # 3. Timing completeness against the plan.
    if grids is not None:
        for tree, arm in TREE_ARMS:
            pairs = grids[arm]["timing"]
            ok_pairs = {
                pair
                for pair, row in timing_latest[tree].items()
                if row.get("status") == "ok"
            }
            missing = sorted(pairs - ok_pairs)
            counts[f"timing_expected_{tree}"] = len(pairs)
            counts[f"timing_missing_{tree}"] = len(missing)
            if missing:
                shown = ", ".join(f"L{length}/K{k}" for length, k in missing[:LIST_LIMIT])
                hard.append(
                    f"{tree} timing missing/not-ok for {len(missing)} (L, K) "
                    f"pairs; first {min(LIST_LIMIT, len(missing))}: {shown}"
                )

    # 4. EXP-2 kernel classes: six lengths at batch 1, main length at batch 8.
    kern_path = RESULTS / "EXP2_stages" / "timesfm" / "kernels.jsonl"
    try:
        kernels = _latest(kern_path, ("L", "batch", "path"))
    except ValueError as exc:
        hard.append(f"EXP2 kernels unreadable: {exc}")
        kernels = {}
    required = [
        (length, 1, path_name)
        for length in SPEC.lengths
        for path_name in ("full", "rolling")
    ]
    required += [(SPEC.main_length, 8, path_name) for path_name in ("full", "rolling")]
    kernel_ok = 0
    for key in required:
        row = kernels.get(key)
        if row is None or row.get("status") != "ok":
            length, batch, path_name = key
            hard.append(
                f"EXP2 kernels missing/not-ok: L={length} batch={batch} {path_name}"
            )
        else:
            kernel_ok += 1
    counts["exp2_kernel_expected"] = len(required)
    counts["exp2_kernel_ok"] = kernel_ok

    # 5. EXP-4 KV diagnostics artifacts.
    csv_path = _find_exp4_file("reduced_k12.csv")
    if csv_path is None:
        hard.append(f"EXP4 reduced_k12.csv missing under {EXP4_DIR}")
    else:
        try:
            with csv_path.open(newline="", encoding="utf-8") as handle:
                reader = csv.reader(handle)
                header = next(reader, None)
                data = [row for row in reader if any(cell.strip() for cell in row)]
        except (OSError, csv.Error) as exc:
            hard.append(f"EXP4 reduced_k12.csv unreadable: {exc}")
            header, data = None, []
        if header is not None and tuple(cell.strip() for cell in header) != EXP4_HEADER:
            hard.append(
                f"EXP4 reduced_k12.csv header mismatch: got {header!r}, "
                f"want {list(EXP4_HEADER)!r}"
            )
        ages: set[int] = set()
        layers: set[int] = set()
        age_layer: set[tuple[int, int]] = set()
        unparseable = 0
        for row in data:
            if len(row) < len(EXP4_HEADER):
                unparseable += 1
                continue
            try:
                age = int(float(row[0]))
                layer = int(float(row[1]))
            except ValueError:
                unparseable += 1
                continue
            ages.add(age)
            layers.add(layer)
            age_layer.add((age, layer))
        counts["exp4_rows"] = len(data)
        counts["exp4_distinct_ages"] = len(ages)
        counts["exp4_distinct_layers"] = len(layers)
        if unparseable:
            hard.append(f"EXP4 reduced_k12.csv has {unparseable} unparseable rows")
        if (
            len(ages) < EXP4_MIN_AGES
            or len(layers) < EXP4_LAYERS
            or len(age_layer) < EXP4_MIN_AGES * EXP4_LAYERS
        ):
            hard.append(
                f"EXP4 reduced_k12.csv coverage too small: {len(ages)} ages x "
                f"{len(layers)} layers ({len(age_layer)} distinct pairs); "
                f"need >= {EXP4_MIN_AGES} x {EXP4_LAYERS}"
            )
    summary_path = _find_exp4_file("summary_per_age.json")
    if summary_path is None:
        hard.append(f"EXP4 summary_per_age.json missing under {EXP4_DIR}")
    else:
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            hard.append(f"EXP4 summary_per_age.json unreadable: {exc}")
        else:
            if not _has_key(summary, "rebase_self_check_max_abs"):
                hard.append(
                    "EXP4 summary_per_age.json lacks rebase_self_check_max_abs"
                )

    # 6. Warning: |D| <= G invariant with percentage-point slack.
    violations: list[str] = []
    for tree in TREES:
        for row in ok_quality[tree]:
            delta = row.get("mae_delta_pct")
            gap = row.get("gap_pct")
            if delta is None or gap is None:
                continue
            if abs(float(delta)) > float(gap) + GAP_TOLERANCE_PP:
                violations.append(
                    f"{tree}:{row.get('dataset')}/L{row.get('L')}/K{row.get('K')}"
                    f"/w{row.get('window')} |D|={abs(float(delta)):.3f} "
                    f"G={float(gap):.3f}"
                )
    counts["d_le_g_violations"] = len(violations)
    if violations:
        warnings.append(
            f"|D|<=G violated on {len(violations)} rows "
            f"(tolerance {GAP_TOLERANCE_PP}pp): "
            + "; ".join(violations[:LIST_LIMIT])
        )

    # 7. Warning: T7 baseline sanity on K=1 five-window medians.
    offenders: list[str] = []
    null_mse: list[str] = []
    for tree in TREES:
        per_cell: dict[tuple[str, int], list[float]] = {}
        for row in ok_quality[tree]:
            if row.get("K") != 1:
                continue
            mse = row.get("mse_h32")
            if mse is None:
                null_mse.append(
                    f"{tree}:{row.get('dataset')}/L{row.get('L')}/w{row.get('window')}"
                )
                continue
            key = (str(row.get("dataset")), int(row.get("L")))
            per_cell.setdefault(key, []).append(float(mse))
        medians: dict[str, dict[int, float]] = {}
        for (dataset, length), values in per_cell.items():
            medians.setdefault(dataset, {})[length] = float(np.median(values))
        for dataset in sorted(medians):
            lengths = sorted(medians[dataset])
            for low, high in zip(lengths, lengths[1:]):
                a = medians[dataset][low]
                b = medians[dataset][high]
                floor = min(a, b)
                ratio = max(a, b) / floor if floor > 0 else float("inf")
                if ratio > T7_JUMP_FACTOR:
                    offenders.append(
                        f"{tree}:{dataset} L{low}->L{high} median mse_h32 "
                        f"{a:.6g}->{b:.6g} ({ratio:.2f}x)"
                    )
    counts["t7_offenders"] = len(offenders)
    counts["t7_null_mse_h32_k1"] = len(null_mse)
    if offenders:
        warnings.append(
            f"T7 adjacent-L median mse_h32 jump >{T7_JUMP_FACTOR:g}x "
            f"({len(offenders)}): " + "; ".join(offenders[:LIST_LIMIT])
        )
    if null_mse:
        warnings.append(
            f"K=1 rows with null mse_h32 ({len(null_mse)}): "
            + ", ".join(null_mse[:LIST_LIMIT])
        )

    # 8. Warning: inconsistent timing flags per tree.
    for tree in TREES:
        inconsistent = sum(
            1
            for row in timing_latest[tree].values()
            if row.get("timing_flag") == "inconsistent"
        )
        counts[f"timing_inconsistent_{tree}"] = inconsistent
        if inconsistent:
            warnings.append(
                f"{tree} has {inconsistent} timing rows flagged inconsistent"
            )

    # 9. Warning: NPZ spot-check on a deterministic sample of predictions.
    pool: list[tuple[str, int, dict[str, Any]]] = []
    for tree, updates in (("EXP1_sweep", SPEC.updates), ("EXP1_ext", EXT_UPDATES)):
        for row in ok_quality[tree]:
            if row.get("preds_file"):
                pool.append((tree, updates, row))
    rng = random.Random(SEED_BASE)
    checked = bad = 0
    for tree, updates, row in rng.sample(pool, min(NPZ_SAMPLE, len(pool))):
        pred_path = RESULTS / row["preds_file"]
        checked += 1
        try:
            with np.load(pred_path, allow_pickle=False) as archive:
                yhat = archive["yhat"]
                if tuple(yhat.shape) != (updates, SPEC.horizon):
                    bad += 1
                    warnings.append(
                        f"NPZ yhat shape {tuple(yhat.shape)} != "
                        f"{(updates, SPEC.horizon)}: {pred_path}"
                    )
                elif not np.isfinite(yhat).all():
                    bad += 1
                    warnings.append(f"NPZ yhat has non-finite values: {pred_path}")
        except Exception as exc:  # missing file, bad archive, bad key
            bad += 1
            warnings.append(
                f"NPZ unreadable: {pred_path} ({type(exc).__name__}: {exc})"
            )
    counts["npz_checked"] = checked
    counts["npz_bad"] = bad

    result = {
        "schema": "rolling-kv-validation-timesfm",
        "ts": utc_now(),
        "passed": not hard,
        "hard_failures": hard,
        "warnings": warnings,
        "counts": counts,
    }
    write_json_atomic(RESULTS / "validation_timesfm.json", result)
    return result


def main() -> int:
    plan_error: str | None = None
    grids: dict[str, dict[str, set]] | None = None
    try:
        launcher = importlib.import_module("timesfm.launch_campaign")
    except Exception as exc:
        plan_error = (
            "cannot import launch_timesfm (source of the expected grid): "
            f"{type(exc).__name__}: {exc}"
        )
    else:
        try:
            grids = expected_cells(launcher.plan())
        except Exception as exc:
            plan_error = (
                f"launch_timesfm.plan() not usable: {type(exc).__name__}: {exc}"
            )
    result = validate(grids, plan_error)
    verdict = "PASS" if result["passed"] else "FAIL"
    print(
        f"validate_campaign[timesfm]: {verdict} "
        f"({len(result['hard_failures'])} hard failures, "
        f"{len(result['warnings'])} warnings)"
    )
    for line in result["hard_failures"]:
        print(f"  HARD: {line}")
    for line in result["warnings"]:
        print(f"  WARN: {line}")
    print("  counts: " + json.dumps(result["counts"], sort_keys=True))
    print(f"  wrote {RESULTS / 'validation_timesfm.json'}")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
