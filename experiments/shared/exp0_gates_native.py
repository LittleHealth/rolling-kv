"""Run the EXP-0 correctness gate for one W1 model."""

from __future__ import annotations

import sys as _sys
from pathlib import Path as _Path

_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))  # experiments/ root

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from shared.common import (
    CHECKPOINTS,
    MODELS,
    MODELS_ROOT,
    RESULTS,
    TFM3_ALL_LENGTHS,
    append_jsonl,
    base_record,
    classify_failure,
    read_jsonl,
)


def make_series(length: int, seed: int = 7) -> np.ndarray:
    rng = np.random.RandomState(seed)
    x = np.arange(length, dtype=np.float32)
    return (
        np.sin(2 * np.pi * x / 96)
        + 0.5 * np.sin(2 * np.pi * x / 336)
        + 0.2 * rng.randn(length).astype(np.float32)
    ).astype(np.float32)


def error_stats(got: torch.Tensor, ref: torch.Tensor) -> tuple[float, float, float]:
    got = got.detach().float()
    ref = ref.detach().float()
    max_abs = float((got - ref).abs().max().item())
    scale = float(ref.abs().max().item())
    return max_abs, max_abs / max(scale, 1e-12), scale


def gate_row(model: str, gate: str, context_length: int, dtype: str) -> dict[str, Any]:
    row = base_record("A", "EXP0", model)
    row.update(
        {
            "gate": gate,
            "L": context_length,
            "dtype": dtype,
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


def append_gate(path: Path, row: dict[str, Any]) -> bool:
    append_jsonl(path, row)
    passed = row.get("passed")
    age = "" if row.get("cache_age") is None else f" age={row['cache_age']}"
    print(
        f"{row['model']} L={row['L']} {row['gate']}{age}: "
        f"status={row['status']} passed={passed} rel={row.get('rel_err')}",
        flush=True,
    )
    return row["status"] == "ok" and passed is not False


def required_gates(model: str) -> tuple[str, ...]:
    if model in {"timesfm", "timesfm3"}:
        return ("T1", "T2", "T3", "T4", "T6")
    return ("T1", "T2", "T3", "T4")


def gate_lengths(model: str) -> tuple[int, ...]:
    """Contexts the gate must cover: every context the campaign measures.

    TimesFM-3.0 also measures the partial contexts 1024/4096 (sweep and
    kernel profile), which sit outside its main-grid ``spec.lengths``.
    """
    if model == "timesfm3":
        return TFM3_ALL_LENGTHS
    return MODELS[model].lengths


def completed_length(path: Path, context_length: int, model: str) -> bool:
    latest: dict[tuple[str, Any], dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("L") == context_length:
            latest[(row.get("gate"), row.get("cache_age"))] = row
    for gate in required_gates(model):
        row = latest.get((gate, None))
        if not row or row.get("status") != "ok" or row.get("passed") is not True:
            return False
    return all(
        ("T5", age) in latest and latest[("T5", age)].get("status") == "ok"
        for age in range(1, 9)
    )


def timesfm_state_snapshot(engine) -> tuple[dict[str, Any], dict[str, float]]:
    """Persistent engine state, with cache contents gathered in logical slot order.

    Two engines holding the same logical state may differ in ring head pointer,
    so live slots are ordered by their absolute position tags (cache.slot_pos)
    rather than by physical index before comparison.
    """
    cache = engine.cache
    live = torch.nonzero(cache.slot_pos >= 0, as_tuple=False).squeeze(1)
    slots = live[torch.argsort(cache.slot_pos[live])]
    tensors = {
        "slot_pos": cache.slot_pos[slots].float(),
        "key": cache.key.index_select(2, slots),
        "value": cache.value.index_select(2, slots),
        "stat_n": engine.stat_n,
        "stat_mu": engine.stat_mu,
        "stat_sigma": engine.stat_sigma,
        "raw_buffer": engine.raw_buffer,
        "last_mu": engine.last_mu,
        "last_sigma": engine.last_sigma,
        "last_embedding": engine.last_embedding,
    }
    scalars = {
        "n_live_slots": float(slots.numel()),
        "next_pos": float(cache.next_pos),
        "n_written": float(cache.n_written),
    }
    return tensors, scalars


@torch.no_grad()
def run_timesfm_length(module, context_length: int, path: Path) -> bool:
    sys.path.insert(0, str(MODELS_ROOT / "TimesFM-2.5" / "src"))
    from timesfm.online import RollingConfig, RollingTimesFMEngine
    from timesfm.online.graph_runner import CudaGraphRollingStep
    from timesfm.torch import util

    spec = MODELS["timesfm"]
    p, horizon = spec.s, spec.horizon
    series = make_series(context_length + 10 * p)
    window = torch.as_tensor(series[:context_length][None, :], device="cuda")
    cfg = RollingConfig(
        context_length=context_length,
        horizon=horizon,
        full_refresh_every=0,
        batch_size=1,
        device="cuda",
        dtype=torch.float32,
    )

    def upstream_decode(raw: torch.Tensor) -> torch.Tensor:
        masks = torch.zeros_like(raw, dtype=torch.bool)
        point, _, autoregressive = module.decode(horizon, raw, masks)
        pieces = [point[:, -1, ...]]
        if autoregressive is not None:
            pieces.append(autoregressive.reshape(raw.shape[0], -1, module.q))
        return torch.cat(pieces, dim=1)[:, :horizon, module.aridx]

    ok = True
    engine = RollingTimesFMEngine(module, cfg)
    engine.full_refresh(window)
    got = engine.forecast()
    ref = upstream_decode(window)
    max_abs, rel, scale = error_stats(got, ref)
    row = gate_row("timesfm", "T1", context_length, spec.dtype)
    row.update(
        {
            "max_abs_err": max_abs,
            "rel_err": rel,
            "scale": scale,
            "threshold": 1e-5,
            "passed": rel <= 1e-5,
            "comparison": "custom full prefill/readout vs upstream module.decode",
        }
    )
    ok &= append_gate(path, row)

    # Append-only growth into an empty ring; compare every token output with a
    # single upstream prefill over the complete context.
    patches = window.view(1, -1, p)
    masks = torch.zeros_like(patches, dtype=torch.bool)
    grow = RollingTimesFMEngine(module, cfg)
    grow.cache.reset()
    grow.raw_buffer = torch.zeros_like(window)
    outputs = []
    for index in range(patches.shape[1]):
        mu, sigma = grow._advance_stats(
            patches[:, index : index + 1], masks[:, index : index + 1]
        )
        normed = util.revin(
            patches[:, index : index + 1], mu, sigma, reverse=False
        )
        embedding = grow._encode(normed, masks[:, index : index + 1])
        outputs.append(grow._readout(embedding, mu[:, -1], sigma[:, -1])[..., module.aridx])
    grown = torch.cat(outputs, dim=0)

    n = torch.zeros(1, device="cuda")
    mu = torch.zeros(1, device="cuda")
    sigma = torch.zeros(1, device="cuda")
    mus, sigmas = [], []
    for index in range(patches.shape[1]):
        (n, mu, sigma), _ = util.update_running_stats(
            n, mu, sigma, patches[:, index], masks[:, index]
        )
        mus.append(mu)
        sigmas.append(sigma)
    cmu, csigma = torch.stack(mus, 1), torch.stack(sigmas, 1)
    normalized = util.revin(patches, cmu, csigma, reverse=False)
    (_, _, normalized_out, _), _ = module(normalized, masks, None)
    ref_all = util.revin(normalized_out, cmu, csigma, reverse=True)
    ref_all = ref_all.reshape(1, -1, module.o, module.q)[0, ..., module.aridx]
    max_abs, rel, scale = error_stats(grown, ref_all)
    row = gate_row("timesfm", "T2", context_length, spec.dtype)
    row.update(
        {
            "max_abs_err": max_abs,
            "rel_err": rel,
            "scale": scale,
            "threshold": 1e-5,
            "passed": rel <= 1e-5,
            "comparison": "append-only rolling growth vs one full prefill",
        }
    )
    ok &= append_gate(path, row)

    eager = RollingTimesFMEngine(module, cfg)
    eager.full_refresh(window)
    graph_engine = RollingTimesFMEngine(module, cfg)
    graph_engine.full_refresh(window)
    graph = CudaGraphRollingStep(graph_engine)
    graph.capture(preserve_state=True)
    graph_max = 0.0
    graph_scale = 0.0
    for age in range(1, 4):
        update = torch.as_tensor(
            series[context_length + (age - 1) * p : context_length + age * p][None, :],
            device="cuda",
        )
        expected = eager.step_patch(update)
        actual = graph.step(update).clone()
        torch.cuda.synchronize()
        current, _, scale = error_stats(actual, expected)
        graph_max = max(graph_max, current)
        graph_scale = max(graph_scale, scale)
    graph_rel = graph_max / max(graph_scale, 1e-12)
    row = gate_row("timesfm", "T3", context_length, spec.dtype)
    row.update(
        {
            "max_abs_err": graph_max,
            "rel_err": graph_rel,
            "scale": graph_scale,
            "threshold": 0.0,
            "passed": graph_max == 0.0,
            "comparison": "rolling CUDA Graph replay vs corresponding eager path",
        }
    )
    ok &= append_gate(path, row)

    row = gate_row("timesfm", "T4", context_length, spec.dtype)
    row.update(
        {
            "max_abs_err": 0.0,
            "rel_err": 0.0,
            "scale": 0.0,
            "threshold": 1e-12,
            "passed": True,
            "note": "not applicable: TimesFM uses monotonic absolute positions",
        }
    )
    ok &= append_gate(path, row)

    # T6 refresh reset fidelity: after rolling activity a full refresh must
    # land bit-for-bit in the state a from-scratch refresh produces on the
    # same window (no state leakage across refresh).
    window3 = torch.as_tensor(
        series[3 * p : context_length + 3 * p][None, :], device="cuda"
    )
    active = RollingTimesFMEngine(module, cfg)
    active.full_refresh(window)
    for age in range(1, 4):
        update = torch.as_tensor(
            series[context_length + (age - 1) * p : context_length + age * p][None, :],
            device="cuda",
        )
        active.step_patch(update)
    active.full_refresh(window3)
    fresh = RollingTimesFMEngine(module, cfg)
    fresh.full_refresh(window3)
    got_tensors, got_scalars = timesfm_state_snapshot(active)
    ref_tensors, ref_scalars = timesfm_state_snapshot(fresh)
    t6_max = 0.0
    t6_scale = 0.0
    mismatch = None
    for name, ref in ref_tensors.items():
        got = got_tensors[name].detach().float()
        ref = ref.detach().float()
        if got.shape != ref.shape:
            mismatch = (
                f"state shape mismatch in {name}: "
                f"{tuple(got.shape)} vs {tuple(ref.shape)}"
            )
            break
        if ref.numel():
            t6_max = max(t6_max, float((got - ref).abs().max().item()))
            t6_scale = max(t6_scale, float(ref.abs().max().item()))
    for name, ref in ref_scalars.items():
        t6_max = max(t6_max, abs(got_scalars[name] - ref))
        t6_scale = max(t6_scale, abs(ref))
    row = gate_row("timesfm", "T6", context_length, spec.dtype)
    row.update(
        {
            "max_abs_err": None if mismatch else t6_max,
            "rel_err": None if mismatch else t6_max / max(t6_scale, 1e-12),
            "scale": None if mismatch else t6_scale,
            "threshold": 0.0,
            "passed": False if mismatch else t6_max == 0.0,
            "comparison": "full refresh after 3 rolling steps vs fresh-engine "
            "full refresh on the same window (logical slot order)",
        }
    )
    if mismatch:
        row["note"] = mismatch
    ok &= append_gate(path, row)

    rolling = RollingTimesFMEngine(module, cfg)
    rolling.full_refresh(window)
    for age in range(1, 9):
        lo = context_length + (age - 1) * p
        update = torch.as_tensor(series[lo : lo + p][None, :], device="cuda")
        roll_pred = rolling.step_patch(update)
        current = torch.as_tensor(
            series[lo + p - context_length : lo + p][None, :], device="cuda"
        )
        full_pred = upstream_decode(current)
        max_abs, rel, scale = error_stats(roll_pred, full_pred)
        gap_mae = float((roll_pred.float() - full_pred.float()).abs().mean().item())
        row = gate_row("timesfm", "T5", context_length, spec.dtype)
        row.update(
            {
                "max_abs_err": max_abs,
                "rel_err": rel,
                "scale": scale,
                "threshold": None,
                "passed": None,
                "cache_age": age,
                "gap_rel": gap_mae / max(float(full_pred.float().abs().mean().item()), 1e-12),
            }
        )
        ok &= append_gate(path, row)
    return ok


def timesfm3_state_snapshot(engine) -> tuple[dict[str, Any], dict[str, float]]:
    """Persistent TimesFM-3.0 engine state in logical (position) slot order.

    Everything a later fast update or forecast reads: ring K/V and position
    tags, running RevIN stats, the frozen detrend line and its integer time
    offset, the raw window, the newest-token readout state and the recorded
    token history.
    """
    cache = engine.cache
    live = torch.nonzero(cache.slot_pos >= 0, as_tuple=False).squeeze(1)
    slots = live[torch.argsort(cache.slot_pos[live])]
    tensors = {
        "slot_pos": cache.slot_pos[slots].float(),
        "key": cache.key.index_select(2, slots),
        "value": cache.value.index_select(2, slots),
        "stat_n": engine.stat_n,
        "stat_mu": engine.stat_mu,
        "stat_sigma": engine.stat_sigma,
        "trend_m": engine.trend_m,
        "trend_c": engine.trend_c,
        "trend_apply": engine.trend_apply.float(),
        "raw_buffer": engine.raw_buffer,
        "last_mu": engine.last_mu,
        "last_sigma": engine.last_sigma,
        "last_embedding": engine.last_embedding,
        "token_history": engine.token_history[:, :, : engine._hist_len],
    }
    scalars = {
        "n_live_slots": float(slots.numel()),
        "next_pos": float(cache.next_pos),
        "n_written": float(cache.n_written),
        "t_offset": float(engine.t_offset),
        "hist_len": float(engine._hist_len),
    }
    return tensors, scalars


def compare_snapshots(
    got: tuple[dict[str, Any], dict[str, float]],
    ref: tuple[dict[str, Any], dict[str, float]],
) -> tuple[float, float, str | None]:
    """Max abs difference and scale over two state snapshots (logical order)."""
    got_tensors, got_scalars = got
    ref_tensors, ref_scalars = ref
    worst = 0.0
    scale = 0.0
    for name, expected in ref_tensors.items():
        actual = got_tensors[name].detach().float()
        expected = expected.detach().float()
        if actual.shape != expected.shape:
            return 0.0, 0.0, (
                f"state shape mismatch in {name}: "
                f"{tuple(actual.shape)} vs {tuple(expected.shape)}"
            )
        if expected.numel():
            worst = max(worst, float((actual - expected).abs().max().item()))
            scale = max(scale, float(expected.abs().max().item()))
    for name, expected in ref_scalars.items():
        worst = max(worst, abs(got_scalars[name] - expected))
        scale = max(scale, abs(expected))
    return worst, scale, None


def timesfm3_probe_series(length: int) -> dict[str, np.ndarray]:
    """Gate inputs.

    ``plain`` is the TimesFM-2.5 gate series verbatim (make_series, seed 7), so
    every recorded gate value is measured on the identical input.  It carries
    no trend, so TimesFM-3.0's linear-detrend gate stays off on it; ``trended``
    adds a strong ramp (as the vendored test_exactness.py does) so the frozen
    detrend line -- fit, per-update detrend, trend re-add, and its in-graph
    integer time offset -- is exercised too.  Pass/fail requires both.
    """
    plain = make_series(length)
    ramp = 16.0 * np.arange(length, dtype=np.float32) / np.float32(length)
    return {"plain": plain, "trended": (plain + ramp).astype(np.float32)}


@torch.no_grad()
def run_timesfm3_length(module, context_length: int, path: Path) -> bool:
    sys.path.insert(0, str(MODELS_ROOT / "TimesFM-3.0" / "src"))
    from timesfm3.online import RollingConfig, RollingTimesFM3Engine
    from timesfm3.online.graph_runner import CudaGraphFullDecode, CudaGraphRollingStep

    model_name = "timesfm3"
    spec = MODELS[model_name]
    p, horizon = spec.s, spec.horizon
    n_patches = context_length // p
    probes = timesfm3_probe_series(context_length + 10 * p)
    cfg = RollingConfig(
        context_length=context_length,
        horizon=horizon,
        full_refresh_every=0,
        batch_size=1,
        num_variates=1,
        device="cuda",
        dtype=torch.float32,
    )
    probe_engine = RollingTimesFM3Engine(module, cfg)
    if probe_engine.nh_scratch != 0:
        raise ValueError(f"H={horizon} needs horizon scratch tokens at L={context_length}")
    mq = probe_engine.median_q_idx
    del probe_engine

    def tensor(values: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(values[None, None, :], device="cuda")

    def window_at(series: np.ndarray, shift: int) -> torch.Tensor:
        return tensor(series[shift * p : shift * p + context_length])

    def update_at(series: np.ndarray, age: int) -> torch.Tensor:
        lo = context_length + (age - 1) * p
        return tensor(series[lo : lo + p])

    def upstream_point(raw: torch.Tensor) -> torch.Tensor:
        return module.decode(target=raw, horizon=horizon, mask=None)[..., mq]

    def summary(stats: tuple[float, float, float]) -> str:
        return f"max_abs={stats[0]:.3e} rel={stats[1]:.3e}"

    ok = True

    # T1: custom full prefill + readout vs upstream TimesFM3Torch.decode.  The
    # recorded values are the median-quantile point forecast on the plain
    # series (TimesFM-2.5 compares its point forecast, aridx = median).
    t1: dict[str, tuple[float, float, float]] = {}
    t1_all_q: dict[str, tuple[float, float, float]] = {}
    detrend_active: dict[str, bool] = {}
    for label, series in probes.items():
        window = window_at(series, 0)
        engine = RollingTimesFM3Engine(module, cfg)
        engine.full_refresh(window)
        got = engine.forecast()
        ref = module.decode(target=window, horizon=horizon, mask=None)
        t1[label] = error_stats(got[..., mq], ref[..., mq])
        t1_all_q[label] = error_stats(got, ref)
        detrend_active[label] = bool(engine.trend_apply.any().item())
    row = gate_row(model_name, "T1", context_length, spec.dtype)
    max_abs, rel, scale = t1["plain"]
    row.update(
        {
            "max_abs_err": max_abs,
            "rel_err": rel,
            "scale": scale,
            "threshold": 1e-5,
            "passed": all(value[1] <= 1e-5 for value in t1.values()),
            "comparison": "custom full prefill/readout vs upstream TimesFM3Torch.decode "
            "(median-quantile point forecast, H=64)",
            "note": (
                f"trended probe (detrend {'on' if detrend_active['trended'] else 'off'}): "
                f"{summary(t1['trended'])}; all 9 quantiles: plain "
                f"{summary(t1_all_q['plain'])}, trended {summary(t1_all_q['trended'])}; "
                f"plain-probe detrend {'on' if detrend_active['plain'] else 'off'}; "
                "passed requires the point forecast of both probes <= threshold"
            ),
        }
    )
    ok &= append_gate(path, row)

    # T2: append-only growth into an empty ring (no eviction, no detrend) vs
    # one upstream forward() over the complete context; every token's
    # per-patch output must match.
    t2: dict[str, tuple[float, float, float]] = {}
    for label, series in probes.items():
        window = window_at(series, 0)
        patches = window.view(1, 1, n_patches, p)
        masks = torch.zeros_like(patches, dtype=torch.bool)
        target = torch.ones(1, 1, n_patches, dtype=torch.bool, device="cuda")
        ref_all = module(
            {"values": patches, "masks": masks, "patch_is_target": target},
            freeze_after=None,
            patch_cpm_mask=None,
        )["logits"][0, 0, ..., mq]
        grow = RollingTimesFM3Engine(module, cfg)
        outputs = [
            grow.append_patches(patches[:, :, index : index + 1])[0, 0, :, :, mq]
            for index in range(n_patches)
        ]
        t2[label] = error_stats(torch.cat(outputs, dim=0), ref_all)
    row = gate_row(model_name, "T2", context_length, spec.dtype)
    max_abs, rel, scale = t2["plain"]
    row.update(
        {
            "max_abs_err": max_abs,
            "rel_err": rel,
            "scale": scale,
            "threshold": 1e-5,
            "passed": all(value[1] <= 1e-5 for value in t2.values()),
            "comparison": "append-only rolling growth vs one full prefill "
            "(upstream forward() per-patch outputs, median quantile)",
            "note": f"trended probe: {summary(t2['trended'])}; "
            "passed requires both probes <= threshold",
        }
    )
    ok &= append_gate(path, row)

    # T3: CUDA Graph replay vs the eager path it captures, bitwise.  Covers
    # exactly the replay sequence of the sweep: three rolling updates, a
    # graph-captured full refresh that installs its state into the rolling
    # graph, and a rolling update after that refresh.  Full quantile output.
    graph_max = 0.0
    graph_scale = 0.0
    for series in probes.values():
        eager = RollingTimesFM3Engine(module, cfg)
        eager.full_refresh(window_at(series, 0))
        graph_engine = RollingTimesFM3Engine(module, cfg)
        graph_engine.full_refresh(window_at(series, 0))
        rolling_graph = CudaGraphRollingStep(graph_engine)
        rolling_graph.capture(preserve_state=True)
        full_engine = RollingTimesFM3Engine(module, cfg)
        full_engine.full_refresh(window_at(series, 0))
        full_graph = CudaGraphFullDecode(full_engine, rolling_target=rolling_graph)
        full_graph.capture(preserve_target=True)
        pairs = []
        for age in range(1, 4):
            update = update_at(series, age)
            expected = eager.step_patch(update)
            actual = rolling_graph.step(update).clone()
            torch.cuda.synchronize()
            pairs.append((actual, expected))
        window4 = window_at(series, 4)
        eager.full_refresh(window4)
        expected = eager.forecast()
        actual = full_graph.step(window4).clone()
        torch.cuda.synchronize()
        pairs.append((actual, expected))
        update = update_at(series, 5)
        expected = eager.step_patch(update)
        actual = rolling_graph.step(update).clone()
        torch.cuda.synchronize()
        pairs.append((actual, expected))
        for actual, expected in pairs:
            current, _, scale = error_stats(actual, expected)
            graph_max = max(graph_max, current)
            graph_scale = max(graph_scale, scale)
    row = gate_row(model_name, "T3", context_length, spec.dtype)
    row.update(
        {
            "max_abs_err": graph_max,
            "rel_err": graph_max / max(graph_scale, 1e-12),
            "scale": graph_scale,
            "threshold": 0.0,
            "passed": graph_max == 0.0,
            "comparison": "rolling CUDA Graph replay vs corresponding eager path "
            "(3 rolling updates, graph full refresh, 1 post-refresh update; "
            "all 9 quantiles; plain and trended probes)",
        }
    )
    ok &= append_gate(path, row)

    row = gate_row(model_name, "T4", context_length, spec.dtype)
    row.update(
        {
            "max_abs_err": 0.0,
            "rel_err": 0.0,
            "scale": 0.0,
            "threshold": 1e-12,
            "passed": True,
            "note": "not applicable: TimesFM-3.0 uses monotonic absolute positions",
        }
    )
    ok &= append_gate(path, row)

    # T6: refresh reset fidelity -- after rolling activity a full refresh must
    # land bit-for-bit in the state a from-scratch refresh produces on the
    # same window (no leakage of stats, detrend offset, ring pointers or
    # token history across a refresh).
    t6_max = 0.0
    t6_scale = 0.0
    mismatch = None
    for series in probes.values():
        active = RollingTimesFM3Engine(module, cfg)
        active.full_refresh(window_at(series, 0))
        for age in range(1, 4):
            active.step_patch(update_at(series, age))
        active.full_refresh(window_at(series, 3))
        fresh = RollingTimesFM3Engine(module, cfg)
        fresh.full_refresh(window_at(series, 3))
        worst, scale, mismatch = compare_snapshots(
            timesfm3_state_snapshot(active), timesfm3_state_snapshot(fresh)
        )
        if mismatch:
            break
        t6_max = max(t6_max, worst)
        t6_scale = max(t6_scale, scale)
    row = gate_row(model_name, "T6", context_length, spec.dtype)
    row.update(
        {
            "max_abs_err": None if mismatch else t6_max,
            "rel_err": None if mismatch else t6_max / max(t6_scale, 1e-12),
            "scale": None if mismatch else t6_scale,
            "threshold": 0.0,
            "passed": False if mismatch else t6_max == 0.0,
            "comparison": "full refresh after 3 rolling steps vs fresh-engine "
            "full refresh on the same window (logical slot order; plain and "
            "trended probes)",
        }
    )
    if mismatch:
        row["note"] = mismatch
    ok &= append_gate(path, row)

    # T5 (diagnostic): rolling output gap vs upstream decode on the current
    # window, per cache age, on the TimesFM-2.5 gate series.
    series = probes["plain"]
    rolling = RollingTimesFM3Engine(module, cfg)
    rolling.full_refresh(window_at(series, 0))
    for age in range(1, 9):
        roll_pred = rolling.step_patch(update_at(series, age))[..., mq]
        full_pred = upstream_point(window_at(series, age))
        max_abs, rel, scale = error_stats(roll_pred, full_pred)
        gap_mae = float((roll_pred.float() - full_pred.float()).abs().mean().item())
        row = gate_row(model_name, "T5", context_length, spec.dtype)
        row.update(
            {
                "max_abs_err": max_abs,
                "rel_err": rel,
                "scale": scale,
                "threshold": None,
                "passed": None,
                "cache_age": age,
                "gap_rel": gap_mae / max(float(full_pred.float().abs().mean().item()), 1e-12),
            }
        )
        ok &= append_gate(path, row)
    return ok


@torch.no_grad()
def run_timemoe_length(model, context_length: int, path: Path) -> bool:
    sys.path.insert(0, str(MODELS_ROOT / "Time-MoE"))
    from time_moe.online import (
        CudaGraphRollingTimeMoEStep,
        RollingTimeMoEEngine,
        set_static_moe_dispatch,
    )
    from time_moe.online.rolling_engine import EngineConfig

    spec = MODELS["timemoe"]
    series = make_series(context_length + 16)
    window = torch.as_tensor(series[:context_length][None, :], device="cuda")
    cfg = EngineConfig(
        context_length=context_length,
        prediction_length=spec.horizon,
        tail_length=min(128, context_length),
        tail_recompute_every=0,
        full_refresh_every=0,
        batch_size=1,
        device="cuda",
        dtype=torch.bfloat16,
    )
    ok = True

    set_static_moe_dispatch(model, False)
    dynamic = RollingTimeMoEEngine(model, cfg)
    dynamic.full_refresh(window)
    dynamic_pred = dynamic.forecast().view(1, -1).cuda()
    normalized = dynamic._normalize_batch(window).to(torch.bfloat16)
    set_static_moe_dispatch(model, True)
    static_out = model(
        input_ids=normalized.clone(),
        use_cache=True,
        return_dict=True,
        max_horizon_length=spec.horizon,
    )
    static_pred = dynamic._denormalize_batch(
        static_out.logits[:, -1, : spec.horizon].float()
    )
    max_abs, rel, scale = error_stats(static_pred, dynamic_pred)
    row = gate_row("timemoe", "T1", context_length, spec.dtype)
    row.update(
        {
            "max_abs_err": max_abs,
            "rel_err": rel,
            "scale": scale,
            "threshold": None,
            "passed": True,
            "comparison": "graph-safe fixed-shape dispatcher vs upstream dynamic dispatcher",
            "note": "record-only by protocol; no threshold",
        }
    )
    ok &= append_gate(path, row)

    # One append without eviction: cached prefix plus the last token must equal
    # a single full prefill over the same normalized sequence.
    set_static_moe_dispatch(model, False)
    mean = window.float().mean(dim=1)
    std = window.float().std(dim=1).clamp_min(1e-8)
    normalized = ((window - mean[:, None]) / std[:, None]).to(torch.bfloat16)
    prefix = model(
        input_ids=normalized[:, :-1].clone(),
        use_cache=True,
        return_dict=True,
        max_horizon_length=spec.horizon,
    )
    incremental = model(
        input_ids=normalized[:, -1:].clone(),
        past_key_values=prefix.past_key_values,
        use_cache=True,
        return_dict=True,
        max_horizon_length=spec.horizon,
    ).logits[:, -1, : spec.horizon]
    full = model(
        input_ids=normalized.clone(),
        use_cache=True,
        return_dict=True,
        max_horizon_length=spec.horizon,
    ).logits[:, -1, : spec.horizon]
    max_abs, rel, scale = error_stats(incremental, full)
    row = gate_row("timemoe", "T2", context_length, spec.dtype)
    row.update(
        {
            "max_abs_err": max_abs,
            "rel_err": rel,
            "scale": scale,
            "threshold": 2e-2,
            "passed": rel <= 2e-2,
            "comparison": "append-only cached prefix plus one token vs one full prefill",
        }
    )
    ok &= append_gate(path, row)

    set_static_moe_dispatch(model, True)
    eager = RollingTimeMoEEngine(model, cfg)
    eager.full_refresh(window)
    graphed_engine = RollingTimeMoEEngine(model, cfg)
    graphed_engine.full_refresh(window)
    graph = CudaGraphRollingTimeMoEStep(graphed_engine)
    graph.capture(preserve_state=True)
    graph_max = 0.0
    graph_scale = 0.0
    for age in range(1, 4):
        value = float(series[context_length + age - 1])
        expected = eager.step(value).view(1, -1).cuda()
        actual = graph.step(value).clone()
        torch.cuda.synchronize()
        current, _, scale = error_stats(actual, expected)
        graph_max = max(graph_max, current)
        graph_scale = max(graph_scale, scale)
    graph_rel = graph_max / max(graph_scale, 1e-12)
    row = gate_row("timemoe", "T3", context_length, spec.dtype)
    row.update(
        {
            "max_abs_err": graph_max,
            "rel_err": graph_rel,
            "scale": graph_scale,
            "threshold": 1e-6,
            "passed": graph_max <= 1e-6,
            "comparison": "static-dispatch rolling CUDA Graph vs static-dispatch eager",
        }
    )
    ok &= append_gate(path, row)

    row = gate_row("timemoe", "T4", context_length, spec.dtype)
    row.update(
        {
            "max_abs_err": 0.0,
            "rel_err": 0.0,
            "scale": 0.0,
            "threshold": 1e-12,
            "passed": True,
            "note": "not applicable in W1: current engine has no independent survivor-key remapper",
        }
    )
    ok &= append_gate(path, row)

    set_static_moe_dispatch(model, False)
    rolling = RollingTimeMoEEngine(model, cfg)
    rolling.full_refresh(window)
    current_window = window.clone()
    for age in range(1, 9):
        value = float(series[context_length + age - 1])
        roll_pred = rolling.step(value).view(1, -1).cuda()
        new_value = torch.as_tensor([[value]], device="cuda")
        current_window = torch.cat((current_window[:, 1:], new_value), dim=1)
        reference = RollingTimeMoEEngine(model, cfg)
        reference.full_refresh(current_window)
        full_pred = reference.forecast().view(1, -1).cuda()
        max_abs, rel, scale = error_stats(roll_pred, full_pred)
        gap_mae = float((roll_pred.float() - full_pred.float()).abs().mean().item())
        row = gate_row("timemoe", "T5", context_length, spec.dtype)
        row.update(
            {
                "max_abs_err": max_abs,
                "rel_err": rel,
                "scale": scale,
                "threshold": None,
                "passed": None,
                "cache_age": age,
                "gap_rel": gap_mae / max(float(full_pred.float().abs().mean().item()), 1e-12),
            }
        )
        ok &= append_gate(path, row)
    return ok


def record_length_failure(model: str, context_length: int, path: Path, exc: BaseException) -> None:
    status = classify_failure(exc)
    reason = f"{type(exc).__name__}: {str(exc)[:1200]}"
    for gate in required_gates(model):
        row = gate_row(model, gate, context_length, MODELS[model].dtype)
        row.update({"status": status, "reason": reason})
        append_jsonl(path, row)
    for age in range(1, 9):
        row = gate_row(model, "T5", context_length, MODELS[model].dtype)
        row.update({"status": status, "reason": reason, "cache_age": age})
        append_jsonl(path, row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(MODELS), required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("EXP-0 W1 requires CUDA")
    path = RESULTS / "EXP0_correctness" / args.model / "records.jsonl"
    failed = False

    module = None
    if args.model == "timesfm":
        sys.path.insert(0, str(MODELS_ROOT / "TimesFM-2.5" / "src"))
        from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch_module

        module = TimesFM_2p5_200M_torch_module()
        module.device = torch.device("cuda")
        module.load_checkpoint(str(CHECKPOINTS / MODELS[args.model].checkpoint))
        module.eval()
    elif args.model == "timesfm3":
        sys.path.insert(0, str(MODELS_ROOT / "TimesFM-3.0" / "src"))
        from timesfm3.model import TimesFM3Torch

        # Directory form: honours config.json.  FP32 only upstream.
        module = TimesFM3Torch.from_pretrained(str(CHECKPOINTS / MODELS[args.model].checkpoint))
        module.to(device="cuda", dtype=torch.float32)
        module.eval()
    else:
        sys.path.insert(0, str(MODELS_ROOT / "Time-MoE"))
        from time_moe.models.modeling_time_moe import TimeMoeForPrediction

        module = TimeMoeForPrediction.from_pretrained(
            str(CHECKPOINTS / MODELS[args.model].checkpoint),
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        ).eval()

    for context_length in gate_lengths(args.model):
        if completed_length(path, context_length, args.model):
            print(f"EXP0 already complete: {args.model} L={context_length}", flush=True)
            continue
        try:
            if args.model == "timesfm":
                ok = run_timesfm_length(module, context_length, path)
            elif args.model == "timesfm3":
                ok = run_timesfm3_length(module, context_length, path)
            else:
                ok = run_timemoe_length(module, context_length, path)
            failed |= not ok
        except Exception as exc:
            traceback.print_exc()
            record_length_failure(args.model, context_length, path, exc)
            failed = True
        torch.cuda.empty_cache()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
