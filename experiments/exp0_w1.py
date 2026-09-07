"""Run the EXP-0 correctness gate for one W1 model."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

from common import (
    CHECKPOINTS,
    MODELS,
    MODELS_ROOT,
    RESULTS,
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


def completed_length(path: Path, context_length: int) -> bool:
    latest: dict[tuple[str, Any], dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("L") == context_length:
            latest[(row.get("gate"), row.get("cache_age"))] = row
    for gate in ("T1", "T2", "T3", "T4"):
        row = latest.get((gate, None))
        if not row or row.get("status") != "ok" or row.get("passed") is not True:
            return False
    return all(
        ("T5", age) in latest and latest[("T5", age)].get("status") == "ok"
        for age in range(1, 9)
    )


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
            "threshold": 1e-6,
            "passed": graph_max <= 1e-6,
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
    for gate in ("T1", "T2", "T3", "T4"):
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
    else:
        sys.path.insert(0, str(MODELS_ROOT / "Time-MoE"))
        from time_moe.models.modeling_time_moe import TimeMoeForPrediction

        module = TimeMoeForPrediction.from_pretrained(
            str(CHECKPOINTS / MODELS[args.model].checkpoint),
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        ).eval()

    for context_length in MODELS[args.model].lengths:
        if completed_length(path, context_length):
            print(f"EXP0 already complete: {args.model} L={context_length}", flush=True)
            continue
        try:
            if args.model == "timesfm":
                ok = run_timesfm_length(module, context_length, path)
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
