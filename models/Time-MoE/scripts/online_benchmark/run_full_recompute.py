"""
Method A: Full-Recompute Baseline
==================================
At each online step, re-encode the most recent `context_length` raw values
from scratch (no KV cache reuse) and forecast the next `prediction_length`
steps.  This is the ground-truth accuracy baseline.

Usage
-----
python scripts/online_benchmark/run_full_recompute.py \
    --model Maple728/TimeMoE-50M \
    --steps 100 \
    --context_length 512 \
    --prediction_length 64 \
    --batch_size 1 \
    --seed 42
"""

import argparse
import time
import json
import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from time_moe.online.metrics import compute_mae, compute_mse, compute_smape, naive_mae


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_device_dtype():
    if torch.cuda.is_available():
        return "cuda", torch.bfloat16
    if torch.backends.mps.is_available():
        return "mps", torch.float32
    return "cpu", torch.float32


def make_synthetic_series(n: int, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    t = np.arange(n, dtype=np.float32)
    series = (
        np.sin(2 * np.pi * t / 24)
        + 0.5 * np.sin(2 * np.pi * t / 168)
        + 0.2 * np.sin(2 * np.pi * t / 720)
        + 0.3 * rng.randn(n).astype(np.float32)
    )
    return series


def load_model(model_path: str, device: str, dtype: torch.dtype):
    try:
        from time_moe.models.modeling_time_moe import TimeMoeForPrediction
        model = TimeMoeForPrediction.from_pretrained(
            model_path,
            device_map=device,
            dtype=dtype,
        )
    except Exception:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
    model.eval()
    return model


def full_recompute_step(model, raw_windows: np.ndarray, H: int, device: str, dtype: torch.dtype):
    """Run one full-recompute forecast step.

    Args:
        raw_windows: [B, L] numpy array (B=1 for single series).
    Returns:
        preds: [B, H] numpy array in original scale.
    """
    if raw_windows.ndim == 1:
        raw_windows = raw_windows[np.newaxis, :]  # [1, L]

    B, L = raw_windows.shape
    means = raw_windows.mean(axis=1, keepdims=True)        # [B, 1]
    stds = np.maximum(raw_windows.std(axis=1, keepdims=True), 1e-8)
    normed = (raw_windows - means) / stds                  # [B, L]

    normed_t = torch.tensor(normed, dtype=dtype, device=device)

    with torch.no_grad():
        out = model(
            input_ids=normed_t,
            use_cache=False,
            return_dict=True,
            max_horizon_length=H,
        )

    pred_normed = out.logits[:, -1, :H].float().cpu().numpy()  # [B, H]
    preds = pred_normed * stds + means                          # [B, H]
    return preds  # [B, H]


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run(args, model=None, all_series: list = None, device: str = None, dtype=None):
    if device is None or dtype is None:
        device, dtype = get_device_dtype()
    print(f"[full_recompute] device={device}  dtype={dtype}  batch={args.batch_size}")

    if model is None:
        print(f"Loading model: {args.model}")
        model = load_model(args.model, device, dtype)

    L, H, B = args.context_length, args.prediction_length, args.batch_size
    total_len = L + args.steps + H

    if all_series is None:
        all_series = [make_synthetic_series(total_len, seed=args.seed + b) for b in range(B)]

    naive_scale = naive_mae(all_series[0][:L], period=1)

    results = []
    latencies = []
    all_preds = []    # list of [B, H] arrays
    all_targets = []  # list of [B, H] arrays

    for step in range(args.steps):
        t = L + step

        raw_windows = np.stack([s[t - L: t] for s in all_series])  # [B, L]
        targets = np.stack([s[t: t + H] for s in all_series])       # [B, H]

        t0 = time.perf_counter()
        preds = full_recompute_step(model, raw_windows, H, device, dtype)
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
        lat = time.perf_counter() - t0

        if targets.shape[1] < H:
            break

        mae = compute_mae(preds, targets)
        mse = compute_mse(preds, targets)
        smape = compute_smape(preds, targets)
        mase = mae / max(naive_scale, 1e-8)

        latencies.append(lat * 1000)
        all_preds.append(preds)
        all_targets.append(targets)

        results.append({
            "step": step,
            "latency_ms": lat * 1000,
            "mae": mae,
            "mse": mse,
            "smape": smape,
            "mase": mase,
        })

        if (step + 1) % 10 == 0:
            print(
                f"  step {step+1:4d}/{args.steps}  "
                f"lat={lat*1000:.1f}ms  mae={mae:.4f}"
            )

    lats = np.array(latencies)
    preds_np = np.concatenate(all_preds, axis=0) if all_preds else np.zeros((0, H))
    targets_np = np.concatenate(all_targets, axis=0) if all_targets else np.zeros((0, H))

    summary = {
        "method": "full_recompute",
        "model": args.model,
        "context_length": L,
        "prediction_length": H,
        "batch_size": B,
        "steps": len(latencies),
        "mean_latency_ms": float(np.mean(lats)),
        "p50_latency_ms": float(np.percentile(lats, 50)),
        "p95_latency_ms": float(np.percentile(lats, 95)),
        # per-sample throughput
        "samples_per_sec": float(B * 1000.0 / np.mean(lats)) if len(lats) else 0,
        "mae": float(compute_mae(preds_np, targets_np)) if len(preds_np) else 0,
        "mse": float(compute_mse(preds_np, targets_np)) if len(preds_np) else 0,
        "smape": float(compute_smape(preds_np, targets_np)) if len(preds_np) else 0,
        "mase": float(
            compute_mae(preds_np, targets_np) / max(naive_scale, 1e-8)
        ) if len(preds_np) else 0,
    }

    print("\n=== Full Recompute Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    if getattr(args, "output", None):
        with open(args.output, "w") as f:
            json.dump({"summary": summary, "steps": results}, f, indent=2)
        print(f"\nResults saved to {args.output}")

    # Return preds as [steps*B, H] for delta_mae computation
    return summary, preds_np


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser("TimeMoE Full-Recompute Benchmark")
    parser.add_argument("--model", default="Maple728/TimeMoE-50M")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--context_length", "-c", type=int, default=512)
    parser.add_argument("--prediction_length", "-p", type=int, default=64)
    parser.add_argument("--batch_size", "-b", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    run(args)
