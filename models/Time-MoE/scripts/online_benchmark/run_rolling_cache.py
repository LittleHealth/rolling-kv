"""
Method B: Rolling KV-Cache Variants
=====================================
Three configurations benchmarked here:

  B0  fast path only   (drop-oldest + append, no periodic refresh)
  B1  fast + full_refresh every K steps
  B2  fast + tail_recompute every M steps + full_refresh every K steps

Usage
-----
python scripts/online_benchmark/run_rolling_cache.py \
    --model Maple728/TimeMoE-50M \
    --steps 100 \
    --context_length 512 \
    --prediction_length 64 \
    --batch_size 1 \
    --method B1 \
    --full_refresh_every 64 \
    --tail_recompute_every 16 \
    --tail_length 128
"""

import argparse
import time
import json
import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from time_moe.online import RollingTimeMoEEngine
from time_moe.online.rolling_engine import EngineConfig
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


def _method_to_config(args, method: str, device: str, dtype: torch.dtype) -> EngineConfig:
    base = dict(
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        tail_length=args.tail_length,
        batch_size=args.batch_size,
        device=device,
        dtype=dtype,
    )
    if method == "B0":
        return EngineConfig(**base, tail_recompute_every=0, full_refresh_every=0)
    elif method == "B1":
        return EngineConfig(**base, tail_recompute_every=0,
                            full_refresh_every=args.full_refresh_every)
    elif method == "B2":
        return EngineConfig(**base,
                            tail_recompute_every=args.tail_recompute_every,
                            full_refresh_every=args.full_refresh_every)
    else:
        raise ValueError(f"Unknown method: {method}")


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run(args, method: str = None, model=None, all_series: list = None,
        device: str = None, dtype=None):
    method = method or args.method

    if device is None or dtype is None:
        device, dtype = get_device_dtype()
    print(f"[rolling/{method}] device={device}  dtype={dtype}  batch={args.batch_size}")

    if model is None:
        print(f"Loading model: {args.model}")
        model = load_model(args.model, device, dtype)

    L, H, B = args.context_length, args.prediction_length, args.batch_size
    total_len = L + args.steps + H

    if all_series is None:
        all_series = [make_synthetic_series(total_len, seed=args.seed + b) for b in range(B)]

    naive_scale = naive_mae(all_series[0][:L], period=1)

    cfg = _method_to_config(args, method, device, dtype)
    engine = RollingTimeMoEEngine(model, cfg)

    # Warm up: feed the initial context window for all B series
    init_windows = torch.tensor(
        np.stack([s[:L] for s in all_series]), dtype=torch.float32
    )  # [B, L]
    print(f"  Initializing with full_refresh over {L} points (batch={B})...")
    t_init0 = time.perf_counter()
    engine.full_refresh(init_windows)
    if device == "cuda":
        torch.cuda.synchronize()
    t_init = time.perf_counter() - t_init0
    print(f"  Init done in {t_init*1000:.0f} ms")

    results = []
    latencies = []
    all_preds = []    # list of [B, H] arrays
    all_targets = []  # list of [B, H] arrays

    for step in range(args.steps):
        t = L + step
        x_news = torch.tensor([s[t] for s in all_series], dtype=torch.float32)  # [B]
        targets = np.stack([s[t + 1: t + 1 + H] for s in all_series])           # [B, H]

        t0 = time.perf_counter()
        pred_t = engine.step(x_news)
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()
        lat = time.perf_counter() - t0

        # pred_t: [H] (B=1) or [B, H]
        if pred_t.dim() == 1:
            pred_t = pred_t.unsqueeze(0)  # [1, H]
        preds = pred_t.numpy()  # [B, H]

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
        "method": method,
        "model": args.model,
        "context_length": L,
        "prediction_length": H,
        "batch_size": B,
        "steps": len(latencies),
        "full_refresh_every": cfg.full_refresh_every,
        "tail_recompute_every": cfg.tail_recompute_every,
        "tail_length": cfg.tail_length,
        "mean_latency_ms": float(np.mean(lats)) if len(lats) else 0,
        "p50_latency_ms": float(np.percentile(lats, 50)) if len(lats) else 0,
        "p95_latency_ms": float(np.percentile(lats, 95)) if len(lats) else 0,
        "samples_per_sec": float(B * 1000.0 / np.mean(lats)) if len(lats) else 0,
        "mae": float(compute_mae(preds_np, targets_np)) if len(preds_np) else 0,
        "mse": float(compute_mse(preds_np, targets_np)) if len(preds_np) else 0,
        "smape": float(compute_smape(preds_np, targets_np)) if len(preds_np) else 0,
        "mase": float(
            compute_mae(preds_np, targets_np) / max(naive_scale, 1e-8)
        ) if len(preds_np) else 0,
    }

    print(f"\n=== Rolling Cache [{method}] Summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    if getattr(args, "output", None):
        with open(args.output, "w") as f:
            json.dump({"summary": summary, "steps": results}, f, indent=2)
        print(f"\nResults saved to {args.output}")

    return summary, preds_np


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser("TimeMoE Rolling-Cache Benchmark")
    parser.add_argument("--model", default="Maple728/TimeMoE-50M")
    parser.add_argument("--method", choices=["B0", "B1", "B2"], default="B1")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--context_length", "-c", type=int, default=512)
    parser.add_argument("--prediction_length", "-p", type=int, default=64)
    parser.add_argument("--batch_size", "-b", type=int, default=1)
    parser.add_argument("--tail_length", type=int, default=128)
    parser.add_argument("--tail_recompute_every", type=int, default=16)
    parser.add_argument("--full_refresh_every", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    run(args)
