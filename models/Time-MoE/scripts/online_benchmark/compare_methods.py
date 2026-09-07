"""
Side-by-side comparison of full-recompute vs rolling-cache variants.

Runs all four methods on the SAME series batch and the SAME loaded model,
then prints a comparison table:

  | method              | lat_ms  | p95_ms  | samp/s  | MAE   | MSE   | MASE  | delta_mae |
  |---------------------|---------|---------|---------|-------|-------|-------|-----------|
  | A: full_recompute   |         |         |         |       |       |       | baseline  |
  | B0: fast_only       |         |         |         |       |       |       |           |
  | B1: fast+refresh    |         |         |         |       |       |       |           |
  | B2: fast+tail+r     |         |         |         |       |       |       |           |

`delta_mae` = MAE( pred_method - pred_full ) averaged over steps × horizons.

Usage
-----
python scripts/online_benchmark/compare_methods.py \
    --model Maple728/TimeMoE-50M \
    --steps 60 \
    --context_length 512 \
    --prediction_length 64 \
    --batch_size 1
"""

import argparse
import sys
import os
import json
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from scripts.online_benchmark.run_full_recompute import (
    run as run_full,
    get_device_dtype,
    load_model,
    make_synthetic_series,
)
from scripts.online_benchmark.run_rolling_cache import run as run_rolling
from time_moe.online.metrics import compute_mae


# ---------------------------------------------------------------------------
# Comparison driver
# ---------------------------------------------------------------------------

def compare(args):
    device, dtype = get_device_dtype()
    print(f"Device: {device}  dtype: {dtype}  batch_size: {args.batch_size}\n")

    L, H, B = args.context_length, args.prediction_length, args.batch_size
    total_len = L + args.steps + H + 10

    # Generate B independent series (different seeds)
    all_series = [make_synthetic_series(total_len, seed=args.seed + b) for b in range(B)]

    print(f"Loading model once: {args.model}")
    model = load_model(args.model, device, dtype)

    # ---- Method A --------------------------------------------------------
    print("\n--- Method A: Full Recompute ---")
    summary_a, preds_a = run_full(args, model=model, all_series=all_series,
                                  device=device, dtype=dtype)

    # ---- Methods B -------------------------------------------------------
    summaries = {"A: full_recompute": summary_a}
    all_preds = {"A: full_recompute": preds_a}

    for method_id, label in [
        ("B0", "B0: fast_only"),
        ("B1", "B1: fast+refresh"),
        ("B2", "B2: fast+tail+refresh"),
    ]:
        print(f"\n--- Method {method_id} ---")
        s, p = run_rolling(args, method=method_id, model=model, all_series=all_series,
                           device=device, dtype=dtype)
        summaries[label] = s
        all_preds[label] = p

    # ---- Delta vs full ---------------------------------------------------
    min_rows = min(len(p) for p in all_preds.values())
    ref_preds = preds_a[:min_rows]

    for label, preds in all_preds.items():
        if label == "A: full_recompute":
            summaries[label]["delta_mae"] = 0.0
        else:
            delta = compute_mae(preds[:min_rows], ref_preds)
            summaries[label]["delta_mae"] = round(float(delta), 5)

    # ---- Print table -----------------------------------------------------
    cols = ["mean_latency_ms", "p95_latency_ms", "samples_per_sec",
            "mae", "mse", "mase", "delta_mae"]
    col_w = 15

    header = f"{'method':<32}" + "".join(f"{c:>{col_w}}" for c in cols)
    sep = "-" * len(header)
    print(f"\n{'='*len(header)}")
    print(header)
    print(sep)
    for label, s in summaries.items():
        row = f"{label:<32}" + "".join(
            f"{s.get(c, 0):>{col_w}.4f}" for c in cols
        )
        print(row)
    print(f"{'='*len(header)}\n")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(summaries, f, indent=2)
        print(f"Results saved to {args.output}")

    return summaries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser("TimeMoE Online Inference Comparison")
    parser.add_argument("--model", default="Maple728/TimeMoE-50M")
    parser.add_argument("--steps", type=int, default=60,
                        help="Number of online steps per method")
    parser.add_argument("--context_length", "-c", type=int, default=512)
    parser.add_argument("--prediction_length", "-p", type=int, default=64)
    parser.add_argument("--batch_size", "-b", type=int, default=1)
    parser.add_argument("--tail_length", type=int, default=128)
    parser.add_argument("--tail_recompute_every", type=int, default=16)
    parser.add_argument("--full_refresh_every", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    compare(args)
