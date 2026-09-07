"""Latency benchmark: Timer-S1 rolling KV cache vs full recompute.

Output format mirrors Time-MoE/scripts/online_benchmark/bench_rolling.py so
the two MoE models can be put side by side, with a horizon sweep added
because Timer-S1's cost structure is horizon-dominated: rolling KV reuse
eliminates only the 24-layer trunk prefill, while the MTP head is
architecturally uncacheable and runs up to 16 fresh full-sequence decoder
passes per step whenever horizon > 16.  Expect a large speedup at H=16 (zero
MTP passes) and only about (24+16)/16 ~ 2.5x at H=272.  A derived
"mtp_ms ~ roll(H) - roll(16)" line per context length makes the MTP share
visible without instrumentation.

The rolling path is additionally split into

  (a) evict : slice_cache physically copies the KV tensors minus the oldest
              slot, then rebases every surviving key by R(-1) (the
              k_scale-aware form in rope_utils) -- reported as % of rolling;
  (b) the 1-token trunk forward plus the full MTP recompute.

The full 8.3B checkpoint only fits A100-40GB in bf16, hence the bf16 default.

Usage:
  python bench_rolling.py --device cuda:0 \
      --context-lengths 1440,5760,11520 --horizons 16,96,272
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from timers1_online import RollingTimerS1Engine, TimerS1RollingConfig  # noqa: E402
from timers1_online.cache_utils import (  # noqa: E402
    get_layer_kv,
    num_layers,
    slice_cache,
)
from timers1_online.rope_utils import rebase_rope_keys_minus_one_  # noqa: E402

DEFAULT_CKPT = os.path.join(os.environ.get("ROLLKV_CKPT", "checkpoints"), "Timer-S1")


def sync(device):
    if device.startswith("cuda"):
        # Pass the device through: a bare synchronize() barriers the CURRENT
        # device (cuda:0), silently under-measuring on --device cuda:1.
        torch.cuda.synchronize(device)
    elif device.startswith("mps"):
        torch.mps.synchronize()


def timeit(fn, device, warmup=3, n=15):
    for _ in range(warmup):
        fn()
    sync(device)
    lats = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        sync(device)
        lats.append((time.perf_counter() - t0) * 1000.0)
    a = np.array(lats)
    return dict(
        median=float(np.median(a)), mean=float(a.mean()),
        p95=float(np.percentile(a, 95)), std=float(a.std()),
    )


def make_series(n, seed=0):
    rng = np.random.RandomState(seed)
    t = np.arange(n, dtype=np.float32)
    return (
        np.sin(2 * np.pi * t / 96)
        + 0.5 * np.sin(2 * np.pi * t / 336)
        + 0.2 * rng.randn(n).astype(np.float32)
    ).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    ap.add_argument("--context-lengths", default="1440,5760,11520")
    ap.add_argument("--horizons", default="16,96,272")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--runs", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--output", default=None)
    ap.add_argument(
        "--force-fp32", action="store_true",
        help="allow --dtype float32 on the full checkpoint (needs ~33 GB)",
    )
    args = ap.parse_args()

    if args.dtype == "float32" and not args.force_fp32:
        print(
            "refusing --dtype float32 on the full 8.3B checkpoint (~33 GB of "
            "weights leaves no activation room on A100-40GB); pass "
            "--force-fp32 to override"
        )
        return 2

    dev = args.device
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    B = args.batch_size
    Ls = [int(x) for x in args.context_lengths.split(",")]

    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt, trust_remote_code=True, torch_dtype=dtype
    ).to(dev).eval()
    patch = int(model.config.input_token_len)
    max_out = int(
        model.config.output_token_lens[-1] + patch * model.config.num_mtp_tokens
    )
    Hs = []
    for tok in args.horizons.split(","):
        h = min(int(tok), max_out)
        if h not in Hs:
            Hs.append(h)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\ndevice={dev}  dtype={args.dtype}  batch={B}  horizons={Hs}")
    print(f"params: {n_params / 1e6:.1f}M")
    if dev.startswith("cuda"):
        print(f"gpu: {torch.cuda.get_device_name(dev)}")

    n_cycle = 256
    results = {}
    derived = {}

    for L in Ls:
        if L % patch or L < patch:
            raise ValueError(f"context length {L} must be a positive multiple of {patch}")
        full_series = np.stack(
            [make_series(L + n_cycle * patch, seed=7 + b) for b in range(B)]
        )  # [B, L + n_cycle*patch]
        raw = full_series[:, :L]
        mu = raw.mean(axis=1, keepdims=True)
        sd = np.maximum(raw.std(axis=1, keepdims=True), 1e-8)
        normed = torch.tensor((raw - mu) / sd, device=dev, dtype=dtype)
        cont = torch.tensor(full_series[:, L:], device=dev)  # fp32 continuation

        roll_by_h = {}
        for H in Hs:

            def full_step():
                with torch.no_grad():
                    out = model(
                        input_ids=normed,
                        use_cache=False,
                        return_dict=True,
                        max_output_length=H,
                        revin=False,
                    )
                    return out.logits

            t_full = timeit(full_step, dev, warmup=args.warmup, n=args.runs)

            cfg = TimerS1RollingConfig(
                context_length=L, horizon=H, full_refresh_every=0,
                batch_size=B, device=dev, dtype=dtype,
            )
            eng = RollingTimerS1Engine(model, cfg)
            eng.full_refresh(raw)

            counter = {"i": 0}

            def roll_step():
                i = counter["i"] % n_cycle
                counter["i"] += 1
                eng.fast_update(cont[:, i * patch : (i + 1) * patch])

            # steady-state: at capacity, every call takes the evict branch
            t_roll = timeit(roll_step, dev, warmup=5, n=args.runs)

            # eviction cost in isolation: slice-copy plus the R(-1) key
            # rebase, on a fresh slice -- never mutating the live cache.
            def evict_only():
                survivors = slice_cache(eng.cache, start=1)
                if eng.rebase_factors is not None:
                    for li in range(num_layers(survivors)):
                        key, _ = get_layer_kv(survivors, li)
                        rebase_rope_keys_minus_one_(key, eng.rebase_factors[li])

            t_evict = timeit(evict_only, dev, warmup=args.warmup, n=args.runs)

            nl = num_layers(eng.cache)
            k0, _ = get_layer_kv(eng.cache, 0)
            kv_mb = sum(
                get_layer_kv(eng.cache, i)[0].numel() * k0.element_size() * 2
                for i in range(nl)
            ) / 1e6

            speedup = t_full["median"] / t_roll["median"]
            roll_by_h[H] = t_roll["median"]
            results[f"{L}x{H}"] = dict(
                context_length=L, horizon=H, full=t_full, rolling=t_roll,
                evict=t_evict, speedup=speedup, kv_mb=kv_mb,
            )

            print(f"\n  L={L:>6}  H={H:>4}")
            print(
                f"    full recompute   : {t_full['median']:8.2f} ms  "
                f"(p95 {t_full['p95']:7.2f})"
            )
            print(
                f"    rolling update   : {t_roll['median']:8.2f} ms  "
                f"(p95 {t_roll['p95']:7.2f})"
            )
            print(
                f"      |- evict       : {t_evict['median']:8.2f} ms  "
                f"({100 * t_evict['median'] / t_roll['median']:5.1f}% of rolling)"
            )
            print(
                f"    speedup          : {speedup:8.2f}x       KV cache {kv_mb:.1f} MB"
            )

        if 16 in roll_by_h:
            derived[str(L)] = {
                f"mtp_ms_H{H}": roll_by_h[H] - roll_by_h[16]
                for H in roll_by_h
                if H != 16
            }
            for name, value in sorted(derived[str(L)].items()):
                print(
                    f"    {name:<14}   : {value:8.2f} ms  "
                    "(roll(H) - roll(16), MTP share)"
                )

    print(f"\n{'=' * 78}")
    print("Timer-S1 (8.3B total / 0.75B activated)  rolling KV cache vs full recompute")
    print(f"{'=' * 78}")
    hdr = (
        f"{'L':>7} {'H':>5} {'full(ms)':>10} {'roll(ms)':>10} {'evict(ms)':>11} "
        f"{'speedup':>9} {'KV(MB)':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for key, r in results.items():
        print(
            f"{r['context_length']:>7} {r['horizon']:>5} {r['full']['median']:>10.2f} "
            f"{r['rolling']['median']:>10.2f} {r['evict']['median']:>11.2f} "
            f"{r['speedup']:>8.2f}x {r['kv_mb']:>8.1f}"
        )
    print(f"{'=' * 78}\n")

    if args.output:
        out_dir = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(out_dir, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(
                {
                    "config": vars(args),
                    "params_M": n_params / 1e6,
                    "results": results,
                    "derived_mtp_ms": derived,
                },
                f,
                indent=2,
            )
        print(f"saved -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
