"""
Memory footprint analysis: model weights vs KV cache across context lengths.

Measures:
  1. Model parameter bytes (static, loaded once)
  2. KV cache bytes for L = 512, 2048, 4096
  3. Per-step HBM traffic for full_recompute vs fast_update
  4. Back-calculated HBM bandwidth from observed latencies
  5. Roofline-predicted latency vs measured latency
"""

import sys, os, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from time_moe.models.modeling_time_moe import TimeMoeForPrediction
from time_moe.online.rolling_engine import RollingTimeMoEEngine, EngineConfig
from time_moe.online.cache_utils import _get_layer_kv, _num_layers

DEVICE = "cuda"
DTYPE  = torch.bfloat16
MODEL_PATH = os.path.join(os.environ.get("ROLLKV_CKPT", "checkpoints"), "TimeMoE-50M")
CONTEXT_LENGTHS = [512, 2048, 4096]
N_TIMING_RUNS   = 15   # median over this many timed iterations

# ── 1. Load model, measure weight bytes ──────────────────────────────────────
torch.cuda.empty_cache()
mem_before = torch.cuda.memory_allocated()

model = TimeMoeForPrediction.from_pretrained(
    MODEL_PATH, device_map=DEVICE, dtype=DTYPE
)
model.eval()

mem_after_model = torch.cuda.memory_allocated()

param_bytes  = sum(p.numel() * p.element_size() for p in model.parameters())
buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
model_gpu_mb = (mem_after_model - mem_before) / 1e6

SEP = "=" * 65

print(f"\n{SEP}")
print("  MODEL WEIGHTS")
print(SEP)
print(f"  Total params     : {sum(p.numel() for p in model.parameters())/1e6:.2f} M")
print(f"  Param bytes      : {param_bytes/1e6:.2f} MB  (bfloat16)")
print(f"  Buffer bytes     : {buffer_bytes/1e6:.3f} MB")
print(f"  GPU allocated    : {model_gpu_mb:.2f} MB")

# ── 2. KV cache size + per-step timing for each L ────────────────────────────
results = {}

for L in CONTEXT_LENGTHS:
    torch.cuda.empty_cache()
    rng    = np.random.RandomState(42)
    series = rng.randn(L + N_TIMING_RUNS + 10).astype(np.float32)

    mem_before_kv = torch.cuda.memory_allocated()

    cfg = EngineConfig(
        context_length=L,
        prediction_length=64,
        full_refresh_every=0,
        tail_recompute_every=0,
        batch_size=1,
        device=DEVICE,
        dtype=DTYPE,
    )
    engine = RollingTimeMoEEngine(model, cfg)

    # ---- full_refresh timing ------------------------------------------------
    t0 = time.perf_counter()
    engine.full_refresh(torch.tensor(series[:L]))
    torch.cuda.synchronize()
    t_full_refresh_ms = (time.perf_counter() - t0) * 1000

    mem_after_kv = torch.cuda.memory_allocated()

    # ---- KV cache byte count ------------------------------------------------
    cache   = engine.observed_cache
    n_layers = _num_layers(cache)
    kv_bytes = 0
    k0, v0  = _get_layer_kv(cache, 0)
    for i in range(n_layers):
        k, v = _get_layer_kv(cache, i)
        kv_bytes += k.numel() * k.element_size() + v.numel() * v.element_size()

    # ---- fast_update timing (warm + median) ---------------------------------
    # 3 warm-up steps
    for j in range(3):
        engine.step(float(series[L + j]))
        torch.cuda.synchronize()

    lats_fast = []
    for j in range(N_TIMING_RUNS):
        t0 = time.perf_counter()
        engine.step(float(series[L + j]))
        torch.cuda.synchronize()
        lats_fast.append((time.perf_counter() - t0) * 1000)
    t_fast_ms = float(np.median(lats_fast))

    # ---- full_recompute timing (fresh forward, no cache) --------------------
    # Normalise the window and time a full forward (use_cache=False)
    window = torch.tensor(series[:L], dtype=torch.float32, device=DEVICE)
    m, s   = window.mean().item(), max(window.std().item(), 1e-8)
    normed = ((window - m) / s).unsqueeze(0).to(DTYPE)   # [1, L]

    lats_full = []
    for _ in range(N_TIMING_RUNS):
        with torch.no_grad():
            t0 = time.perf_counter()
            out = model(input_ids=normed, use_cache=False,
                        return_dict=True, max_horizon_length=64)
            torch.cuda.synchronize()
            lats_full.append((time.perf_counter() - t0) * 1000)
    t_fullrecompute_ms = float(np.median(lats_full))

    results[L] = dict(
        kv_bytes           = kv_bytes,
        n_layers           = n_layers,
        k_shape            = list(k0.shape),
        gpu_delta_mb       = (mem_after_kv - mem_before_kv) / 1e6,
        t_full_refresh_ms  = t_full_refresh_ms,
        t_fast_ms          = t_fast_ms,
        t_fullrecompute_ms = t_fullrecompute_ms,
    )

    print(f"\n{SEP}")
    print(f"  KV CACHE  L={L}")
    print(SEP)
    print(f"  Num layers          : {n_layers}")
    print(f"  K or V shape/layer  : {list(k0.shape)}   (B, heads, seq, head_dim)")
    print(f"  KV cache total      : {kv_bytes/1e6:.2f} MB")
    print(f"  GPU delta (alloc)   : {(mem_after_kv-mem_before_kv)/1e6:.2f} MB")
    print(f"  KV / model weights  : {kv_bytes/param_bytes:.3f}x")
    print(f"  full_refresh (init) : {t_full_refresh_ms:.1f} ms  (incl kernel warmup)")
    print(f"  fast_update median  : {t_fast_ms:.1f} ms")
    print(f"  full_recompute med  : {t_fullrecompute_ms:.1f} ms  (use_cache=False)")

    del engine
    torch.cuda.empty_cache()

# ── 3. Roofline analysis ──────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  ROOFLINE ANALYSIS")
print(SEP)

# Back-calculate effective HBM bandwidth from L=512 fast_update.
# At L=512 the KV cache is small; weight loading dominates.
r512 = results[512]
# Lower bound: only weights
bw_weights_only = param_bytes / (r512["t_fast_ms"] / 1000) / 1e9
# Upper bound: weights + KV read(L-1) + KV write(1)  ≈  weights + kv
bw_with_kv      = (param_bytes + r512["kv_bytes"]) / (r512["t_fast_ms"] / 1000) / 1e9

print(f"  HBM bandwidth est (weights only)   : {bw_weights_only:.1f} GB/s")
print(f"  HBM bandwidth est (weights + KV512): {bw_with_kv:.1f} GB/s")
print()

# Per-step HBM traffic breakdown
# fast_update : read all weights  + read KV(L-1) + write KV(1/layer)
# full_recomp : read all weights  + write KV(L)
# (no-cache full recompute writes nothing)
hdr = (f"{'L':>6}  {'weights':>10}  {'KV_total':>10}  {'KV/W':>7}"
       f"  {'fast BW':>10}  {'full BW':>10}"
       f"  {'pred_fast':>10}  {'meas_fast':>10}  {'meas_full':>10}  {'speedup':>8}")
print(hdr)
print("-" * len(hdr))

for L in CONTEXT_LENGTHS:
    r     = results[L]
    W_mb  = param_bytes / 1e6
    KV_mb = r["kv_bytes"] / 1e6
    ratio = r["kv_bytes"] / param_bytes

    # HBM bytes per step
    fast_bw_mb = (param_bytes + r["kv_bytes"]) / 1e6           # read W + read KV(L)
    full_bw_mb = (param_bytes + r["kv_bytes"]) / 1e6           # read W + write KV(L)

    # Predicted latency using bandwidth derived from L=512
    pred_fast_ms = fast_bw_mb / (bw_weights_only * 1e3)        # bw in GB/s → MB/ms

    speedup = r["t_fullrecompute_ms"] / r["t_fast_ms"]

    print(f"{L:>6}  {W_mb:>10.1f}  {KV_mb:>10.2f}  {ratio:>7.3f}x"
          f"  {fast_bw_mb:>10.1f}  {full_bw_mb:>10.1f}"
          f"  {pred_fast_ms:>10.1f}  {r['t_fast_ms']:>10.1f}"
          f"  {r['t_fullrecompute_ms']:>10.1f}  {speedup:>8.2f}x")

print()
print("Columns: L | weights(MB) | KV_total(MB) | KV/Weights |",
      "fast_BW(MB) | full_BW(MB) |",
      "predicted_fast(ms) | measured_fast(ms) | measured_full(ms) | speedup")
print()
print("Note: fast_BW ≈ full_BW (both read ~same bytes from HBM).")
print("Speedup comes from COMPUTE reduction (O(L) vs O(L^2) attention FLOPs),")
print("visible only when FLOPs exceed the memory-bandwidth ceiling.")
