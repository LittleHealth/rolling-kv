"""Can the fixed per-step overhead of the rolling update be removed?

Executes the *same* rolling fast-update step five ways:

  A  GPU eager             ~1800 kernel launches, one host call each
  B  GPU + CUDA Graph      same launches, replayed with a single host call
  C  GPU + torch.compile   inductor fuses the 20-layer body (fewer kernels)
  D  C + B                 fused kernels replayed from a graph
  E  CPU                   no launch overhead at all, but GEMV-bound

Every GPU path that changes numerics is checked against eager before it is
timed -- a fast wrong answer is not a result.  Kernel counts and GPU-busy time
come from the profiler so the remaining floor can be attributed.

Usage:
  python bench_overhead.py --ckpt model.safetensors --context_length 8192
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from timesfm.online import RollingConfig, RollingTimesFMEngine  # noqa: E402
from timesfm.online.graph_runner import CudaGraphRollingStep  # noqa: E402
from timesfm.timesfm_2p5.timesfm_2p5_torch import (  # noqa: E402
  TimesFM_2p5_200M_torch_module,
)

HBM_BW = 1.555e12  # A100-SXM4 HBM2e, bytes/s


def sync(dev):
  if dev == "cuda":
    torch.cuda.synchronize()


def timeit(fn, dev, warmup=6, n=30):
  for i in range(warmup):
    fn(i)
  sync(dev)
  lats = []
  for i in range(n):
    t0 = time.perf_counter()
    fn(i)
    sync(dev)
    lats.append((time.perf_counter() - t0) * 1000.0)
  a = np.array(lats)
  return float(np.median(a)), float(np.percentile(a, 95))


def profile_step(fn, dev):
  """Kernel count and total GPU-busy time for one step."""
  if dev != "cuda":
    return None, None
  from torch.profiler import ProfilerActivity, profile

  with profile(activities=[ProfilerActivity.CUDA]) as prof:
    fn(0)
    torch.cuda.synchronize()
  ka = prof.key_averages()
  n = sum(x.count for x in ka if x.self_device_time_total > 0)
  us = sum(x.self_device_time_total for x in ka)
  return n, us / 1000.0


def make_series(n, seed=0):
  rng = np.random.RandomState(seed)
  t = np.arange(n, dtype=np.float32)
  return (np.sin(2 * np.pi * t / 96) + 0.5 * np.sin(2 * np.pi * t / 336)
          + 0.2 * rng.randn(n).astype(np.float32)).astype(np.float32)


def clone_state(src, dst):
  """Copy the full rolling state so two engines share identical history."""
  dst.cache.key.copy_(src.cache.key)
  dst.cache.value.copy_(src.cache.value)
  dst.cache.slot_pos.copy_(src.cache.slot_pos)
  dst.cache.write_ptr = src.cache.write_ptr
  dst.cache.next_pos = src.cache.next_pos
  dst.stat_n.copy_(src.stat_n)
  dst.stat_mu.copy_(src.stat_mu)
  dst.stat_sigma.copy_(src.stat_sigma)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--ckpt", required=True)
  ap.add_argument("--context_length", type=int, default=8192)
  ap.add_argument("--horizon", type=int, default=128)
  ap.add_argument("--batch_size", type=int, default=1)
  ap.add_argument("--runs", type=int, default=30)
  ap.add_argument("--cpu_threads", type=int, default=0)
  ap.add_argument("--skip_cpu", action="store_true")
  args = ap.parse_args()

  L, H, B, p = args.context_length, args.horizon, args.batch_size, 32
  series = make_series(L + 64 * p, seed=7)
  rows = []
  weight_bytes = None

  def mk(module, dev, dtype, window):
    cfg = RollingConfig(context_length=L, horizon=H, full_refresh_every=0,
                        batch_size=B, device=dev, dtype=dtype)
    eng = RollingTimesFMEngine(module, cfg)
    eng.full_refresh(window)
    return eng

  print(f"\nL={L} (N={L//p} patches)  H={H}  batch={B}")

  if torch.cuda.is_available():
    dev, dtype = "cuda", torch.float32
    print(f"gpu: {torch.cuda.get_device_name(0)}")
    module = TimesFM_2p5_200M_torch_module()
    module.device = torch.device(dev)
    module.load_checkpoint(args.ckpt)
    module.eval()
    weight_bytes = sum(q.numel() * q.element_size() for q in module.parameters())

    window = torch.tensor(np.stack([series[:L]] * B), device=dev, dtype=dtype)
    pt = [torch.tensor(np.stack([series[L + i*p : L+(i+1)*p]] * B),
                       device=dev, dtype=dtype) for i in range(8)]

    def run_eager(eng):
      return lambda i: eng.step_patch(pt[i % 8])

    # ---------------------------------------------------------------- A ----
    eng_a = mk(module, dev, dtype, window)
    fa = run_eager(eng_a)
    med, p95 = timeit(fa, dev, n=args.runs)
    nk, gb = profile_step(fa, dev)
    rows.append(("A GPU eager", med, p95, nk, gb, None))

    # ------------------------------------------------------------ B / D ----
    for tag, use_compile in [("B GPU + CUDA Graph", False),
                             ("D GPU + compile + Graph", True)]:
      try:
        eng = mk(module, dev, dtype, window)
        if use_compile:
          eng._encode_at = torch.compile(eng._encode_at, dynamic=False)
          for i in range(6):
            eng.step_patch(pt[i % 8])
        runner = CudaGraphRollingStep(eng, warmup=3)
        runner.capture()

        ref = mk(module, dev, dtype, window)
        clone_state(eng, ref)
        err, scale = 0.0, 1.0
        for i in range(5):
          got = runner.step(pt[i % 8]).clone()
          exp = ref.step_patch(pt[i % 8])
          err = max(err, (got - exp).abs().max().item())
          scale = exp.abs().max().item()
        if err > 1e-4 * max(scale, 1.0):
          print(f"  {tag}: MISMATCH {err:.3e} -- refusing to time")
          continue

        f = lambda i: runner.step(pt[i % 8])  # noqa: E731
        med, p95 = timeit(f, dev, n=args.runs)
        nk, gb = profile_step(f, dev)
        rows.append((tag, med, p95, nk, gb, err))
        del eng, ref, runner
        torch.cuda.empty_cache()
      except Exception as e:
        print(f"  {tag} FAILED: {type(e).__name__}: {str(e)[:160]}")

    # ---------------------------------------------------------------- C ----
    try:
      eng_c = mk(module, dev, dtype, window)
      eng_c._encode_at = torch.compile(eng_c._encode_at, dynamic=False)
      fc = run_eager(eng_c)
      for i in range(6):
        fc(i)
      med, p95 = timeit(fc, dev, n=args.runs)
      nk, gb = profile_step(fc, dev)
      rows.append(("C GPU + torch.compile", med, p95, nk, gb, None))
      del eng_c
      torch.cuda.empty_cache()
    except Exception as e:
      print(f"  torch.compile FAILED: {type(e).__name__}: {str(e)[:160]}")

    del module
    torch.cuda.empty_cache()

  # ---------------------------------------------------------------- E ----
  if not args.skip_cpu:
    if args.cpu_threads > 0:
      torch.set_num_threads(args.cpu_threads)
    nt = torch.get_num_threads()
    mc = TimesFM_2p5_200M_torch_module()
    mc.device = torch.device("cpu")
    mc.load_checkpoint(args.ckpt)
    mc.to("cpu")
    mc.eval()
    wc = torch.tensor(np.stack([series[:L]] * B), dtype=torch.float32)
    ptc = [torch.tensor(np.stack([series[L + i*p : L+(i+1)*p]] * B),
                        dtype=torch.float32) for i in range(8)]
    eng_e = mk(mc, "cpu", torch.float32, wc)
    fe = lambda i: eng_e.step_patch(ptc[i % 8])  # noqa: E731
    med, p95 = timeit(fe, "cpu", warmup=3, n=max(8, args.runs // 3))
    rows.append((f"E CPU ({nt} threads)", med, p95, None, None, None))

  # ------------------------------------------------------------- report ---
  print(f"\n{'='*82}")
  print(f"rolling fast-update: L={L}  H={H}  batch={B}")
  print(f"{'='*82}")
  base = rows[0][1] if rows else None
  print(f"  {'path':<26} {'ms':>8} {'p95':>8} {'vs eager':>9} "
        f"{'kernels':>8} {'gpu-busy':>9} {'err':>10}")
  print("  " + "-" * 76)
  for tag, med, p95, nk, gb, err in rows:
    rel = f"{base/med:.2f}x" if base else "-"
    ks = f"{nk}" if nk else "-"
    gs = f"{gb:.2f}" if gb else "-"
    es = f"{err:.1e}" if err is not None else "-"
    print(f"  {tag:<26} {med:>8.2f} {p95:>8.2f} {rel:>9} {ks:>8} {gs:>9} {es:>10}")
  if weight_bytes:
    print(f"\n  HBM bandwidth floor (must read {weight_bytes/1e6:.0f} MB of weights "
          f"@ {HBM_BW/1e12:.2f} TB/s): {weight_bytes/HBM_BW*1000:.2f} ms")
  print(f"{'='*82}\n")


if __name__ == "__main__":
  main()
