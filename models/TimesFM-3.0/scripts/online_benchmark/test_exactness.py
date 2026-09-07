"""Correctness gates for the TimesFM-3.0 rolling engine.

T1a fidelity      : engine.full_refresh + forecast  ==  upstream model.decode
                    on the same window, H = 64 (no scratch tokens).  Proves
                    the re-expressed forward (RoPE / qk-norm / per-dim-scale /
                    variate attention / detrend / mask / readout) is faithful.
T1b fidelity      : same at H = 256, gating the horizon-scratch fork: the
                    window-anchored mask, folded scratch stats, restarted CPM
                    revin refinement, stitching, and the trend re-add.
T2  V-1 exactness : growing window (append only, no eviction) must reproduce
                    upstream forward()'s per-patch logits exactly.  This is
                    the ground truth for the ring buffer, position tags and
                    mask; any failure here is an engine bug, not a cache gap.
                    T2 lives at *forward* level, not decode level: decode()'s
                    detrend is a whole-window least-squares fit (model.py:
                    472-511), not causal per patch, so only forward()'s
                    per-patch logits are prefix-causal quantities.
T3  graph         : CUDA-graph replay of the fast-update step must be
                    bit-exact vs the eager engine, including across a
                    graph-captured full refresh.  CUDA only.
T4  eviction      : machinery gate.  Rolling-vs-recompute equality does NOT
                    hold for TimesFM-3.0, for two independent reasons: (a)
                    survivors' cached deep-layer K/V were computed while the
                    now-evicted patches were still visible (exactly the
                    bounded-staleness gap D1 measures), and (b) attention
                    logits are not invariant under position renumbering,
                    because RoPE is followed by elementwise-affine qk RMSNorm
                    and PerDimScale whose trained weights are not symmetric
                    within a rotation pair (see Pi-4 in rolling_engine.py).
                    T4 therefore drives a second engine whose ring is large
                    enough to never reuse a slot (`capacity_slack`) through
                    the identical refresh + update stream and asserts the two
                    engines match: same protocol semantics, different
                    physical slot layout, so any divergence beyond floating-
                    point reduction-order noise is a wraparound / tagging /
                    eviction bookkeeping bug.
D1  cache gap     : rolling vs decode() on the shifted raw window, per cache
                    age, plus a survivor re-encode split (remapped vs true
                    absolute positions) that separates the position term from
                    KV staleness.  Measurements, not assertions -- this *is*
                    the quantity the theory bounds.

Note on --dtype: only float32 runs.  The vendored TimesFM-3.0 forward is
fp32-only (util.get_running_stats keeps fp32 stats, so model._preprocess
builds an fp32 resblock input; torch 2.5.1 rejects fp32 input x bf16 weight
in nn.Linear), so at bf16 every reference call would crash before printing a
PASS/FAIL line.  The bf16 threshold of the contract cannot be exercised for
this model without modifying upstream.

Usage:
  python test_exactness.py --device cuda:0 [--dtype float32] [--ckpt DIR]
"""

import argparse
import dataclasses
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from timesfm3.model import TimesFM3Torch  # noqa: E402
from timesfm3.online import RollingConfig, RollingTimesFM3Engine  # noqa: E402

DEFAULT_CKPT = os.path.join(
  os.environ.get("ROLLKV_CKPT", "checkpoints"), "TimesFM-3.0"
)


def make_series(n, seed=0, trend=0.0):
  rng = np.random.RandomState(seed)
  t = np.arange(n, dtype=np.float32)
  base = (
    np.sin(2 * np.pi * t / 96)
    + 0.5 * np.sin(2 * np.pi * t / 336)
    + 0.2 * rng.randn(n).astype(np.float32)
  )
  return (base + trend * t / n).astype(np.float32)


def make_windows(B, V, total, seed0=7, trend=16.0):
  """(B, V, total) array.  Variate 0 carries a strong linear trend so
  decode()'s detrend gate fires (std_det << std_orig); the others carry none
  so it stays off -- both `torch.where` branches get exercised."""
  out = np.zeros((B, V, total), dtype=np.float32)
  for b in range(B):
    for v in range(V):
      tr = trend if v == 0 else 0.0
      out[b, v] = make_series(total, seed=seed0 + 13 * b + 101 * v, trend=tr)
  return out


def rel_err(got, ref):
  got, ref = got.float(), ref.float()
  denom = ref.abs().max().clamp_min(1e-9)
  return ((got - ref).abs().max() / denom).item()


def load_model(args, device, dtype):
  if args.random_init:
    torch.manual_seed(0)
    model = TimesFM3Torch()
  else:
    # The DIRECTORY form honors config.json (use_rope_var, use_sdpa,
    # use_stitching, use_frozen_running_stats, causal_attention, ...).
    model = TimesFM3Torch.from_pretrained(args.ckpt)
  model.to(device=device, dtype=dtype)
  model.eval()
  return model


def banner(title):
  print(f"\n{'=' * 68}")
  print(title)
  print(f"{'=' * 68}")


# --------------------------------------------------------------------- T1 ---


def run_t1(model, args, dev, dtype, th, horizon, label):
  banner(f"{label}  engine.full_refresh + forecast  vs  model.decode  (H={horizon})")
  L = args.context_length
  fails = 0
  for B, V in [(1, 1), (2, 3)]:
    win = torch.tensor(make_windows(B, V, L), device=dev, dtype=dtype)
    cfg = RollingConfig(
      context_length=L, horizon=horizon, full_refresh_every=0,
      batch_size=B, num_variates=V, device=dev, dtype=dtype,
    )
    eng = RollingTimesFM3Engine(model, cfg)
    eng.full_refresh(win)
    got = eng.forecast()  # [B, V, H, nq]
    ref = model.decode(target=win, horizon=horizon, mask=None)
    r = rel_err(got, ref)
    ok = r <= th
    fails += not ok
    print(
      f"  B={B} V={V}  scratch_patches={eng.nh_scratch:>2}  "
      f"rel_err {r:.3e}  (th {th:.0e})  -> {'PASS' if ok else 'FAIL'}"
    )
  return fails


# --------------------------------------------------------------------- T2 ---


def run_t2(model, args, dev, dtype, th):
  banner("T2  V-1 growing-window exactness (append only, no eviction)")
  L = args.context_length
  p = model.input_patch_len
  N = L // p
  fails = 0
  for B, V in [(1, 1), (2, 3)]:
    win = torch.tensor(make_windows(B, V, L), device=dev, dtype=dtype)
    patches = win.view(B, V, N, p)
    masks = torch.zeros(B, V, N, p, dtype=torch.bool, device=dev)
    pit = torch.ones(B, V, N, dtype=torch.bool, device=dev)
    with torch.no_grad():
      ref = model(
        {"values": patches, "masks": masks, "patch_is_target": pit},
        freeze_after=None,
        patch_cpm_mask=None,
      )["logits"]  # [B, V, N, o, nq], reverse-revin'd + clamped
    scale = ref.float().abs().max().clamp_min(1e-9)

    cfg = RollingConfig(
      context_length=L, horizon=model.output_patch_len, full_refresh_every=0,
      batch_size=B, num_variates=V, device=dev, dtype=dtype,
    )

    # T2a: one patch at a time.
    eng = RollingTimesFM3Engine(model, cfg)
    worst, worst_j = 0.0, -1
    for j in range(N):
      out = eng.append_patches(patches[:, :, j : j + 1])
      d = (out[:, :, 0].float() - ref[:, :, j].float()).abs().max()
      r = (d / scale).item()
      if r > worst:
        worst, worst_j = r, j
    ok_a = worst <= th
    fails += not ok_a
    print(
      f"  B={B} V={V}  T2a stride 1: rel_err {worst:.3e}  "
      f"(worst patch {worst_j}/{N - 1})  -> {'PASS' if ok_a else 'FAIL'}"
    )

    # T2b: chunks of 4 (multi-token reserve + multi-query mask).
    eng2 = RollingTimesFM3Engine(model, cfg)
    worst_b = 0.0
    for j in range(0, N, 4):
      out = eng2.append_patches(patches[:, :, j : j + 4])
      d = (out.float() - ref[:, :, j : j + 4].float()).abs().max()
      worst_b = max(worst_b, (d / scale).item())
    ok_b = worst_b <= th
    fails += not ok_b
    print(
      f"  B={B} V={V}  T2b stride 4: rel_err {worst_b:.3e}"
      f"{'':<22}  -> {'PASS' if ok_b else 'FAIL'}"
    )
  return fails


# --------------------------------------------------------------------- T3 ---


def run_t3(model, args, dev, dtype):
  banner("T3  CUDA graph replay vs eager (expect bit-exact, th 1e-6)")
  if not str(dev).startswith("cuda"):
    print("  device is not CUDA -- skipped (not counted as failure)")
    return 0
  from timesfm3.online.graph_runner import (  # noqa: E402
    CudaGraphFullDecode,
    CudaGraphRollingStep,
  )

  B, V, H, steps = 1, 2, 64, 8
  L = args.context_length
  p = model.input_patch_len
  data = make_windows(B, V, L + (steps + 4) * p)
  win0 = torch.tensor(data[:, :, :L], device=dev, dtype=dtype)
  cfg = RollingConfig(
    context_length=L, horizon=H, full_refresh_every=0,
    batch_size=B, num_variates=V, device=dev, dtype=dtype,
  )
  eng_eager = RollingTimesFM3Engine(model, cfg)
  eng_graph = RollingTimesFM3Engine(model, cfg)
  eng_eager.full_refresh(win0)
  eng_graph.full_refresh(win0)

  runner = CudaGraphRollingStep(eng_graph)
  runner.capture(preserve_state=True)

  fails = 0
  worst = 0.0
  for k in range(steps):
    lo = L + k * p
    patch = torch.tensor(data[:, :, lo : lo + p], device=dev, dtype=dtype)
    got_e = eng_eager.step_patch(patch)
    got_g = runner.step(patch)
    worst = max(worst, rel_err(got_g, got_e))
  ok = worst <= 1e-6
  fails += not ok
  print(f"  rolling step, {steps} replays : rel_err {worst:.3e}  "
        f"-> {'PASS' if ok else 'FAIL'}")

  # Graph-captured full refresh installing state into the rolling graph.
  fd = CudaGraphFullDecode(eng_graph, rolling_target=runner)
  fd.capture(warmup=3, preserve_target=True)
  lo = L + steps * p
  win_ref = torch.tensor(
    data[:, :, lo - L + p : lo + p], device=dev, dtype=dtype
  )
  eng_eager.full_refresh(win_ref)
  fd_out = fd.step(win_ref)
  r_fd = rel_err(fd_out, eng_eager.forecast())
  ok_fd = r_fd <= 1e-6
  fails += not ok_fd
  print(f"  graph full refresh output    : rel_err {r_fd:.3e}  "
        f"-> {'PASS' if ok_fd else 'FAIL'}")

  lo2 = L + (steps + 1) * p
  patch = torch.tensor(data[:, :, lo2 : lo2 + p], device=dev, dtype=dtype)
  got_e = eng_eager.step_patch(patch)
  got_g = runner.step(patch)
  r_post = rel_err(got_g, got_e)
  ok_post = r_post <= 1e-6
  fails += not ok_post
  print(f"  rolling step after refresh   : rel_err {r_post:.3e}  "
        f"-> {'PASS' if ok_post else 'FAIL'}")
  return fails


# --------------------------------------------------------------------- T4 ---


def run_t4(model, args, dev, dtype, th):
  """Eviction-machinery gate: ring wraparound vs a no-reuse reference engine.

  A rolling-vs-recompute assertion is mathematically unsound here (KV
  staleness under Pi-6 plus the position term of the affine post-RoPE
  qk-norm; see the module docstring), so this gate isolates what CAN be
  asserted: the wraparound / tagging / eviction bookkeeping.  The reference
  is a second engine with `capacity_slack` large enough that its ring never
  reuses a slot during the test; both engines consume the identical
  refresh + update stream, compute identical math over identical position
  tags, and differ only in physical slot layout.  Their outputs must agree
  up to floating-point reduction order.

  T4a (H=64) wraps the ring on every fast update (capacity == window);
  T4b (H=256) additionally exercises the horizon-scratch mark/rollback fork
  across the wrap boundary.
  """
  banner("T4  eviction machinery: ring wraparound vs no-reuse reference engine")
  B, V = 1, 2
  L = args.context_length
  p = model.input_patch_len
  N = L // p
  k = args.evict_steps
  data = make_windows(B, V, L + (k + 1) * p, seed0=23)
  win0 = torch.tensor(data[:, :, :L], device=dev, dtype=dtype)
  fails = 0
  for label, H in [("T4a", 64), ("T4b", 256)]:
    cfg = RollingConfig(
      context_length=L, horizon=H, full_refresh_every=0,
      batch_size=B, num_variates=V, device=dev, dtype=dtype,
    )
    cfg_ref = dataclasses.replace(cfg, capacity_slack=k + 8)
    eng = RollingTimesFM3Engine(model, cfg)
    ref = RollingTimesFM3Engine(model, cfg_ref)
    eng.full_refresh(win0)
    ref.full_refresh(win0)
    worst = 0.0
    for i in range(k):
      lo = L + i * p
      patch = torch.tensor(data[:, :, lo : lo + p], device=dev, dtype=dtype)
      out_a = eng.step_patch(patch)
      out_b = ref.step_patch(patch)
      worst = max(worst, rel_err(out_a, out_b))
    ok = worst <= th
    fails += not ok
    print(
      f"  {label} H={H:>3}  {k} evictions, ring {eng.cache.capacity} vs "
      f"{ref.cache.capacity} slots: rel_err {worst:.3e}  (th {th:.0e})  "
      f"-> {'PASS' if ok else 'FAIL'}"
    )
  return fails


# --------------------------------------------------------------------- D1 ---


def run_d1(model, args, dev, dtype):
  banner("D1  cache gap after eviction (measurement, not an assertion)")
  print("  rolling (frozen stats + frozen trend) vs decode() on the shifted")
  print("  raw window -- this is the quantity the theory bounds.")
  B, V, H = 1, 2, 64
  L = args.context_length
  p = model.input_patch_len
  k_max = args.evict_steps
  data = make_windows(B, V, L + (k_max + 1) * p, seed0=23)
  win0 = torch.tensor(data[:, :, :L], device=dev, dtype=dtype)
  cfg = RollingConfig(
    context_length=L, horizon=H, full_refresh_every=0,
    batch_size=B, num_variates=V, device=dev, dtype=dtype,
  )
  eng = RollingTimesFM3Engine(model, cfg)
  eng.full_refresh(win0)
  print(f"  {'age k':>6}  {'max|Y~-Y^F|':>13}  {'rel':>10}  {'MAE':>10}")
  for k in range(1, k_max + 1):
    lo = L + (k - 1) * p
    patch = torch.tensor(data[:, :, lo : lo + p], device=dev, dtype=dtype)
    roll = eng.step_patch(patch)
    cur = torch.tensor(data[:, :, k * p : L + k * p], device=dev, dtype=dtype)
    full = model.decode(target=cur, horizon=H, mask=None)
    gap = (roll.float() - full.float()).abs()
    print(
      f"  {k:>6}  {gap.max().item():>13.4e}  "
      f"{(gap.max() / full.float().abs().max().clamp_min(1e-9)).item():>10.3e}  "
      f"{gap.mean().item():>10.4e}"
    )

  # ---- survivor re-encode split (a measurement, NOT an assertion) ---------
  # The engine's cached survivors keep deep-layer K/V computed while the now-
  # evicted patches were still visible, so rolling cannot equal a recompute
  # over the survivors alone; and a recompute at REMAPPED positions 0..N-1
  # additionally shifts every RoPE angle, which the post-RoPE affine qk
  # RMSNorm / PerDimScale make observable (Pi-4 in rolling_engine.py).
  # Re-encoding the engine's own recorded tokens (frozen stats + detrend
  # baked in) at both position choices splits the gap:
  #   remapped 0..N-1        -> KV staleness + position term
  #   true absolute k..k+N-1 -> KV staleness alone (segment_pos supported
  #                             by upstream transformer_stack)
  N = L // p
  with torch.no_grad():
    got = model.output_head(eng.last_embedding)
    tok = eng.token_history  # [B, V, N, 2*(p+o)] resblock inputs, as encoded
    x = model.pre_transformer_resblock(tok)
    zmask = torch.zeros(B, V, N, dtype=torch.bool, device=dev)
    out_remap, _, _ = model.transformer_stack(x, zmask)
    ref_remap = model.output_head(out_remap[:, :, -1:])
    seg_pos = (
      torch.arange(k_max, k_max + N, dtype=torch.int32, device=dev)
      .unsqueeze(0)
      .expand(B, N)
    )
    out_abs, _, _ = model.transformer_stack(x, zmask, segment_pos=seg_pos)
    ref_abs = model.output_head(out_abs[:, :, -1:])
  print(f"  survivor re-encode at age k={k_max} (last-patch head logits):")
  print(
    f"    remapped positions 0..N-1  : rel {rel_err(got, ref_remap):.3e}"
    "   (KV staleness + position term)"
  )
  print(
    f"    true absolute k..k+N-1     : rel {rel_err(got, ref_abs):.3e}"
    "   (KV staleness alone)"
  )


# -------------------------------------------------------------------- main --


def main():
  ap = argparse.ArgumentParser(description=__doc__)
  ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
  ap.add_argument(
    "--dtype", default="float32", choices=["float32", "bfloat16"],
    help="bfloat16 is refused with an explanation (upstream forward is "
    "fp32-only); kept as a choice so the refusal is discoverable",
  )
  ap.add_argument("--ckpt", default=DEFAULT_CKPT)
  ap.add_argument(
    "--random_init", action="store_true",
    help="random-initialized default config instead of loading weights "
    "(weightless dry runs)",
  )
  ap.add_argument("--context_length", type=int, default=1024)
  ap.add_argument("--evict_steps", type=int, default=8)
  args = ap.parse_args()

  if args.dtype == "bfloat16":
    # Upstream limitation, not an engine bug: util.get_running_stats keeps
    # fp32 stats, so model._preprocess's revin output and masks_cat.float()
    # make the resblock input fp32, and torch 2.5.1 raises on
    # F.linear(fp32 input, bf16 weight).  Every T1/T2/D1 reference call would
    # die with a traceback before any PASS/FAIL line, so refuse up front.
    ap.error(
      "--dtype bfloat16 unsupported: the vendored TimesFM-3.0 forward is "
      "fp32-only (fp32 resblock input x bf16 nn.Linear weight raises in "
      "torch 2.5.1), so the reference side of every gate crashes; the bf16 "
      "contract threshold cannot be exercised without modifying upstream. "
      "Run with --dtype float32."
    )

  dev = args.device
  dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
  th = 1e-5 if dtype == torch.float32 else 2e-2

  model = load_model(args, dev, dtype)
  print(
    f"model: layers={len(model.transformer_stack.layers)} "
    f"p={model.input_patch_len} o={model.output_patch_len} "
    f"nq={model.num_quantiles} stitching={model.use_stitching} "
    f"detrend={model.use_linear_detrending} cpm={model.use_iterative_cpm_revin} "
    f"frozen_stats={model.use_frozen_running_stats}"
  )
  print(f"device={dev}  dtype={args.dtype}  L={args.context_length}  th={th:.0e}")

  fails = 0
  fails += run_t1(model, args, dev, dtype, th, horizon=64, label="T1a")
  fails += run_t1(model, args, dev, dtype, th, horizon=256, label="T1b")
  fails += run_t2(model, args, dev, dtype, th)
  fails += run_t3(model, args, dev, dtype)
  fails += run_t4(model, args, dev, dtype, th)
  run_d1(model, args, dev, dtype)

  print(f"\n{'=' * 68}")
  print(f"RESULT: {'ALL GATES PASS' if fails == 0 else f'{fails} GATE(S) FAILED'}")
  print(f"{'=' * 68}\n")
  return 1 if fails else 0


if __name__ == "__main__":
  sys.exit(main())
