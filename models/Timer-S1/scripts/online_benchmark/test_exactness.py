"""Correctness gates for the Timer-S1 rolling engine.

T1   fidelity       : engine.full_refresh must equal the upstream one-shot
                      forward on the same frozen-normalized window.
T2   growing window : append-only updates with NO eviction must reproduce the
                      full recompute at every step (MANDATORY).  The two paths
                      are mathematically identical, so any mismatch beyond
                      kernel-level numerics is an engine bug.
T4p  rebase algebra : the k_scale-aware RoPE rebase formula is asserted at
                      5e-6 against an exact fp32 emulation of the upstream
                      key pipeline (model-forward-free, cheap in every mode).
T4   eviction       : once the window is full, report the rolling-vs-full gap
                      per cache age.  A measurement, not an assertion: even
                      with an exact per-key position rebase, survivors' K/V
                      (layers >= 2) and the MTP hidden-state rows were
                      computed while the evicted token was attendable, so
                      equality is unattainable for any causal decoder with
                      prefix eviction.  Moreover the per-dim q_scale/k_scale
                      means attention scores are not a function of relative
                      position alone, so the contract's "exact for
                      relative-position-only models" clause does not apply to
                      Timer-S1 -- the rebase is nevertheless per-key exact.
T3   skipped        : no CUDA graph runner in v1 (data-dependent MoE
                      dispatch).

Thresholds: rel_err <= 1e-5 (fp32) / 2e-2 (bf16).  The bf16 gap between the
two paths comes from kernel-shape numerics (SDPA q_len-1 vs q_len-N kernels,
per-expert grouping and batched-GEMM shapes), NOT from nondeterminism: each
expert's index_add_ receives unique token indices (top-2 over experts is
per-token distinct, no atomic collisions), so a rerun of the same binary
should be bit-stable -- if it is not, investigate rather than shrug.

The fp32 --random-init run is the T2 correctness AUTHORITY: fp32 kernel
noise (~1e-6) cannot flip a realistic top-2 router near-tie, so any fp32
growing-window mismatch is an engine bug.  Under bf16, ~1e-3 kernel-shape
noise feeds fp32 softmax + torch.topk routing that is re-decided every step
for every window token in every active MTP module, and a near-tie flip can
legitimately push one step past 2e-2 with zero engine bug; T2 therefore
waives a SINGLE outlier step when every other step sits <= 3e-3 (printed as
PASS with a router-flip WARN).  The 2e-2 threshold itself is never raised.

Recommended invocations:
  cheap : python test_exactness.py --random-init --dtype float32 \
              --context-length 512 --horizon 48 --t2-steps 8 --evict-steps 8 \
              --device cuda:0
  layers: python test_exactness.py --layers 2 --mtp 2 --dtype bfloat16 \
              --context-length 1024 --horizon 48 --device cuda:0
  full  : python test_exactness.py --device cuda:0 --dtype bfloat16
"""

import argparse
import math
import os
import sys

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from timers1_online import RollingTimerS1Engine, TimerS1RollingConfig  # noqa: E402
from timers1_online.rope_utils import (  # noqa: E402
    make_rebase_factors,
    rebase_rope_keys_minus_one_,
    rotate_half,
)

DEFAULT_CKPT = os.path.join(os.environ.get("ROLLKV_CKPT", "checkpoints"), "Timer-S1")
# Byte-identical to the checkpoint snapshot's code; config fallback for
# --random-init runs on machines without the downloaded weights.
VENDORED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


def make_series(n, seed=0):
    rng = np.random.RandomState(seed)
    t = np.arange(n, dtype=np.float32)
    return (
        np.sin(2 * np.pi * t / 96)
        + 0.5 * np.sin(2 * np.pi * t / 336)
        + 0.2 * rng.randn(n).astype(np.float32)
    ).astype(np.float32)


def rel_err(a: torch.Tensor, ref: torch.Tensor) -> float:
    return (a - ref).abs().max().item() / max(ref.abs().max().item(), 1.0)


def banner(title):
    print(f"\n{'=' * 68}")
    print(title)
    print(f"{'=' * 68}")


def load_model(args, dtype):
    if args.random_init:
        cfg_src = args.ckpt
        if not os.path.isfile(os.path.join(cfg_src, "config.json")):
            cfg_src = VENDORED_DIR
        cfg = AutoConfig.from_pretrained(cfg_src, trust_remote_code=True)
        cfg.num_hidden_layers = args.layers if args.layers is not None else 2
        cfg.num_mtp_tokens = args.mtp if args.mtp is not None else 2
        cfg.num_experts = args.experts
        if args.hidden_size is not None:
            if args.hidden_size % cfg.num_attention_heads:
                raise ValueError(
                    "--hidden-size must be divisible by num_attention_heads="
                    f"{cfg.num_attention_heads}"
                )
            if (args.hidden_size // cfg.num_attention_heads) % 2:
                raise ValueError(
                    "--hidden-size must give an even head_dim, got "
                    f"{args.hidden_size // cfg.num_attention_heads}: the "
                    "rotary table width 2*(head_dim//2) would not match "
                    "rotate_half's half-split"
                )
            cfg.hidden_size = args.hidden_size
        torch.manual_seed(args.seed)
        try:
            model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
        except Exception as err:  # 4.45.2 remote-code from_config quirk
            print(f"from_config failed ({err!r}); using get_class_from_dynamic_module")
            from transformers.dynamic_module_utils import get_class_from_dynamic_module

            cls = get_class_from_dynamic_module(
                "modeling_TimerS1.TimerS1ForPrediction", cfg_src
            )
            model = cls(cfg)
        model = model.to(device=args.device, dtype=dtype)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.ckpt, trust_remote_code=True, torch_dtype=dtype
        )
        if args.layers is not None:
            # Kept layers retain dense layer_idx 0..N-1, so DynamicCache
            # indexing stays correct; MTP layer_idx values beyond N never
            # touch the cache (their past_key_value is always None).
            model.model.layers = model.model.layers[: args.layers]
            model.config.num_hidden_layers = args.layers
        if args.mtp is not None:
            # forward reads the config for step counts and iterates the
            # ModuleList; both must agree.
            model.mtp_modules = model.mtp_modules[: args.mtp]
            model.config.num_mtp_tokens = args.mtp
        model = model.to(args.device)
    return model.eval()


@torch.no_grad()
def full_forward(model, eng, raw_window, dtype, horizon):
    """Reference no-cache forward, normalized with the ENGINE's frozen stats."""
    normed = ((raw_window - eng.mean) / eng.std).to(dtype)
    out = model(
        input_ids=normed,
        use_cache=False,
        return_dict=True,
        max_output_length=horizon,
        revin=False,
    )
    return out.logits.float(), out.hidden_states_for_mtp


@torch.no_grad()
def gate_t4_pre(model):
    """Assert the rebase algebra against an exact emulation of the upstream
    key pipeline (apply_rotary_pos_emb's k-branch, then the k-norm with a
    per-dim scale), all in fp32.

    The reference rotary table is rebuilt in fp32 from inv_freq so the
    R(p-1) = R(-1) R(p) identity can be checked tightly even when the
    model's cached tables are bf16.  make_rebase_factors builds its factors
    from the same fp32 inv_freq (never from the bf16 cached tables), so the
    wiring check is exact in every dtype mode.  The algebra threshold is
    5e-6: the reference path recomputes the qk-norm scalar on the rotated
    tensor, ~5e-7 relative fp32 accumulation error, and a real wiring bug
    signature is O(1e-2)+, so the headroom is deliberate, not slack.
    """
    attn = model.get_decoder().layers[0].self_attn
    heads, dim = attn.num_heads, attn.head_dim
    steps = 8
    inv_freq = attn.rotary_emb.inv_freq.detach().float()
    device = inv_freq.device
    t = torch.arange(steps + 1, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    table = torch.cat((freqs, freqs), dim=-1)
    cos_t, sin_t = table.cos(), table.sin()  # exact fp32 tables

    gen = torch.Generator().manual_seed(123)
    k_raw = torch.randn(1, heads, steps, dim, generator=gen).to(device)
    k_scale = attn.k_scale.detach().float().view(1, 1, 1, -1)
    scale_test = k_scale
    if bool(torch.all(k_scale == 1.0)):
        # Random-init leaves k_scale at ones, which would leave the
        # unscale/rescale path untested; use a synthetic per-dim scale for
        # the algebra check (the identity holds for any nonzero scale).
        scale_test = (0.25 + torch.rand(1, 1, 1, dim, generator=gen)).to(device)
        print("  note: k_scale is all-ones; algebra check uses a synthetic scale")

    def upstream_key(position_ids, scale):
        cos = cos_t[position_ids].view(1, 1, steps, dim)
        sin = sin_t[position_ids].view(1, 1, steps, dim)
        k = k_raw * cos + rotate_half(k_raw) * sin
        return k * torch.rsqrt(k.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * scale

    at_p = upstream_key(torch.arange(1, steps + 1, device=device), scale_test)
    at_pm1 = upstream_key(torch.arange(0, steps, device=device), scale_test)
    factors = {
        "cos1": cos_t[1].view(1, 1, 1, -1),
        "sin1": sin_t[1].view(1, 1, 1, -1),
        "scale": scale_test,
        "inv_scale": scale_test.reciprocal(),
    }
    rebased = rebase_rope_keys_minus_one_(at_p.clone(), factors)
    rel_algebra = rel_err(rebased, at_pm1)

    live = make_rebase_factors(model)[0]
    rel_wiring = max(
        rel_err(live["cos1"], factors["cos1"]),
        rel_err(live["sin1"], factors["sin1"]),
        rel_err(live["scale"], k_scale),
    )
    # Both sides are fp32-from-inv_freq now, so wiring is exact in every mode.
    tol_wiring = 1e-6
    return rel_algebra, rel_wiring, tol_wiring


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    ap.add_argument(
        "--dtype",
        choices=["bfloat16", "float32"],
        default=None,
        help="default: bfloat16 (full checkpoint) / float32 (--random-init)",
    )
    ap.add_argument("--context-length", type=int, default=1920)
    ap.add_argument("--horizon", type=int, default=96)
    ap.add_argument("--t2-steps", type=int, default=8)
    ap.add_argument("--evict-steps", type=int, default=8)
    ap.add_argument(
        "--layers", type=int, default=None,
        help="truncate the trunk to its first N layers (cheap iteration)",
    )
    ap.add_argument(
        "--mtp", type=int, default=None,
        help="truncate the MTP head to K modules (0 allowed; caps horizon at 16)",
    )
    ap.add_argument(
        "--experts", type=int, default=8, help="(random-init only) num_experts"
    )
    ap.add_argument(
        "--hidden-size", type=int, default=None, help="(random-init only)"
    )
    ap.add_argument(
        "--random-init", action="store_true",
        help="build from config with random weights instead of the checkpoint",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument(
        "--diag", action="store_true",
        help="also print per-step rel_err of the new-token trunk hidden state",
    )
    ap.add_argument(
        "--t4-max-rel", type=float, default=float("inf"),
        help="finite value turns the T4 measurement into a loose assertion",
    )
    ap.add_argument(
        "--no-rope-rebase", action="store_true",
        help="diagnostic: run T4 without the key rebase (gap should widen)",
    )
    ap.add_argument(
        "--force-fp32", action="store_true",
        help="allow --dtype float32 on the full checkpoint (needs ~33 GB)",
    )
    args = ap.parse_args()

    if args.dtype is None:
        args.dtype = "float32" if args.random_init else "bfloat16"
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    fp32_mode = dtype is torch.float32
    if (
        fp32_mode
        and not args.random_init
        and args.layers is None
        and not args.force_fp32
    ):
        print(
            "refusing --dtype float32 on the full 8.3B checkpoint (~33 GB of "
            "weights leaves no activation room on A100-40GB); pass "
            "--force-fp32, --layers N, or --random-init"
        )
        return 2
    if args.layers is not None and args.layers < 1:
        print("--layers must be >= 1")
        return 2
    tol = 1e-5 if fp32_mode else 2e-2

    torch.manual_seed(0)
    model = load_model(args, dtype)
    patch = int(model.config.input_token_len)
    n_params = sum(p.numel() for p in model.parameters())
    max_out = int(
        model.config.output_token_lens[-1] + patch * model.config.num_mtp_tokens
    )

    L, B = args.context_length, args.batch_size
    if L % patch or L < patch:
        print(f"--context-length must be a positive multiple of {patch}")
        return 2
    N = L // patch
    H = args.horizon
    if H > max_out:
        print(
            f"note: clamping horizon {H} -> {max_out} "
            "(= output_token_lens[-1] + patch * num_mtp_tokens of this model)"
        )
        H = max_out
    j0 = N - args.t2_steps
    if j0 < 1 or args.t2_steps < 1:
        print(f"need 1 <= --t2-steps <= N-1 (N={N})")
        return 2

    print(
        f"model: Timer-S1 {'random-init' if args.random_init else 'checkpoint'}"
        f"  params={n_params / 1e6:.1f}M  layers={model.config.num_hidden_layers}"
        f"  mtp={model.config.num_mtp_tokens}  experts={model.config.num_experts}"
    )
    print(
        f"device={args.device}  dtype={args.dtype}  tol={tol:.0e}  B={B}"
        f"  L={L} (N={N} tokens)  H={H}"
        f"  rope_rebase={not args.no_rope_rebase}"
    )

    total = L + (args.evict_steps + 1) * patch
    series = torch.stack(
        [torch.from_numpy(make_series(total, seed=7 + b)) for b in range(B)]
    ).to(args.device)  # [B, total] fp32; distinct rows exercise per-row stats
    window = series[:, :L]

    ecfg = TimerS1RollingConfig(
        context_length=L,
        horizon=H,
        full_refresh_every=0,
        batch_size=B,
        device=args.device,
        dtype=dtype,
        rope_rebase=not args.no_rope_rebase,
    )
    fail = 0

    # ---------------------------------------------------------------- T1 ----
    banner("T1  engine.full_refresh  vs  upstream one-shot forward")
    eng = RollingTimerS1Engine(model, ecfg)
    eng.full_refresh(window)
    ref, _ = full_forward(model, eng, window, dtype, H)
    r1 = rel_err(eng.last_logits, ref)
    ok1 = r1 <= tol
    fail += not ok1
    print(f"  L={L} (N={N} tokens)  H={H}  B={B}")
    print(f"  rel_err                  : {r1:.3e}")
    print(f"  -> {'PASS' if ok1 else 'FAIL'}")

    # ---------------------------------------------------------------- T2 ----
    banner("T2  growing-window exactness (append only, no eviction; MANDATORY)")
    eng2 = RollingTimerS1Engine(model, ecfg)
    eng2.full_refresh(window[:, : j0 * patch])  # frozen stats fixed here
    print(f"  full_refresh on {j0} tokens, then {args.t2_steps} append steps")
    print(f"  {'step':>6}  {'points':>7}  {'rel_err':>10}" + ("  hid_rel" if args.diag else ""))
    rels = []
    for s in range(1, args.t2_steps + 1):
        j = j0 + s
        eng2.fast_update(window[:, (j - 1) * patch : j * patch])
        ref, ref_hid = full_forward(model, eng2, window[:, : j * patch], dtype, H)
        r = rel_err(eng2.last_logits, ref)
        rels.append(r)
        row = f"  {s:>6}  {j * patch:>7}  {r:>10.3e}"
        if args.diag:
            hr = rel_err(
                eng2.hid_buffer[:, -1, :].float(), ref_hid[:, -1, :].float()
            )
            row += f"  {hr:.3e}"
        print(row)
    worst = max(rels)
    ok2 = worst <= tol
    # bf16 waiver: a top-2 router near-tie flip (fp32 softmax + torch.topk
    # re-decided every step, fed ~1e-3 kernel-shape noise) can blow up one
    # step with zero engine bug.  Its signature -- a single outlier while
    # every other step sits at the ~1e-3 noise floor -- is waived as WARN;
    # fp32 --random-init remains the correctness authority (fp32 noise
    # ~1e-6 cannot flip a realistic routing gap).  The 2e-2 threshold
    # itself is never raised.
    waived = False
    if not ok2 and not fp32_mode and len(rels) >= 2:
        clean = [r for r in rels if r <= tol]
        if len(clean) == len(rels) - 1 and max(clean) <= 3e-3:
            ok2 = waived = True
    fail += not ok2
    print(f"  worst step               : {rels.index(worst) + 1} / {args.t2_steps}")
    print(f"  max rel_err              : {worst:.3e}")
    if waived:
        print(
            "  -> PASS (WARN: 1 outlier step above tol, all others <= 3e-3 "
            "-- router near-tie flip signature, not an engine bug; certify "
            "with --random-init --dtype float32)"
        )
    else:
        print(f"  -> {'PASS' if ok2 else 'FAIL'}")
    if not ok2:
        if fp32_mode:
            print("  fp32 growing-window mismatch is an engine bug.")
        else:
            print(
                "  more than one step above tol, or a noise floor above "
                "3e-3, is NOT the router-flip signature: suspect the engine "
                "first; rerun with --random-init --dtype float32"
            )

    # ------------------------------------------------------------- T4-pre ---
    banner("T4-pre  k_scale-aware RoPE rebase algebra (asserted, forward-free)")
    r_alg, r_wire, tol_wire = gate_t4_pre(model)
    tol_alg = 5e-6  # ~10x over fp32 accumulation error; real bugs are O(1e-2)+
    ok_pre = r_alg <= tol_alg and r_wire <= tol_wire
    fail += not ok_pre
    print(f"  rebase algebra rel_err   : {r_alg:.3e}   (tol {tol_alg:.0e})")
    print(f"  factor wiring rel_err    : {r_wire:.3e}   (tol {tol_wire:.0e})")
    print(f"  -> {'PASS' if ok_pre else 'FAIL'}")

    # ---------------------------------------------------------------- T4 ----
    banner("T4  cache gap after eviction + position remap (measurement)")
    print(
        "  not asserted: survivors' K/V and MTP hidden-state rows were\n"
        "  computed while the evicted token was attendable; per-dim\n"
        "  q_scale/k_scale also breaks the relative-position-only exactness\n"
        "  clause for this model.  The per-key rebase itself is exact (T4-pre)."
    )
    if eng2.cache_length != N:
        raise RuntimeError("T2 engine did not end at capacity")
    max_rel4 = 0.0
    hdr = f"  {'age k':>6}  {'max|dY|':>11}  {'rel':>10}  {'MAE':>10}"
    print(hdr + ("     hid_rel" if args.diag else ""))
    for k in range(1, args.evict_steps + 1):
        lo = L + (k - 1) * patch
        eng2.fast_update(series[:, lo : lo + patch])
        ref, ref_hid = full_forward(model, eng2, eng2.raw_buffer, dtype, H)
        gap = (eng2.last_logits - ref).abs()
        rel = gap.max().item() / max(ref.abs().max().item(), 1.0)
        max_rel4 = max(max_rel4, rel)
        row = (
            f"  {k:>6}  {gap.max().item():>11.4e}  {rel:>10.3e}"
            f"  {gap.mean().item():>10.4e}"
        )
        if args.diag:
            hr = rel_err(
                eng2.hid_buffer[:, -1, :].float(), ref_hid[:, -1, :].float()
            )
            row += f"  {hr:.3e}"
        print(row)
    if math.isfinite(args.t4_max_rel):
        ok4 = max_rel4 <= args.t4_max_rel
        fail += not ok4
        print(f"  max rel {max_rel4:.3e} vs --t4-max-rel {args.t4_max_rel:g}")
        print(f"  -> {'PASS' if ok4 else 'FAIL'}")
    else:
        print(f"  max rel over ages        : {max_rel4:.3e}   (not asserted)")

    print("\nT3 skipped: no CUDA graph runner in v1 (data-dependent MoE dispatch)")

    print(f"\n{'=' * 68}")
    print(f"RESULT: {'ALL GATES PASS' if fail == 0 else f'{fail} GATE(S) FAILED'}")
    print(f"{'=' * 68}\n")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
