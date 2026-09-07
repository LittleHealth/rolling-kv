"""Patch-aligned rolling KV cache for Timer-S1 (thuml, 8.3B MoE, eager-only).

Timer-S1 maps every non-overlapping 16-point patch to one Transformer token.
Rolling therefore advances by exactly one patch: at capacity the oldest
cached token is evicted (with an exact RoPE + k_scale-aware key rebase, see
rope_utils) and only the new patch runs through the 24-layer trunk.  The
multi-token-prediction (MTP) head is architecturally uncacheable -- each of
its up-to-16 layers re-attends over the whole window from scratch -- so the
engine maintains the two tensors the upstream forward needs to rebuild it
exactly: ``full_input_ids`` (the frozen-stat normalized window) and
``full_hidden_states`` (the accumulated post-final-RMSNorm trunk states).

Why rolling matches full recompute before eviction (the T1/T2 argument):
``full_refresh`` is the upstream one-shot forward on the same normalized
tensor (an empty DynamicCache's first update returns its inputs unmodified;
positions and mask are identical).  For a growing window, causality makes
token t's trunk hidden state and K/V independent of later tokens, so the
cached entries and hidden-state buffer rows equal their full-recompute
counterparts; the MTP head is recomputed from scratch every step over
``full_input_ids`` plus the buffer, exactly as the no-cache path computes
it.  Residual divergence is kernel-level numerics only (SDPA q_len-1 vs
q_len-N kernels, per-expert grouping and batched-GEMM shape effects in the
MoE combine and patch embedding); it is shape effects, not nondeterminism --
each expert's index_add_ receives unique token indices (top-2 over experts
is per-token distinct), so a rerun of the same binary is run-to-run
deterministic.  After eviction,
equality is NOT claimed: survivors' K/V (layers >= 2) and hidden-state rows
were computed while the evicted token was still attendable; the exactness
gate reports that gap per cache age without asserting it.

Design notes:

* Eager-only v1.  The 32-expert top-2 router dispatches with per-expert
  ``torch.where``/``nonzero`` (data-dependent shapes; 32 host syncs x 24
  layers per step), which precludes CUDA graph capture without a static
  dispatch path.  Time-MoE's ``set_static_moe_dispatch`` is the precedent
  for future work; it is deliberately not ported in this round, and would
  never be added by editing the vendored modeling file.
* The upstream ``_get_usable_past_kv_length`` is exception-driven on
  transformers 4.45.2 (``get_max_cache_shape`` raises AttributeError and the
  code falls back to ``get_seq_length``) -- harmless and correct for
  DynamicCache.
* A ``full_refresh`` resets all three staleness channels at once: the
  survivor K/V entries, the MTP hidden-state buffer, and the frozen
  normalization statistics.
* ``full_refresh`` accepts any patch-multiple window in
  [patch, context_length], so a growing phase runs through the exact public
  API the T2 gate exercises online (deviation from Timer-HF, whose engine
  requires the full window up front).
* ``cfg.dtype`` must match the model's parameter dtype: the normalized
  window feeds straight into the patch embedding's Linear layers.
* Horizons beyond ``output_token_lens[-1] + input_token_len *
  num_mtp_tokens`` (272 for the released checkpoint) need the upstream
  quantile-median autoregressive generate loop (ts_generation_mixin.py) --
  future work, v1 is single-forward only.
"""

from __future__ import annotations

import dataclasses

import torch
from transformers import DynamicCache

from .cache_utils import get_layer_kv, num_layers, slice_cache
from .rope_utils import make_rebase_factors, rebase_rope_keys_minus_one_


@dataclasses.dataclass
class TimerS1RollingConfig:
    context_length: int = 11520
    horizon: int = 96
    # 0 = never.  For long bf16 steady-state runs set this non-zero: each
    # eviction re-quantizes the rebased keys into bf16 storage, a ~2e-3 *
    # sqrt(m) random walk over a key's m rebases (see rope_utils), and a
    # periodic refresh caps m.
    full_refresh_every: int = 0
    batch_size: int = 1
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    rope_rebase: bool = True


class RollingTimerS1Engine:
    """Online Timer-S1 forecaster with frozen-normalization rolling KV reuse."""

    def __init__(self, model, cfg: TimerS1RollingConfig):
        self.model = model
        self.cfg = cfg
        self.patch = int(model.config.input_token_len)
        self.max_output = int(
            model.config.output_token_lens[-1]
            + model.config.input_token_len * model.config.num_mtp_tokens
        )
        if cfg.context_length < self.patch or cfg.context_length % self.patch:
            raise ValueError(
                f"context_length must be a positive multiple of {self.patch}"
            )
        if not 1 <= cfg.horizon <= self.max_output:
            raise ValueError(
                f"horizon must be in [1, {self.max_output}]; longer horizons "
                "need the upstream quantile-median autoregressive generate "
                "loop (ts_generation_mixin.py) -- future work, v1 is "
                "single-forward only"
            )
        self.n_tokens = cfg.context_length // self.patch
        self.rebase_factors = make_rebase_factors(model) if cfg.rope_rebase else None
        self.raw_buffer: torch.Tensor | None = None       # [B, L_cur] raw fp32
        self.mean: torch.Tensor | None = None             # [B, 1] fp32, frozen
        self.std: torch.Tensor | None = None              # [B, 1] fp32, frozen
        self.normed_buffer: torch.Tensor | None = None    # [B, L_cur] cfg.dtype
        self.hid_buffer: torch.Tensor | None = None       # [B, T_cur, hidden]
        self.cache: DynamicCache | None = None
        self.last_logits: torch.Tensor | None = None      # [B, Q, H] fp32 normed
        self.last_prediction: torch.Tensor | None = None  # [B, Q, H] fp32 raw
        self.n_updates = 0
        self.model.eval()

    def _as_window(self, raw_window) -> torch.Tensor:
        value = torch.as_tensor(raw_window, device=self.cfg.device, dtype=torch.float32)
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if (
            value.ndim != 2
            or value.shape[0] != self.cfg.batch_size
            or not self.patch <= value.shape[1] <= self.cfg.context_length
            or value.shape[1] % self.patch
        ):
            raise ValueError(
                f"expected [{self.cfg.batch_size}, L] with L a multiple of "
                f"{self.patch} in [{self.patch}, {self.cfg.context_length}], "
                f"got {list(value.shape)}"
            )
        return value

    def _as_patch(self, new_patch) -> torch.Tensor:
        value = torch.as_tensor(new_patch, device=self.cfg.device, dtype=torch.float32)
        if value.ndim == 1:
            value = value.unsqueeze(0)
        if value.shape != (self.cfg.batch_size, self.patch):
            raise ValueError(
                f"expected [{self.cfg.batch_size}, {self.patch}], got {list(value.shape)}"
            )
        return value

    def _finish(self, result) -> torch.Tensor:
        # logits are [B, num_quantiles, horizon]; the frozen stats are [B, 1]
        # and must broadcast over both trailing dims -- unsqueeze is required
        # for batch sizes > 1 (the upstream in-forward revin denorm is
        # B==1-only, which is one reason the engine never uses revin=True).
        self.last_logits = result.logits.float()
        self.last_prediction = (
            self.last_logits * self.std.unsqueeze(-1) + self.mean.unsqueeze(-1)
        )
        return self.last_prediction

    @torch.no_grad()
    def full_refresh(self, raw_window) -> torch.Tensor:
        window = self._as_window(raw_window)
        self.raw_buffer = window.clone()
        # Unbiased std is the house convention (Timer-HF engine); both paths
        # of every gate share the engine's frozen stats, so the choice
        # cancels in all comparisons.
        self.mean = window.mean(dim=-1, keepdim=True)
        self.std = window.std(dim=-1, keepdim=True).clamp_min(1e-8)
        self.normed_buffer = ((window - self.mean) / self.std).to(self.cfg.dtype)
        # full_input_ids / full_hidden_states are omitted: the trunk covers
        # the whole window, so full_inputs_embeds == inputs_embeds and the
        # MTP head sees exactly the one-shot picture.
        result = self.model(
            input_ids=self.normed_buffer,
            past_key_values=DynamicCache(),
            use_cache=True,
            return_dict=True,
            max_output_length=self.cfg.horizon,
            revin=False,
        )
        self.cache = result.past_key_values
        self.hid_buffer = result.hidden_states_for_mtp
        return self._finish(result)

    @torch.no_grad()
    def fast_update(self, new_patch) -> torch.Tensor:
        if self.cache is None:
            raise RuntimeError("call full_refresh() before fast_update()")
        patch = self._as_patch(new_patch)
        patch_normed = ((patch - self.mean) / self.std).to(self.cfg.dtype)

        length = self.cache_length
        if not (
            self.hid_buffer.shape[1]
            == length
            == self.normed_buffer.shape[1] // self.patch
        ):
            raise RuntimeError(
                f"state desync: hid rows {self.hid_buffer.shape[1]}, cache "
                f"{length}, normed tokens {self.normed_buffer.shape[1] // self.patch}"
            )
        if length < self.n_tokens:
            # Growing phase: append only -- no slice, no rebase.
            past_cache = self.cache
            hs_prev = self.hid_buffer
            full_normed = torch.cat((self.normed_buffer, patch_normed), dim=-1)
            self.raw_buffer = torch.cat((self.raw_buffer, patch), dim=-1)
        elif length == self.n_tokens:
            # At capacity: drop the oldest token and rebase survivors to p-1.
            past_cache = slice_cache(self.cache, start=1)
            if self.rebase_factors is not None:
                for layer_idx in range(num_layers(past_cache)):
                    key, _ = get_layer_kv(past_cache, layer_idx)
                    rebase_rope_keys_minus_one_(key, self.rebase_factors[layer_idx])
            hs_prev = self.hid_buffer[:, 1:, :]
            full_normed = torch.cat(
                (self.normed_buffer[:, self.patch:], patch_normed), dim=-1
            )
            self.raw_buffer = torch.cat(
                (self.raw_buffer[:, self.patch:], patch), dim=-1
            )
        else:
            raise RuntimeError(
                f"cache length {length} exceeds window capacity {self.n_tokens}"
            )

        # No explicit position_ids: the model defaults to cache-relative
        # arange(past_len, past_len + 1), i.e. [T] while growing and [N-1]
        # over the rebased survivors at capacity -- correct in both branches.
        # (Absolute growing positions would IndexError on the model's
        # [:kv_seq_len] rotary-table slice.)  full_input_ids is re-embedded
        # inside the upstream forward; that recompute is unavoidable without
        # upstream edits (forward has no full_inputs_embeds parameter) and
        # costs about a quarter of one MTP layer.
        result = self.model(
            input_ids=patch_normed,
            past_key_values=past_cache,
            use_cache=True,
            return_dict=True,
            max_output_length=self.cfg.horizon,
            revin=False,
            full_input_ids=full_normed,
            full_hidden_states=hs_prev,
        )
        # result.past_key_values is past_cache itself, mutated in place by
        # the trunk (the MTP layers never touch it); rebind, never reuse the
        # pre-slice cache.
        self.cache = result.past_key_values
        # The model's internal mtp_hidden_states is overwritten by the MTP
        # modules, so the buffer is rebuilt from hs_prev plus the exported
        # new-token trunk hidden state, never read back from the model.
        self.hid_buffer = torch.cat((hs_prev, result.hidden_states_for_mtp), dim=1)
        self.normed_buffer = full_normed
        return self._finish(result)

    @torch.no_grad()
    def step_patch(self, new_patch) -> torch.Tensor:
        if self.raw_buffer is None:
            raise RuntimeError("call full_refresh() before step_patch()")
        self.n_updates += 1
        refresh = self.cfg.full_refresh_every
        if refresh > 0 and self.n_updates % refresh == 0:
            patch = self._as_patch(new_patch)
            # cat-then-trim handles both the growing and the full phase.
            window = torch.cat((self.raw_buffer, patch), dim=-1)
            return self.full_refresh(window[:, -self.cfg.context_length:])
        return self.fast_update(new_patch)

    def forecast(self) -> torch.Tensor:
        if self.last_prediction is None:
            raise RuntimeError("call full_refresh() first")
        return self.last_prediction

    @property
    def cache_length(self) -> int:
        return 0 if self.cache is None else int(self.cache.get_seq_length())
