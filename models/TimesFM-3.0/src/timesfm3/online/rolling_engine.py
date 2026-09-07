"""Rolling KV-cache engine for online forecasting with TimesFM-3.0.

This re-expresses `TimesFM3Torch.decode` as a sliding-window update loop over
a position-tagged ring buffer (rolling_cache.py).  The forward pass is written
out here rather than monkey-patched into `MultiHeadAttention` so that positions
and masks are fully under protocol control; every weight and every op is taken
from the loaded model, so the math is identical to upstream.

Protocol summary (patch-aligned, stride = p = input_patch_len)
--------------------------------------------------------------
Pi-2  token mapping : the window slides by exactly one patch.  Survivor patch
                      identity is exact, retokenization defect = 0.
Pi-3  preprocessing : two per-window quantities are **frozen at refresh time**
                      and reused by every fast update:
                      (a) the running RevIN stats.  TimesFM-3.0 normalizes
                          patch j by the causal *prefix* running stats over
                          patches 1..j; recomputing the prefix on the shifted
                          window would change every survivor's normalization
                          and collapse rolling back into full recompute.  The
                          stats therefore keep accumulating from the last
                          refresh.
                      (b) the linear-detrend coefficients (m, c) and their
                          time anchor.  `decode()` fits one least-squares line
                          to the whole window; refitting per update would
                          re-normalize every cached token.  New patches are
                          detrended with the frozen line evaluated at their
                          true offset (an *integer* point count since refresh,
                          so the eager path and the CUDA graph agree bit-for-
                          bit), and the same line is re-added to the forecast.
                      Drift from both is bounded by `full_refresh_every`.
Pi-4  position      : monotone absolute positions, never renumbered.  For
                      TimesFM-3.0 this is load-bearing: RoPE is followed by
                      elementwise-affine qk RMSNorm and PerDimScale
                      (transformer.py:282-290, both applied AFTER the
                      rotation).  The RMS scalar itself is rotation-invariant
                      (the rotation preserves per-pair norms), but the affine
                      weights and per-dim scales of a trained checkpoint are
                      not symmetric within a RoPE rotation pair (dims i and
                      i + head_dim/2), so q.k picks up a reflection term
                      proportional to (w_i - w_{i+half}) that depends on
                      theta_q + theta_k, i.e. on ABSOLUTE positions.  Exact
                      position-remap invariance therefore does NOT hold for
                      this architecture (it would only under uniform weights,
                      e.g. at random init); the engine never renumbers, its
                      absolute positions match upstream exactly, and the
                      eviction machinery is gated by T4 against a no-reuse
                      reference engine rather than against a recompute at
                      shifted positions.
Pi-5  mask          : position-tag driven (see rolling_cache.py); eviction
                      costs one integer update and moves no memory.
Pi-6  update        : survivors are reused verbatim; only the new patch is
                      encoded through all layers.
Pi-7  readout       : horizon <= stitching extract_len needs no extra tokens
                      (the forecast reads only the last context patch's
                      logits).  Longer horizons encode all scratch (CPM)
                      patches in ONE forward -- TimesFM-3.0 is not
                      autoregressive -- inside a mark/rollback fork, so
                      scratch tokens never pollute the history.
Pi-8  refresh       : re-runs the identical prefill path, hence eps_ref = 0
                      by construction.

Scope / assumptions
-------------------
* Target-only variates (no covariates).  All B series and V variates advance
  in lockstep; the cache's leading dim is B*V.
* The window is fully observed: no NaNs after caller-side cleaning, no padding
  masks, and `context_length % input_patch_len == 0`.  Online rolling always
  has a complete window by definition.
* Everything runs under `torch.no_grad()` (never `inference_mode`: the cache
  tensors must stay mutable).
* All hyperparameters and behavior flags are read off the LOADED model object
  (checkpoint config decides use_sdpa / rescale_logits / use_rope_var /
  stitching / CPM / frozen stats), never hardcoded.
"""

from __future__ import annotations

import dataclasses
import math

import torch
import torch.nn.functional as F

from .. import util
from .rolling_cache import RollingKVCache

revin = util.revin
update_running_stats = util.update_running_stats


@dataclasses.dataclass
class RollingConfig:
  context_length: int = 2048  # L, must be a multiple of the patch size
  horizon: int = 64  # H
  full_refresh_every: int = 0  # in patch-updates; 0 == never
  batch_size: int = 1  # B
  num_variates: int = 1  # V
  device: str = "cuda"
  dtype: torch.dtype = torch.float32
  # Extra ring slots beyond window + scratch.  Functionally inert (masking is
  # position-tag driven, so unused slots never influence attention); the T4
  # gate uses it to build a reference engine whose ring never reuses a slot,
  # isolating wraparound/tagging/eviction bookkeeping from the model math.
  capacity_slack: int = 0


class RollingTimesFM3Engine:
  """Online forecaster with a sliding KV cache over patch tokens."""

  def __init__(self, model, cfg: RollingConfig):
    self.model = model
    self.cfg = cfg
    model.eval()

    # ---- read every hyperparameter off the loaded model -------------------
    self.p = model.input_patch_len  # 32
    self.o = model.output_patch_len  # 64
    self.rolls = model.rolls  # o // p
    self.nq = model.num_quantiles  # 9
    self.median_q_idx = model.num_quantiles // 2
    self.value_clip = model.value_clip
    self.use_stitching = model.use_stitching
    self.extract_len = model._stitching_extract_len if model.use_stitching else None
    self.use_linear_detrending = model.use_linear_detrending
    self.use_iterative_cpm_revin = model.use_iterative_cpm_revin
    self.use_frozen_running_stats = model.use_frozen_running_stats

    self.layers = model.transformer_stack.layers
    self.n_layers = len(self.layers)
    t_cfg = model.transformer_config.transformer
    self.num_heads = t_cfg.num_heads
    self.d_model = t_cfg.model_dims
    self.head_dim = self.d_model // self.num_heads

    for layer in self.layers:
      if not layer.seq_attn.causal_attention:
        raise ValueError(
          "rolling engine requires causal sequence attention; this checkpoint "
          "was built with causal_attention=False"
        )
    param_dtype = next(model.parameters()).dtype
    if param_dtype != cfg.dtype:
      raise ValueError(
        f"cfg.dtype ({cfg.dtype}) must match the model parameter dtype "
        f"({param_dtype}); call model.to(dtype) first"
      )
    if cfg.context_length % self.p != 0:
      raise ValueError(
        f"context_length ({cfg.context_length}) must be a multiple of the "
        f"patch size ({self.p})"
      )
    if cfg.horizon <= 0:
      raise ValueError("horizon must be > 0")
    self.n_patches = cfg.context_length // self.p

    # ---- scratch sizing (mirrors decode()'s horizon padding) --------------
    # H <= extract_len (stitching) resp. H <= o (chunked) means the number of
    # forecast patches is 1, and decode()'s stitch/gather reads ONLY the last
    # context patch's logits: horizon tokens are causally invisible to it and
    # zero scratch tokens are needed.  Otherwise decode() appends
    # num_forecast_patches + rolls - 1 (stitching) resp. padded_horizon / p
    # (chunked) fully-masked CPM patches; the fork encodes exactly those.
    H = cfg.horizon
    if self.use_stitching:
      if H <= self.extract_len:
        self.nh_scratch = 0
      else:
        overlap = self.extract_len - self.p
        nf = max(math.ceil((H - overlap) / self.p), 1)
        self.nh_scratch = nf + self.rolls - 1
    else:
      if H <= self.o:
        self.nh_scratch = 0
      else:
        padded_horizon = H + (-H) % self.o
        self.nh_scratch = padded_horizon // self.p

    if cfg.capacity_slack < 0:
      raise ValueError(f"capacity_slack must be >= 0, got {cfg.capacity_slack}")
    capacity = self.n_patches + self.nh_scratch + cfg.capacity_slack
    B, V = cfg.batch_size, cfg.num_variates
    dev = cfg.device

    self.cache = RollingKVCache(
      num_layers=self.n_layers,
      batch_leading=B * V,
      capacity=capacity,
      window=self.n_patches,
      num_heads=self.num_heads,
      head_dim=self.head_dim,
      device=dev,
      dtype=cfg.dtype,
    )

    # ---- state buffers ----------------------------------------------------
    # Stats are kept in fp32, matching upstream get_running_stats' init.
    self.stat_n = torch.zeros(B, V, device=dev)
    self.stat_mu = torch.zeros(B, V, device=dev)
    self.stat_sigma = torch.zeros(B, V, device=dev)

    # Frozen linear-detrend state (Pi-3b).  t_offset counts POINTS elapsed
    # since the refresh anchor as a Python int; the refresh window's last
    # point is t = 0, matching decode()'s t_ctx = arange(-(L-1), 1) / L.
    self.trend_m = torch.zeros(B, V, 1, device=dev)
    self.trend_c = torch.zeros(B, V, 1, device=dev)
    self.trend_apply = torch.zeros(B, V, 1, dtype=torch.bool, device=dev)
    self.t_offset = 0

    self.raw_buffer: torch.Tensor | None = None  # [B, V, L] raw (pre-detrend)
    self.last_embedding: torch.Tensor | None = None  # [B, V, 1, d_model]
    self.last_mu: torch.Tensor | None = None  # [B, V] stats at newest patch
    self.last_sigma: torch.Tensor | None = None
    self._n_updates = 0

    # Rolling record of the resblock inputs of the live window, so the D1
    # survivor re-encode measurement can push the surviving tokens through the
    # upstream modules without duplicating preprocessing.  Frozen stats and
    # frozen detrend are baked in by construction (tokens are recorded as
    # encoded).  Scratch tokens never enter it.
    token_dim = 2 * (self.p + self.o)
    self.token_history = torch.zeros(
      B, V, self.n_patches, token_dim, dtype=cfg.dtype, device=dev
    )
    self._hist_len = 0

    # Constant resblock input of a CPM/horizon patch: values fully masked, so
    # the token is [0 x (p+o) | 1 x (p+o)] (see _build_tokens for the layout).
    self._scratch_token = torch.cat(
      [
        torch.zeros(self.p + self.o, device=dev, dtype=cfg.dtype),
        torch.ones(self.p + self.o, device=dev, dtype=cfg.dtype),
      ]
    ).view(1, 1, 1, token_dim)

    # Variate attention is time-invariant: positions arange(V), mask all-True.
    # The mask is passed explicitly (upstream passes a mask tensor; passing
    # None could select a different SDPA kernel).
    self._var_pos = torch.arange(V, dtype=torch.int32, device=dev).unsqueeze(0)
    self._var_mask = torch.ones(1, 1, V, V, dtype=torch.bool, device=dev)

    # Constants for the manual (use_sdpa=False) attention path, hoisted out of
    # the per-call bodies: a scalar -> CUDA tensor creation is a pageable H2D
    # copy, which both costs latency and aborts CUDA graph stream capture.
    # Values and dtype promotion match upstream transformer.py:365-369.
    self._attn_zero = torch.zeros((), device=dev)
    self._attn_neg = torch.full((), -1e9, device=dev)

  # ------------------------------------------------------------------------
  # Forward pass with the rolling cache
  # ------------------------------------------------------------------------

  def _attn_forward(self, layer_idx, attn, x, positions, mask, slots):
    """One sequence MultiHeadAttention forward over the ring buffer.

    Mirrors `timesfm3.transformer.MultiHeadAttention.forward` exactly
    (transformer.py:254-394): projections -> RoPE -> qk RMSNorm ->
    PerDimScale(query) -> value norm -> SDPA/manual attention -> out proj.
    Keys are cached *after* RoPE and key_ln, as upstream's DecodeCache path
    stores them (transformer.py:296-307).

    x: [B*V, n_new, d_model].
    """
    BV, n, _ = x.shape

    query = attn.query_proj(x).view(BV, n, attn.num_heads, attn.head_dim)
    key = attn.key_proj(x).view(BV, n, attn.num_heads, attn.head_dim)
    value = attn.value_proj(x).view(BV, n, attn.num_heads, attn.head_dim)

    if attn.rotary_position_embedding is not None:
      pos = positions[None, :]  # broadcasts over the leading dim
      query = attn.rotary_position_embedding(query, pos)
      key = attn.rotary_position_embedding(key, pos)

    if attn.query_ln is not None:
      query = attn.query_ln(query)
    if attn.key_ln is not None:
      key = attn.key_ln(key)
    if attn.per_dim_scale is not None:
      query = attn.per_dim_scale(query)
    if attn.value_ln is not None:
      value = attn.value_ln(value)

    self.cache.write_layer(layer_idx, slots, key, value)

    k_all = self.cache.key[layer_idx]  # [B*V, C, H, D]
    v_all = self.cache.value[layer_idx]

    # RoPE's sin/cos are built in fp32, so query comes out of the rotation in
    # fp32 even for a bf16 model.  Keys/values were cast on the way into the
    # cache, so bring the query back to the cache dtype before attending.
    query = query.to(k_all.dtype)

    q_t = query.transpose(1, 2)  # [B*V, H, n, D]
    k_t = k_all.transpose(1, 2)
    v_t = v_all.transpose(1, 2)
    attn_mask = mask.expand(BV, attn.num_heads, -1, -1)

    if attn.use_sdpa:
      # Upstream scale rule (transformer.py:343-360): rescale_logits=True is
      # Flax MEA=False (net scale 1.0); rescale_logits=False is MEA=True
      # (query pre-multiplied by sqrt(d), no internal division).
      attn_scale = 1.0 if attn.rescale_logits else math.sqrt(attn.head_dim)
      out = F.scaled_dot_product_attention(
        q_t, k_t, v_t, attn_mask=attn_mask, scale=attn_scale
      )
    else:
      # Manual path, verbatim from transformer.py:361-385 (mask constants
      # hoisted to __init__ buffers; values identical).
      float_mask = torch.where(attn_mask, self._attn_zero, self._attn_neg)
      q_t = q_t * math.sqrt(attn.head_dim)
      if attn.rescale_logits:
        attn_logits = (
          torch.matmul(q_t, k_t.transpose(-2, -1)) / math.sqrt(attn.head_dim)
          + float_mask
        )
      else:
        attn_logits = torch.matmul(q_t, k_t.transpose(-2, -1)) + float_mask
      out = torch.matmul(F.softmax(attn_logits, dim=-1), v_t)

    out = out.transpose(1, 2).contiguous().view(BV, n, attn.in_features)
    return attn.out_proj(out)

  def _var_attn_forward(self, attn, x_var):
    """Variate attention: non-causal, cache-free, time-invariant.

    Mirrors the upstream var_attn call (transformer.py:516-528 feeding
    MultiHeadAttention with decode_cache=None and an all-False patch mask):
    positions arange(V), mask all-True.  It must run even for V == 1 --
    softmax over self still contributes through out_proj and the residual.

    x_var: [B*n_new, V, d_model].
    """
    Bn, v, _ = x_var.shape

    query = attn.query_proj(x_var).view(Bn, v, attn.num_heads, attn.head_dim)
    key = attn.key_proj(x_var).view(Bn, v, attn.num_heads, attn.head_dim)
    value = attn.value_proj(x_var).view(Bn, v, attn.num_heads, attn.head_dim)

    if attn.rotary_position_embedding is not None:
      query = attn.rotary_position_embedding(query, self._var_pos)
      key = attn.rotary_position_embedding(key, self._var_pos)

    if attn.query_ln is not None:
      query = attn.query_ln(query)
    if attn.key_ln is not None:
      key = attn.key_ln(key)
    if attn.per_dim_scale is not None:
      query = attn.per_dim_scale(query)
    if attn.value_ln is not None:
      value = attn.value_ln(value)

    q_t = query.transpose(1, 2)
    k_t = key.transpose(1, 2)
    v_t = value.transpose(1, 2)
    attn_mask = self._var_mask.expand(Bn, attn.num_heads, -1, -1)

    if attn.use_sdpa:
      attn_scale = 1.0 if attn.rescale_logits else math.sqrt(attn.head_dim)
      out = F.scaled_dot_product_attention(
        q_t, k_t, v_t, attn_mask=attn_mask, scale=attn_scale
      )
    else:
      float_mask = torch.where(attn_mask, self._attn_zero, self._attn_neg)
      q_t = q_t * math.sqrt(attn.head_dim)
      if attn.rescale_logits:
        attn_logits = (
          torch.matmul(q_t, k_t.transpose(-2, -1)) / math.sqrt(attn.head_dim)
          + float_mask
        )
      else:
        attn_logits = torch.matmul(q_t, k_t.transpose(-2, -1)) + float_mask
      out = torch.matmul(F.softmax(attn_logits, dim=-1), v_t)

    out = out.transpose(1, 2).contiguous().view(Bn, v, attn.in_features)
    return attn.out_proj(out)

  def _layer_forward(self, layer_idx, layer, x, positions, mask, slots):
    """Mirrors `timesfm3.transformer.MixingTransformer.forward` structurally.

    x: [B, V, n_new, d_model].
    """
    B, V, n, d = x.shape

    # --- Sequence attention ---
    seq_in = layer.pre_seq_attn_ln(x).reshape(B * V, n, d)
    seq_out = self._attn_forward(
      layer_idx, layer.seq_attn, seq_in, positions, mask, slots
    )
    h1 = layer.post_seq_attn_ln(seq_out.view(B, V, n, d)) + x

    # --- Variate attention ---
    if layer.use_variate_attention:
      var_in = layer.pre_var_attn_ln(h1).permute(0, 2, 1, 3).reshape(B * n, V, d)
      var_out = self._var_attn_forward(layer.var_attn, var_in)
      var_out = var_out.view(B, n, V, d).permute(0, 2, 1, 3)
      h2 = layer.post_var_attn_ln(var_out) + h1
    else:
      h2 = h1

    # --- FeedForward ---
    ff_out = layer.ff1(layer.activation(layer.ff0(layer.pre_ff_ln(h2))))
    return layer.post_ff_ln(ff_out) + h2

  def _build_tokens(self, normed_patches):
    """Assemble the 192-dim resblock input of a target-only context patch.

    Layout per model.py:236-251 with patch_is_target=True and no missing
    points: [ normed values (p) | future-covariate values, fully masked to
    zeros (o) | value mask = zeros (p) | future-covariate mask = ones (o) ].

    normed_patches: [B, V, n_new, p] -> [B, V, n_new, 2*(p+o)].
    """
    B, V, n, _ = normed_patches.shape
    vals_fcov = normed_patches.new_zeros(B, V, n, self.o)
    mask_vals = normed_patches.new_zeros(B, V, n, self.p)
    mask_fcov = normed_patches.new_ones(B, V, n, self.o)
    return torch.cat([normed_patches, vals_fcov, mask_vals, mask_fcov], dim=-1)

  def _encode_at(self, tokens, slots, positions, window_anchor=None):
    """Encode into the given slots. Touches no Python-side pointer state.

    Split out from `_encode` so the CUDA-graph runner can drive it with
    fixed-address buffers (see graph_runner.py): everything here is either a
    tensor op or a compile-time constant, so the whole body is capturable.
    """
    # Tag before attending so the mask already sees the new tokens (upstream
    # writes into the cache and then attends over the whole cache).
    self.cache.slot_pos.index_copy_(0, slots, positions)
    mask = self.cache.build_mask(positions, window_anchor)

    x = self.model.pre_transformer_resblock(tokens)
    for i, layer in enumerate(self.layers):
      x = self._layer_forward(i, layer, x, positions, mask, slots)
    return x

  def _encode(self, tokens, window_anchor=None):
    """Encode new tokens into the cache and return their final embeddings.

    tokens: [B, V, n_new, 2*(p+o)] resblock inputs.
    Returns [B, V, n_new, d_model].
    """
    n_new = tokens.shape[2]
    slots, positions = self.cache.reserve(n_new)
    x = self._encode_at(tokens, slots, positions, window_anchor)
    self.cache.advance(n_new)
    return x

  # ------------------------------------------------------------------------
  # Normalization (Pi-3a): running prefix stats, frozen at encode time
  # ------------------------------------------------------------------------

  def _advance_stats(self, patches, masks):
    """Fold `n_new` patches into the running stats, one patch at a time.

    Uses upstream `util.update_running_stats` on same-dtype inputs so the
    fold is bit-identical to `util.get_running_stats` inside `forward()`.

    patches / masks: [B, V, n_new, p].  Returns per-patch (mu, sigma), each
    [B, V, n_new].
    """
    mus, sigmas = [], []
    n, mu, sigma = self.stat_n, self.stat_mu, self.stat_sigma
    for i in range(patches.shape[2]):
      n, mu, sigma = update_running_stats(
        n, mu, sigma, patches[:, :, i], masks[:, :, i]
      )
      mus.append(mu)
      sigmas.append(sigma)
    self.stat_n, self.stat_mu, self.stat_sigma = n, mu, sigma
    return torch.stack(mus, dim=2), torch.stack(sigmas, dim=2)

  # ------------------------------------------------------------------------
  # Detrending (Pi-3b): whole-window fit at refresh, frozen afterwards
  # ------------------------------------------------------------------------

  def _fit_trend(self, ctx_vals):
    """Whole-window least-squares detrend, verbatim from model.py:472-511.

    The all-True `valid` mask and the `torch.where(valid, ...)` forms are
    kept even though the window is fully observed, so every reduction is
    bit-identical to upstream `decode()`.

    ctx_vals: [B, V, L] raw values.
    Returns (m_trend, c_trend, apply_detrend, out_vals) with the first three
    shaped [B, V, 1] and out_vals = where(apply, detrended, raw).
    """
    B, V, L = ctx_vals.shape
    device = ctx_vals.device
    t_ctx = torch.arange(-(L - 1), 1, dtype=torch.float32, device=device)
    t_norm = t_ctx[None, None, :] / L

    valid = torch.ones(B, V, L, dtype=torch.bool, device=device)
    n_v = valid.float().sum(dim=-1, keepdim=True)
    sum_t = torch.where(valid, t_norm, 0.0).sum(dim=-1, keepdim=True)
    sum_t2 = torch.where(valid, t_norm**2, 0.0).sum(dim=-1, keepdim=True)
    sum_y = torch.where(valid, ctx_vals, 0.0).sum(dim=-1, keepdim=True)
    sum_ty = torch.where(valid, t_norm * ctx_vals, 0.0).sum(dim=-1, keepdim=True)

    det = n_v * sum_t2 - sum_t**2
    safe_det = torch.where(det == 0.0, 1.0, det)
    m_trend = torch.where(det == 0.0, 0.0, (n_v * sum_ty - sum_t * sum_y) / safe_det)
    c_trend = torch.where(
      det == 0.0,
      torch.where(n_v > 0, sum_y / torch.clamp_min(n_v, 1.0), 0.0),
      (sum_y - m_trend * sum_t) / torch.clamp_min(n_v, 1.0),
    )

    detrended = ctx_vals - (m_trend * t_norm + c_trend)

    mean_y = sum_y / torch.clamp_min(n_v, 1.0)
    sum_y2 = torch.where(valid, ctx_vals**2, 0.0).sum(dim=-1, keepdim=True)
    var_orig = torch.clamp_min(sum_y2 / torch.clamp_min(n_v, 1.0) - mean_y**2, 0.0)
    std_orig = torch.sqrt(var_orig)

    sum_yd = torch.where(valid, detrended, 0.0).sum(dim=-1, keepdim=True)
    mean_yd = sum_yd / torch.clamp_min(n_v, 1.0)
    sum_yd2 = torch.where(valid, detrended**2, 0.0).sum(dim=-1, keepdim=True)
    var_det = torch.clamp_min(sum_yd2 / torch.clamp_min(n_v, 1.0) - mean_yd**2, 0.0)
    std_det = torch.sqrt(var_det)

    apply_detrend = std_det < self.model.linear_detrending_threshold * std_orig
    out_vals = torch.where(apply_detrend, detrended, ctx_vals)
    return m_trend, c_trend, apply_detrend, out_vals

  # ------------------------------------------------------------------------
  # Ingestion pipeline (shared by refresh / fast update / append)
  # ------------------------------------------------------------------------

  def _record_tokens(self, tokens):
    """Append committed resblock inputs to the rolling token history."""
    n_new = tokens.shape[2]
    W = self.n_patches
    if n_new >= W:
      self.token_history.copy_(tokens[:, :, -W:])
      self._hist_len = W
    elif self._hist_len + n_new <= W:
      self.token_history[:, :, self._hist_len : self._hist_len + n_new] = tokens
      self._hist_len += n_new
    else:
      keep = W - n_new
      self.token_history[:, :, :keep] = self.token_history[
        :, :, self._hist_len - keep : self._hist_len
      ].clone()
      self.token_history[:, :, keep:] = tokens
      self._hist_len = W

  def _ingest(self, patches):
    """Clamp -> stats -> revin -> tokens -> encode, mirroring `forward()`.

    patches: [B, V, n_new, p], already detrended where applicable (decode()
    detrends BEFORE forward() clamps, model.py:472-511 then 286-287).
    Returns the per-patch final logits [B, V, n_new, o, nq].
    """
    x = torch.clamp(
      torch.nan_to_num(patches, nan=0.0), -self.value_clip, self.value_clip
    )
    masks = torch.zeros_like(x, dtype=torch.bool)

    mu, sigma = self._advance_stats(x, masks)
    # Stats are fp32 (Welford in bf16 is not accurate enough); cast back so
    # the resblock sees the model's own dtype.
    normed = revin(x, mu, sigma, reverse=False).to(self.cfg.dtype)
    tokens = self._build_tokens(normed)
    self._record_tokens(tokens)

    emb = self._encode(tokens)
    self.last_embedding = emb[:, :, -1:, :]
    self.last_mu, self.last_sigma = mu[..., -1], sigma[..., -1]
    return self._readout_patches(emb, mu, sigma)

  # ------------------------------------------------------------------------
  # Readout
  # ------------------------------------------------------------------------

  def _readout_patches(self, emb, mu, sigma):
    """Output head + reverse revin + clamp (model.py:324, 344-351).

    emb: [B, V, n, d_model]; mu / sigma: [B, V, n].
    Returns [B, V, n, o, nq].
    """
    B, V, n, _ = emb.shape
    raw = self.model.output_head(emb)
    out = revin(raw, mu, sigma, reverse=True)
    out = torch.clamp(out, -self.value_clip, self.value_clip)
    return out.view(B, V, n, self.o, self.nq)

  def _cpm_refine_scratch(self, raw_scratch_logits):
    """Restarted CPM iterative RevIN refinement for the scratch patches.

    Upstream (`cpm_revin_refine.cpm_iterative_revin_refine`) loops over ALL
    patches; every non-CPM iteration fully overwrites the carry with the
    actual running stats, resets block_offset to 0, and refreshes the anchor.
    Restarting the loop at the first scratch patch with

      carry        = running stats at the last context patch,
      anchor       = clamp(revin(median raw logits of the last context patch)),
      block_offset = 0,

    is therefore exactly equivalent to upstream's full loop (its state after
    the last context iteration is precisely this).

    raw_scratch_logits: [B, V, nh, o*nq] output-head logits in NORMALIZED
    space (before reverse revin), as CPM refine expects.
    Returns (refined_mu, refined_sigma), each [B, V, nh].
    """
    B, V, nh, _ = raw_scratch_logits.shape
    device = raw_scratch_logits.device
    rolls, p, nq = self.rolls, self.p, self.nq

    median_scratch = raw_scratch_logits.reshape(B, V, nh, rolls, p, nq)[
      :, :, :, :, :, self.median_q_idx
    ]

    # Anchor seed: at the last context patch upstream sets
    # anchor = clamp(revin(median logits, actual stats)), cpm_revin_refine.py
    # :127-136 with should_update_anchor True.
    raw_last = self.model.output_head(self.last_embedding)  # [B, V, 1, o*nq]
    median_last = raw_last.reshape(B, V, rolls, p, nq)[..., self.median_q_idx]
    anchor = revin(median_last, self.last_mu, self.last_sigma, reverse=True)
    anchor = torch.clamp(anchor, -self.value_clip, self.value_clip)

    carry_n, carry_mu, carry_sigma = self.stat_n, self.stat_mu, self.stat_sigma
    block_offset = torch.zeros(B, dtype=torch.long, device=device)
    step_masks = torch.zeros(B, V, p, dtype=torch.bool, device=device)
    rolls_range = torch.arange(rolls, device=device)

    mu_list, sigma_list = [], []
    for i in range(nh):
      # is_cpm is True at every scratch patch, so upstream's where(is_cpm,..)
      # selections all reduce to the CPM branch (cpm_revin_refine.py:96-141).
      offset_onehot = torch.eq(
        rolls_range.unsqueeze(0), block_offset.unsqueeze(1)
      ).float()
      predicted_values_step = torch.einsum(
        "br,bvrp->bvp", offset_onehot, anchor
      )
      carry_n, carry_mu, carry_sigma = update_running_stats(
        carry_n, carry_mu, carry_sigma, predicted_values_step, step_masks
      )

      block_offset = (block_offset + 1) % rolls
      should_update_anchor = torch.eq(block_offset, 0)

      step_predicted = revin(
        median_scratch[:, :, i], carry_mu, carry_sigma, reverse=True
      )
      step_predicted = torch.clamp(
        step_predicted, -self.value_clip, self.value_clip
      )
      anchor = torch.where(
        should_update_anchor.view(B, 1, 1, 1), step_predicted, anchor
      )

      mu_list.append(carry_mu)
      sigma_list.append(carry_sigma)

    return torch.stack(mu_list, dim=2), torch.stack(sigma_list, dim=2)

  # ------------------------------------------------------------------------
  # Core operations
  # ------------------------------------------------------------------------

  def _to_bvx(self, x, last_dim):
    """Move to device/dtype and unsqueeze up to [B, V, last_dim]."""
    x = x.to(device=self.cfg.device, dtype=self.cfg.dtype)
    while x.dim() < 3:
      x = x.unsqueeze(0)
    B, V, n = x.shape
    if B != self.cfg.batch_size or V != self.cfg.num_variates or n != last_dim:
      raise ValueError(
        f"expected [{self.cfg.batch_size}, {self.cfg.num_variates}, "
        f"{last_dim}], got {tuple(x.shape)}"
      )
    return x

  @torch.no_grad()
  def full_refresh(self, raw_window: torch.Tensor) -> None:
    """Re-encode the whole window from scratch (Pi-8). eps_ref = 0.

    raw_window: [B, V, L] raw values ([V, L] / [L] are unsqueezed).
    """
    raw_window = self._to_bvx(raw_window, self.cfg.context_length)
    B, V, L = raw_window.shape

    self.raw_buffer = raw_window.clone()
    self.cache.reset()
    self.stat_n = torch.zeros(B, V, device=self.cfg.device)
    self.stat_mu = torch.zeros(B, V, device=self.cfg.device)
    self.stat_sigma = torch.zeros(B, V, device=self.cfg.device)
    self._hist_len = 0

    if self.use_linear_detrending:
      m_trend, c_trend, apply_detrend, ctx = self._fit_trend(raw_window)
      self.trend_m.copy_(m_trend)
      self.trend_c.copy_(c_trend)
      self.trend_apply.copy_(apply_detrend)
    else:
      self.trend_m.zero_()
      self.trend_c.zero_()
      self.trend_apply.fill_(False)
      ctx = raw_window
    self.t_offset = 0

    self._ingest(ctx.view(B, V, self.n_patches, self.p))

  @torch.no_grad()
  def append_patches(self, raw_patches: torch.Tensor) -> torch.Tensor:
    """Append patches to a growing window with NO eviction (the T2 path).

    Reproduces `model.forward`'s per-patch logits on the growing prefix
    exactly: no detrend, no CPM, no trend re-add.  This is the instrument
    for the growing-window exactness gate, not part of the sliding protocol
    (it does not touch the detrend state).

    raw_patches: [B, V, n_new, p] raw values.
    Returns [B, V, n_new, o, nq] final logits.
    """
    x = raw_patches.to(device=self.cfg.device, dtype=self.cfg.dtype)
    if x.dim() != 4 or x.shape[-1] != self.p:
      raise ValueError(f"expected [B, V, n_new, {self.p}], got {tuple(x.shape)}")
    return self._ingest(x)

  @torch.no_grad()
  def fast_update(self, new_patch: torch.Tensor) -> None:
    """Slide the window by one patch: evict oldest, encode newest (Pi-6).

    The detrend coefficients and their time anchor stay FROZEN at the last
    refresh (recomputing them would re-normalize every cached token,
    collapsing rolling into full recompute); the new patch is detrended with
    the frozen line evaluated at its true integer offset.  Drift is bounded
    by `full_refresh_every`.

    new_patch: [B, V, p] raw values.
    """
    new_patch = self._to_bvx(new_patch, self.p)
    B, V, _ = new_patch.shape

    self.raw_buffer = torch.cat(
      [self.raw_buffer[:, :, self.p :], new_patch], dim=2
    )

    if self.use_linear_detrending:
      t = (
        self.t_offset
        + torch.arange(1, self.p + 1, dtype=torch.float32, device=new_patch.device)
      ) / float(self.cfg.context_length)
      detrended = new_patch - (self.trend_m * t[None, None, :] + self.trend_c)
      x = torch.where(self.trend_apply, detrended, new_patch)
    else:
      x = new_patch
    self.t_offset += self.p

    # Eviction is implicit: the sliding-window term of the mask drops the
    # oldest position as soon as the new one is committed.  No memory moves.
    self._ingest(x.view(B, V, 1, self.p))

  # ------------------------------------------------------------------------
  # Forecast (Pi-7)
  # ------------------------------------------------------------------------

  @torch.no_grad()
  def forecast(self) -> torch.Tensor:
    """H-step quantile forecast in the original scale. Returns [B, V, H, nq].

    Reproduces `decode()`'s readout: for H <= extract_len the stitch/gather
    collapses to the last context patch's logits (util.py:359-360 +
    model.py:613-635); otherwise all scratch CPM patches are encoded in one
    forward inside a mark/rollback fork, refined (CPM revin), stitched or
    chunk-gathered, and the frozen trend is re-added.
    """
    B, V = self.cfg.batch_size, self.cfg.num_variates
    H = self.cfg.horizon
    device = self.cfg.device

    logits_last = self._readout_patches(
      self.last_embedding, self.last_mu[..., None], self.last_sigma[..., None]
    )[:, :, 0]  # [B, V, o, nq]

    if self.nh_scratch == 0:
      hor = logits_last[:, :, :H, :]
    else:
      nh = self.nh_scratch
      # One-shot fork (never AR): encode all scratch patches at once, with
      # the window anchored at the last committed position so scratch queries
      # see the full live window instead of sliding past its oldest patches.
      anchor_pos = self.cache.next_pos - 1
      mark = self.cache.mark()
      tokens = self._scratch_token.expand(B, V, nh, -1).contiguous()
      emb = self._encode(tokens, window_anchor=anchor_pos)
      raw_scratch = self.model.output_head(emb)  # [B, V, nh, o*nq]

      if self.use_iterative_cpm_revin:
        mu_s, sigma_s = self._cpm_refine_scratch(raw_scratch)
      elif self.use_frozen_running_stats:
        # Upstream freeze is a slice-copy of the stats at the last context
        # patch (model.py:220-228): an exact broadcast, not a re-fold.
        mu_s = self.last_mu[:, :, None].expand(B, V, nh)
        sigma_s = self.last_sigma[:, :, None].expand(B, V, nh)
      else:
        # Fold nh all-masked zero patches through the upstream Welford update
        # in LOCAL variables (self.stat_* stays untouched).  This reproduces
        # upstream's repeated sqrt(sigma^2)/(n*mu)/n round-trips bit-for-bit,
        # which a plain stats copy would not.
        n_l, mu_l, sg_l = self.stat_n, self.stat_mu, self.stat_sigma
        zero_patch = torch.zeros(B, V, self.p, device=device)
        ones_mask = torch.ones(B, V, self.p, dtype=torch.bool, device=device)
        mus, sgs = [], []
        for _ in range(nh):
          n_l, mu_l, sg_l = update_running_stats(
            n_l, mu_l, sg_l, zero_patch, ones_mask
          )
          mus.append(mu_l)
          sgs.append(sg_l)
        mu_s = torch.stack(mus, dim=2)
        sigma_s = torch.stack(sgs, dim=2)

      scratch_out = revin(raw_scratch, mu_s, sigma_s, reverse=True)
      scratch_out = torch.clamp(scratch_out, -self.value_clip, self.value_clip)
      scratch_out = scratch_out.view(B, V, nh, self.o, self.nq)

      if self.use_stitching:
        # forecast_indices = arange(nf) + (n_ctx - 1)  (model.py:619-622):
        # index 0 is the last context patch, 1..nf-1 the first scratch ones.
        nf = nh - self.rolls + 1
        patch_preds = torch.cat(
          [
            logits_last[:, :, None, : self.extract_len, :],
            scratch_out[:, :, : nf - 1, : self.extract_len, :],
          ],
          dim=2,
        )
        hor = util.stitch_patches(patch_preds, self.p)[:, :, :H, :]
      else:
        # forecast_indices = arange(chunks)*rolls + (n_ctx - 1)
        # (model.py:628-635) mapped into [last_ctx] + scratch order.
        chunks = nh // self.rolls
        all_out = torch.cat([logits_last[:, :, None], scratch_out], dim=2)
        chunk_idx = torch.arange(chunks, device=device) * self.rolls
        hor = all_out[:, :, chunk_idx].reshape(B, V, -1, self.nq)[:, :, :H, :]

      self.cache.rollback(mark)

    if self.use_linear_detrending:
      # Frozen-trend re-add at the true integer offset (model.py:637-645; at
      # t_offset == 0 this is bit-identical to upstream decode()).
      t_f = (
        self.t_offset
        + torch.arange(1, H + 1, dtype=torch.float32, device=device)
      ) / float(self.cfg.context_length)
      trend = (
        self.trend_m[:, :, 0, None] * t_f[None, None, :]
        + self.trend_c[:, :, 0, None]
      )
      trend = torch.where(self.trend_apply[:, :, 0, None], trend, 0.0)
      hor = hor + trend[:, :, :, None]

    return hor

  # ------------------------------------------------------------------------
  # Online driver
  # ------------------------------------------------------------------------

  @torch.no_grad()
  def step_patch(self, new_patch: torch.Tensor) -> torch.Tensor:
    """Ingest one new patch (p points) and forecast. Returns [B, V, H, nq]."""
    self._n_updates += 1
    k = self.cfg.full_refresh_every
    if k > 0 and self._n_updates % k == 0:
      new_patch = self._to_bvx(new_patch, self.p)
      window = torch.cat([self.raw_buffer[:, :, self.p :], new_patch], dim=2)
      self.full_refresh(window)
    else:
      self.fast_update(new_patch)
    return self.forecast()

  @property
  def cache_age(self) -> int:
    """Number of patch evictions since the last refresh."""
    k = self.cfg.full_refresh_every
    return self._n_updates % k if k > 0 else self._n_updates
