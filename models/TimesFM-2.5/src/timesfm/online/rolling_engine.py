"""Rolling KV-cache engine for online forecasting with TimesFM-2.5.

This implements protocol Pi_TFM variant A (patch-aligned, stride = p = 32) from
`rolling_kv_cache_timesfm25_design_0803.md`.  The forward pass is re-expressed
here rather than monkey-patched into `MultiHeadAttention` so that positions and
masks are fully under protocol control; every weight and every op is taken from
the loaded module, so the math is identical to upstream.

Protocol summary
----------------
Pi-2  token mapping : window slides by exactly one patch (32 points).  Survivor
                      patch identity is therefore exact, retokenization
                      defect = 0.
Pi-3  preprocessing : normalization statistics are **frozen at encode time**.
                      TimesFM normalizes patch j by the causal *prefix* running
                      stats over patches 1..j, so a survivor's own normalization
                      would change if we recomputed the prefix on the shifted
                      window -- which would invalidate every cached token and
                      collapse rolling back into full recompute.  We therefore
                      let the running stats continue accumulating from the last
                      refresh and never recompute them for survivors.  Drift is
                      bounded by `full_refresh_every`.
Pi-4  position      : monotone absolute positions, never renumbered.
Pi-5  mask          : position-tag driven (see rolling_cache.py); eviction costs
                      one integer update and moves no memory.
Pi-6  update        : survivors are reused verbatim; only the new patch is
                      encoded through all 20 layers.
Pi-7  readout       : horizon <= 128 needs no AR.  For longer horizons the AR
                      steps run in scratch slots and are rolled back, so
                      self-generated patches never pollute the history.
Pi-8  refresh       : re-runs the identical prefill path, hence eps_ref = 0 by
                      construction.

Assumptions
-----------
* `context_length % 32 == 0` and the window is fully observed (no padding
  masks).  Online rolling always has a complete window by definition.
* All B series advance in lockstep, so position tags are shared across batch.
"""

from __future__ import annotations

import dataclasses

import torch
import torch.nn.functional as F

from ..torch import util
from .rolling_cache import RollingKVCache

revin = util.revin
update_running_stats = util.update_running_stats


@dataclasses.dataclass
class RollingConfig:
  context_length: int = 2048  # L, must be a multiple of the patch size
  horizon: int = 128  # H
  full_refresh_every: int = 0  # in patch-updates; 0 == never
  batch_size: int = 1
  device: str = "cuda"
  dtype: torch.dtype = torch.float32


class RollingTimesFMEngine:
  """Online forecaster with a sliding KV cache over patch tokens."""

  def __init__(self, module, cfg: RollingConfig):
    self.module = module
    self.cfg = cfg
    module.eval()

    self.p = module.p  # 32
    self.o = module.o  # 128
    self.q = module.q  # 10
    self.m = module.m  # 4
    self.aridx = module.aridx  # 5
    self.n_layers = module.x  # 20
    self.n_heads = module.h  # 16
    self.head_dim = module.hd  # 80

    if cfg.context_length % self.p != 0:
      raise ValueError(
        f"context_length ({cfg.context_length}) must be a multiple of the "
        f"patch size ({self.p})"
      )
    self.n_patches = cfg.context_length // self.p

    # Scratch slots for autoregressive decoding beyond one output patch.
    self.num_decode_steps = max(0, (cfg.horizon - 1) // self.o)
    capacity = self.n_patches + self.num_decode_steps * self.m

    self.cache = RollingKVCache(
      num_layers=self.n_layers,
      batch_size=cfg.batch_size,
      capacity=capacity,
      window=self.n_patches,
      num_heads=self.n_heads,
      head_dim=self.head_dim,
      device=cfg.device,
      dtype=cfg.dtype,
    )

    B = cfg.batch_size
    dev = cfg.device
    self.stat_n = torch.zeros(B, device=dev)
    self.stat_mu = torch.zeros(B, device=dev)
    self.stat_sigma = torch.zeros(B, device=dev)

    self.raw_buffer: torch.Tensor | None = None  # [B, L]
    self.last_mu: torch.Tensor | None = None  # [B] stats of the newest patch
    self.last_sigma: torch.Tensor | None = None
    self.last_embedding: torch.Tensor | None = None  # [B, 1, 1280]
    self._n_updates = 0

  # ------------------------------------------------------------------------
  # Forward pass with the rolling cache
  # ------------------------------------------------------------------------

  def _attn_forward(self, layer_idx, attn, x, positions, mask, slots):
    """One MultiHeadAttention forward, reading/writing the ring buffer.

    Mirrors `timesfm.torch.transformer.MultiHeadAttention.forward` exactly:
    RoPE -> qk RMSNorm -> PerDimScale(query) -> unscaled SDPA -> out proj.
    Keys are cached *after* RoPE and qk-norm, as upstream does.
    """
    B, n, _ = x.shape

    if attn.fuse_qkv:
      qkv = attn.qkv_proj(x)
      query, key, value = torch.chunk(qkv, 3, dim=-1)
    else:
      query, key, value = attn.query(x), attn.key(x), attn.value(x)
    query = query.view(B, n, attn.num_heads, attn.head_dim)
    key = key.view(B, n, attn.num_heads, attn.head_dim)
    value = value.view(B, n, attn.num_heads, attn.head_dim)

    if attn.use_rotary_position_embeddings:
      pos = positions[None, :].expand(B, n)
      query = attn.rotary_position_embedding(query, pos)
      key = attn.rotary_position_embedding(key, pos)

    query = attn.query_ln(query)
    key = attn.key_ln(key)
    if attn.use_per_dim_scale:
      query = attn.per_dim_scale(query)

    self.cache.write_layer(layer_idx, slots, key, value)

    k_all = self.cache.key[layer_idx]  # [B, C, H, D]
    v_all = self.cache.value[layer_idx]

    # RoPE's sin/cos are built in fp32, so query comes out of the rotation in
    # fp32 even for a bf16 module.  Keys/values were cast on the way into the
    # cache, so bring the query back to the cache dtype before attending.
    query = query.to(k_all.dtype)

    out = F.scaled_dot_product_attention(
      query.permute(0, 2, 1, 3),
      k_all.permute(0, 2, 1, 3),
      v_all.permute(0, 2, 1, 3),
      attn_mask=mask,
      scale=1.0,
    ).permute(0, 2, 1, 3)

    return attn.out(out.reshape(B, n, attn.in_features))

  def _layer_forward(self, layer_idx, layer, x, positions, mask, slots):
    """Mirrors `timesfm.torch.transformer.Transformer.forward`."""
    attn_out = self._attn_forward(
      layer_idx, layer.attn, layer.pre_attn_ln(x), positions, mask, slots
    )
    h = layer.post_attn_ln(attn_out) + x
    return (
      layer.post_ff_ln(layer.ff1(layer.activation(layer.ff0(layer.pre_ff_ln(h))))) + h
    )

  def _encode_at(self, normed_patches, patch_masks, slots, positions):
    """Encode into the given slots. Touches no Python-side pointer state.

    Split out from `_encode` so the CUDA-graph runner can drive it with
    fixed-address buffers (see graph_runner.py): everything here is either a
    tensor op or a compile-time constant, so the whole body is capturable.
    """
    # Tag before attending so the mask already sees the new tokens (upstream
    # writes into the cache and then attends over the whole cache).
    self.cache.slot_pos.index_copy_(0, slots, positions)
    mask = self.cache.build_mask(positions)

    tokenizer_inputs = torch.cat(
      [normed_patches, patch_masks.to(normed_patches.dtype)], dim=-1
    )
    x = self.module.tokenizer(tokenizer_inputs)

    for i, layer in enumerate(self.module.stacked_xf):
      x = self._layer_forward(i, layer, x, positions, mask, slots)
    return x

  def _encode(self, normed_patches, patch_masks):
    """Encode new patches into the cache and return their final embeddings.

    Args:
      normed_patches: [B, n_new, 32] already-normalized values.
      patch_masks: [B, n_new, 32] bool.

    Returns:
      [B, n_new, model_dims]
    """
    n_new = normed_patches.shape[1]
    slots, positions = self.cache.reserve(n_new)
    x = self._encode_at(normed_patches, patch_masks, slots, positions)
    self.cache.advance(n_new)
    return x

  # ------------------------------------------------------------------------
  # Normalization (Pi-3): running prefix stats, frozen at encode time
  # ------------------------------------------------------------------------

  def _advance_stats(self, patches, masks):
    """Fold `n_new` patches into the running stats, one patch at a time.

    Returns per-patch (mu, sigma), each [B, n_new] -- exactly the prefix
    statistics upstream's `decode()` computes.
    """
    mus, sigmas = [], []
    n, mu, sigma = self.stat_n, self.stat_mu, self.stat_sigma
    for i in range(patches.shape[1]):
      (n, mu, sigma), _ = update_running_stats(n, mu, sigma, patches[:, i], masks[:, i])
      mus.append(mu)
      sigmas.append(sigma)
    self.stat_n, self.stat_mu, self.stat_sigma = n, mu, sigma
    return torch.stack(mus, dim=1), torch.stack(sigmas, dim=1)

  # ------------------------------------------------------------------------
  # Core operations
  # ------------------------------------------------------------------------

  @torch.no_grad()
  def full_refresh(self, raw_window: torch.Tensor) -> None:
    """Re-encode the whole window from scratch (Pi-8). eps_ref = 0."""
    if raw_window.dim() == 1:
      raw_window = raw_window.unsqueeze(0)
    raw_window = raw_window.to(device=self.cfg.device, dtype=self.cfg.dtype)
    B, L = raw_window.shape
    if L != self.cfg.context_length:
      raise ValueError(f"expected window of {self.cfg.context_length}, got {L}")

    self.raw_buffer = raw_window.clone()

    self.cache.reset()
    self.stat_n = torch.zeros(B, device=self.cfg.device)
    self.stat_mu = torch.zeros(B, device=self.cfg.device)
    self.stat_sigma = torch.zeros(B, device=self.cfg.device)

    patches = raw_window.view(B, self.n_patches, self.p)
    masks = torch.zeros_like(patches, dtype=torch.bool)

    mu, sigma = self._advance_stats(patches, masks)
    # Stats are kept in fp32 (Welford in bf16 is not accurate enough); cast
    # back so the tokenizer sees the module's own dtype.
    normed = revin(patches, mu, sigma, reverse=False).to(self.cfg.dtype)

    emb = self._encode(normed, masks)
    self.last_embedding = emb[:, -1:, :]
    self.last_mu, self.last_sigma = mu[:, -1], sigma[:, -1]

  @torch.no_grad()
  def fast_update(self, new_patch: torch.Tensor) -> None:
    """Slide the window by one patch: evict oldest, encode newest (Pi-6)."""
    new_patch = new_patch.to(device=self.cfg.device, dtype=self.cfg.dtype)
    if new_patch.dim() == 1:
      new_patch = new_patch.unsqueeze(0)
    B = new_patch.shape[0]
    patches = new_patch.view(B, 1, self.p)
    masks = torch.zeros_like(patches, dtype=torch.bool)

    self.raw_buffer = torch.cat([self.raw_buffer[:, self.p :], new_patch], dim=1)

    mu, sigma = self._advance_stats(patches, masks)
    normed = revin(patches, mu, sigma, reverse=False).to(self.cfg.dtype)

    # Eviction is implicit: the sliding-window term of the mask drops the
    # oldest position as soon as the new one is committed.  No memory moves.
    emb = self._encode(normed, masks)
    self.last_embedding = emb
    self.last_mu, self.last_sigma = mu[:, -1], sigma[:, -1]

  # ------------------------------------------------------------------------
  # Readout (Pi-7)
  # ------------------------------------------------------------------------

  @torch.no_grad()
  def _readout(self, emb, mu, sigma):
    """Point + quantile head on one patch embedding -> [B, 128, 10]."""
    out = self.module.output_projection_point(emb)  # [B, 1, 1280]
    out = revin(out, mu[:, None], sigma[:, None], reverse=True)
    return out.reshape(-1, self.o, self.q)

  @torch.no_grad()
  def forecast(self) -> torch.Tensor:
    """H-step point forecast in the original scale. Returns [B, H]."""
    H = self.cfg.horizon
    pieces = [self._readout(self.last_embedding, self.last_mu, self.last_sigma)]

    if self.num_decode_steps > 0:
      # AR fork: scratch slots + snapshot of the running stats (Pi-7).
      mark = self.cache.mark()
      saved = (self.stat_n.clone(), self.stat_mu.clone(), self.stat_sigma.clone())

      last = pieces[0][..., self.aridx]  # [B, 128] point forecast
      for _ in range(self.num_decode_steps):
        B = last.shape[0]
        new_patches = last.reshape(B, self.m, self.p)
        new_masks = torch.zeros_like(new_patches, dtype=torch.bool)
        mu, sigma = self._advance_stats(new_patches, new_masks)
        normed = revin(new_patches, mu, sigma, reverse=False).to(self.cfg.dtype)
        emb = self._encode(normed, new_masks)
        step_out = self._readout(emb[:, -1:, :], mu[:, -1], sigma[:, -1])
        pieces.append(step_out)
        last = step_out[..., self.aridx]

      self.cache.rollback(mark)
      self.stat_n, self.stat_mu, self.stat_sigma = saved

    full = torch.cat(pieces, dim=1)  # [B, 128*(1+steps), 10]
    return full[:, :H, self.aridx]

  # ------------------------------------------------------------------------
  # Online driver
  # ------------------------------------------------------------------------

  @torch.no_grad()
  def step_patch(self, new_patch: torch.Tensor) -> torch.Tensor:
    """Ingest one new patch (32 points) and forecast. Returns [B, H]."""
    self._n_updates += 1
    k = self.cfg.full_refresh_every
    if k > 0 and self._n_updates % k == 0:
      new_patch = new_patch.to(device=self.cfg.device, dtype=self.cfg.dtype)
      if new_patch.dim() == 1:
        new_patch = new_patch.unsqueeze(0)
      window = torch.cat([self.raw_buffer[:, self.p :], new_patch], dim=1)
      self.full_refresh(window)
    else:
      self.fast_update(new_patch)
    return self.forecast()

  @property
  def cache_age(self) -> int:
    """Number of patch evictions since the last refresh."""
    k = self.cfg.full_refresh_every
    return self._n_updates % k if k > 0 else self._n_updates
