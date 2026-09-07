"""Ring-buffer KV cache with absolute position tags for TimesFM-3.0.

Why not reuse `timesfm3.util.DecodeCache`?
------------------------------------------
Upstream's cache is append-only and its mask (`transformer.make_attn_mask`)
encodes the assumption that the masked-out KV slots form a *prefix* interval:

    keep[s] = (q_index >= kv_index) & (kv_index >= num_all_masked_kv)

For online rolling we must evict from the left indefinitely, so the live slots
form a *wrapping* interval in physical memory.  We therefore tag every slot
with the absolute position of the token stored in it and build the mask from
those tags instead of from physical indices:

    keep[q, s] = valid[s]                       # slot has been written
               & (pos[s] <= pos[q])             # causal
               & (pos[s] >= anchor - (W - 1))   # sliding window of W tokens

where `anchor` is normally the query's own position (a token entering the
window pushes the oldest one out) but can be pinned to the last *committed*
position for the horizon-scratch fork (see `build_mask`).

This makes eviction a pure integer update (no memory movement at all), which
is the main implementation win over the `slice_cache`-style eviction used for
TimeMoE.

Protocol notes (mirroring the TimesFM-2.5 engine)
-------------------------------------------------
* Positions are **monotone absolute** and never renumbered (Pi-4).  For
  TimesFM-3.0 this is load-bearing, not merely convenient: RoPE is followed
  by elementwise-affine qk RMSNorm and PerDimScale, whose per-dim weights are
  not symmetric within a rotation pair, so attention logits are NOT invariant
  under a global position shift (see the Pi-4 note in rolling_engine.py).
  Keys are stored post-RoPE / post-qk-RMSNorm, exactly as upstream's
  DecodeCache stores them (transformer.py writes the cache after RoPE and
  key_ln).
* The leading dimension is batch*variates: TimesFM-3.0 flattens (b, v) before
  sequence attention, and all B series and V variates advance in lockstep, so
  `slot_pos` is shared across the whole leading dimension.
* `mark()` / `rollback()` implement the one-shot horizon-scratch fork (Pi-7).
  TimesFM-3.0 is not autoregressive: all horizon patches are encoded in a
  single forward, read out, and immediately rolled back, so scratch tokens
  never become history.
"""

from __future__ import annotations

import dataclasses

import torch


@dataclasses.dataclass
class CacheMark:
  """Opaque snapshot of the cache pointers, for scratch-fork rollback."""

  write_ptr: int
  next_pos: int
  n_written: int


class RollingKVCache:
  """Per-layer ring buffers of post-RoPE keys and raw values."""

  def __init__(
    self,
    num_layers: int,
    batch_leading: int,
    capacity: int,
    window: int,
    num_heads: int,
    head_dim: int,
    device,
    dtype=torch.float32,
  ):
    if capacity < window:
      raise ValueError(f"capacity ({capacity}) must be >= window ({window})")

    self.num_layers = num_layers
    self.batch_leading = batch_leading
    self.capacity = capacity
    self.window = window
    self.num_heads = num_heads
    self.head_dim = head_dim
    self.device = device
    self.dtype = dtype

    shape = (num_layers, batch_leading, capacity, num_heads, head_dim)
    self.key = torch.zeros(shape, device=device, dtype=dtype)
    self.value = torch.zeros(shape, device=device, dtype=dtype)

    # Absolute position stored in each physical slot; -1 == never written.
    self.slot_pos = torch.full((capacity,), -1, dtype=torch.long, device=device)

    self.write_ptr = 0  # next physical slot to write
    self.next_pos = 0  # next absolute position to hand out
    self.n_written = 0  # total tokens ever written (for diagnostics)

  # ---------------------------------------------------------------- reset ---

  def reset(self) -> None:
    """Full reset. Used by `full_refresh` (Pi-8), which rebuilds everything."""
    self.slot_pos.fill_(-1)
    self.write_ptr = 0
    self.next_pos = 0
    self.n_written = 0

  # --------------------------------------------------------------- append ---

  def reserve(self, n_new: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Reserve `n_new` slots.

    Returns:
      slots: [n_new] physical indices (may wrap around the ring).
      positions: [n_new] absolute positions assigned to those slots.

    Nothing is written yet: the engine tags the slots (`slot_pos.index_copy_`
    in `_encode_at`, so the mask sees the new tokens before attending), writes
    each layer via `write_layer`, and finally calls `advance`.  Splitting
    reservation from publication lets all layers share one slot computation.
    """
    if n_new > self.capacity:
      raise ValueError(
        f"cannot reserve {n_new} slots in a ring of capacity {self.capacity}: "
        "wrapped duplicate indices would make index_copy_ nondeterministic"
      )
    idx = torch.arange(n_new, device=self.device)
    slots = (self.write_ptr + idx) % self.capacity
    positions = self.next_pos + idx
    return slots, positions

  def write_layer(
    self,
    layer_idx: int,
    slots: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
  ) -> None:
    """Write one layer's K/V into the reserved slots.

    key / value: [batch_leading, n_new, num_heads, head_dim]
    """
    self.key[layer_idx].index_copy_(1, slots, key.to(self.dtype))
    self.value[layer_idx].index_copy_(1, slots, value.to(self.dtype))

  def advance(self, n_new: int) -> None:
    """Move the Python-side pointers past `n_new` freshly written slots."""
    self.write_ptr = int((self.write_ptr + n_new) % self.capacity)
    self.next_pos = int(self.next_pos + n_new)
    self.n_written += n_new

  # ----------------------------------------------------------------- mask ---

  def build_mask(
    self,
    q_positions: torch.Tensor,
    window_anchor: int | None = None,
  ) -> torch.Tensor:
    """Attention mask over the whole ring for the given query positions.

    Args:
      q_positions: [n_q] absolute positions of the queries.
      window_anchor: If None (the normal append/slide case), the window is
        anchored per query: `pos >= q - (W - 1)`, so committing a new token
        implicitly evicts the oldest one.  For the horizon-scratch fork the
        queries sit at positions *past* the last committed token; a per-query
        anchor would slide the window forward with them and drop the oldest
        live context patches, which upstream's horizon patches can still see.
        Passing the last committed position pins the window there instead:
        `pos >= window_anchor - (W - 1)`, so every scratch query sees the full
        committed window plus the scratch tokens before it.

    Returns:
      [1, 1, n_q, capacity] bool tensor, True == attend.  Broadcasts over the
      leading (batch*variate) dim and heads; the engine expands it to the full
      (b*v, heads, n_q, capacity) shape upstream's SDPA call uses.
    """
    pos = self.slot_pos[None, :]  # [1, capacity]
    q = q_positions[:, None]  # [n_q, 1]

    valid = pos >= 0
    causal = pos <= q
    if window_anchor is None:
      in_window = pos >= (q - (self.window - 1))
    else:
      in_window = pos >= (window_anchor - (self.window - 1))
    return (valid & causal & in_window)[None, None, :, :]

  # --------------------------------------------------------- scratch fork ---

  def mark(self) -> CacheMark:
    return CacheMark(self.write_ptr, self.next_pos, self.n_written)

  def rollback(self, mark: CacheMark) -> None:
    """Discard everything written since `mark` (Pi-7 scratch fork).

    Only the position tags of the scratch slots need clearing; the stale K/V
    bytes stay in place and are overwritten on the next append.  O(scratch).
    """
    n_scratch = self.next_pos - mark.next_pos
    if n_scratch <= 0:
      return
    idx = torch.arange(n_scratch, device=self.device)
    slots = (mark.write_ptr + idx) % self.capacity
    self.slot_pos.index_fill_(0, slots, -1)
    self.write_ptr = mark.write_ptr
    self.next_pos = mark.next_pos
    self.n_written = mark.n_written

  # ------------------------------------------------------------ accounting --

  @property
  def live_tokens(self) -> int:
    """Number of tokens currently visible to the newest query."""
    return min(self.next_pos, self.window)

  def nbytes(self) -> int:
    return self.key.numel() * self.key.element_size() * 2
