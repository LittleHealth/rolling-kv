"""Ring-buffer KV cache with absolute position tags for TimesFM-2.5.

Why not reuse `timesfm.torch.util.DecodeCache`?
----------------------------------------------
Upstream's cache is append-only and its mask (`make_attn_mask`) encodes the
assumption that the masked-out KV slots form a *prefix* interval:

    keep[s] = (q_index >= kv_index) & (kv_index >= num_all_masked_kv)

For online rolling we must evict from the left indefinitely, so the live slots
form a *wrapping* interval in physical memory.  We therefore tag every slot
with the absolute position of the token stored in it and build the mask from
those tags instead of from physical indices:

    keep[q, s] = valid[s]                       # slot has been written
               & (pos[s] <= pos[q])             # causal
               & (pos[s] >= pos[q] - (W - 1))   # sliding window of W tokens

This makes eviction a pure integer update (no memory movement at all), which is
the main implementation win over the `slice_cache`-style eviction used for
TimeMoE.

Protocol notes (see rolling_kv_cache_timesfm25_design_0803.md)
-------------------------------------------------------------
* Positions are **monotone absolute** and never renumbered (Pi-4).  Keys are
  stored post-RoPE / post-qk-RMSNorm, exactly as upstream stores them.
* All batch elements advance in lockstep, so `slot_pos` / `valid` are shared
  across the batch dimension.  This is asserted by construction: the engine
  always feeds all B series the same number of tokens.
* `mark()` / `rollback()` implement the AR fork required by Pi-7: autoregressive
  decode steps write into scratch slots past the committed context and are
  discarded afterwards, so self-generated patches never become history.
"""

from __future__ import annotations

import dataclasses

import torch


@dataclasses.dataclass
class CacheMark:
  """Opaque snapshot of the cache pointers, for AR rollback."""

  write_ptr: int
  next_pos: int
  n_written: int


class RollingKVCache:
  """Per-layer ring buffers of post-RoPE keys and raw values."""

  def __init__(
    self,
    num_layers: int,
    batch_size: int,
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
    self.batch_size = batch_size
    self.capacity = capacity
    self.window = window
    self.num_heads = num_heads
    self.head_dim = head_dim
    self.device = device
    self.dtype = dtype

    shape = (num_layers, batch_size, capacity, num_heads, head_dim)
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

    Nothing is written yet: the caller writes each layer via `write_layer`
    and then calls `commit`.  Splitting reserve/commit lets all 20 layers
    share one slot computation.
    """
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

    key / value: [B, n_new, num_heads, head_dim]
    """
    self.key[layer_idx].index_copy_(1, slots, key.to(self.dtype))
    self.value[layer_idx].index_copy_(1, slots, value.to(self.dtype))

  def advance(self, n_new: int) -> None:
    """Move the Python-side pointers past `n_new` freshly written slots."""
    self.write_ptr = int((self.write_ptr + n_new) % self.capacity)
    self.next_pos = int(self.next_pos + n_new)
    self.n_written += n_new

  def commit(self, slots: torch.Tensor, positions: torch.Tensor) -> None:
    """Publish the reserved slots: tag them and advance the pointers."""
    self.slot_pos.index_copy_(0, slots, positions)
    self.advance(slots.shape[0])

  # ----------------------------------------------------------------- mask ---

  def build_mask(self, q_positions: torch.Tensor) -> torch.Tensor:
    """Attention mask over the whole ring for the given query positions.

    Args:
      q_positions: [n_q] absolute positions of the queries.

    Returns:
      [1, 1, n_q, capacity] bool tensor, True == attend.  Broadcasts over
      batch and heads, matching `F.scaled_dot_product_attention`'s expectation.
    """
    pos = self.slot_pos[None, :]  # [1, capacity]
    q = q_positions[:, None]  # [n_q, 1]

    valid = pos >= 0
    causal = pos <= q
    in_window = pos >= (q - (self.window - 1))
    return (valid & causal & in_window)[None, None, :, :]

  # -------------------------------------------------------------- AR fork ---

  def mark(self) -> CacheMark:
    return CacheMark(self.write_ptr, self.next_pos, self.n_written)

  def rollback(self, mark: CacheMark) -> None:
    """Discard everything written since `mark` (Pi-7 AR fork).

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
