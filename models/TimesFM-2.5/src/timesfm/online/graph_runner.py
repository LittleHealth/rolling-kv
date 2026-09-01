"""CUDA Graph capture of the TimesFM-2.5 rolling fast-update step.

Why this captures cleanly
-------------------------
A CUDA graph is a recorded sequence of kernel launches replayed with one host
call.  Capture requires every tensor to live at a fixed address and every shape
to be static.  In contrast, the original TimeMoE dispatcher needs a separate
fixed-shape expert path because `torch.where` selects a data-dependent number
of tokens per expert.

TimesFM-2.5 has **no MoE** -- the FFN is a plain dense swish block -- and the
rolling fast update always processes exactly one patch token against a
fixed-size ring buffer.  Every shape in the step is therefore a compile-time
constant, and the whole thing captures cleanly.

Design
------
The graph is made fully self-contained: the slot index and absolute position
advance *inside* the graph as in-place tensor ops, and the running
normalization stats live in fixed buffers that are updated in place.  A replay
therefore needs exactly one host-side action -- copying the new patch into the
input buffer -- instead of ~300 kernel launches.

The eviction is free as always: it is implied by the sliding-window term of the
position-tag mask, which is recomputed inside the graph from `slot_pos`.
"""

from __future__ import annotations

import torch

from ..torch import util

revin = util.revin
update_running_stats = util.update_running_stats


class CudaGraphRollingStep:
  """Captures `fast_update` + point readout into a single replayable graph.

  Only valid for horizon <= output_patch_len (no autoregressive decode), which
  is the case the graph is worth capturing for: the AR path re-enters the
  transformer a variable number of times and is better handled by capturing
  this step and looping it.
  """

  def __init__(self, engine, warmup: int = 3):
    if engine.num_decode_steps > 0:
      raise ValueError(
        "CUDA graph capture supports horizon <= output_patch_len "
        f"({engine.o}); got horizon={engine.cfg.horizon}"
      )
    self.eng = engine
    cfg = engine.cfg
    B, p = cfg.batch_size, engine.p
    dev, dt = cfg.device, cfg.dtype

    # ---- fixed-address buffers -------------------------------------------
    self.x = torch.zeros(B, 1, p, device=dev, dtype=dt)
    self.zero_mask = torch.zeros(B, 1, p, device=dev, dtype=torch.bool)
    self.slots = torch.zeros(1, dtype=torch.long, device=dev)
    self.positions = torch.zeros(1, dtype=torch.long, device=dev)

    self.n = torch.zeros(B, device=dev)
    self.mu = torch.zeros(B, device=dev)
    self.sigma = torch.zeros(B, device=dev)

    self._cap = engine.cache.capacity
    self.graph: torch.cuda.CUDAGraph | None = None
    self.out: torch.Tensor | None = None
    self._warmup = warmup

  # -------------------------------------------------------------- body ----

  def _body(self):
    """One fast update, written entirely in fixed-address tensor ops."""
    eng = self.eng

    # Welford step, in place so the buffers keep their addresses.
    (n2, mu2, sg2), _ = update_running_stats(
      self.n, self.mu, self.sigma, self.x[:, 0], self.zero_mask[:, 0]
    )
    self.n.copy_(n2)
    self.mu.copy_(mu2)
    self.sigma.copy_(sg2)

    normed = revin(self.x, self.mu[:, None], self.sigma[:, None], reverse=False)
    normed = normed.to(eng.cfg.dtype)
    emb = eng._encode_at(normed, self.zero_mask, self.slots, self.positions)
    out = eng._readout(emb, self.mu, self.sigma)

    # Advance the ring inside the graph: replay then needs no host bookkeeping.
    self.slots.add_(1).remainder_(self._cap)
    self.positions.add_(1)

    return out[:, : eng.cfg.horizon, eng.aridx]

  # ----------------------------------------------------------- capture ----

  @torch.no_grad()
  def capture(self, preserve_state: bool = False) -> None:
    """Sync host state into the buffers, warm up, then record the graph.

    ``preserve_state`` is for accuracy evaluation.  CUDA requires warm-up
    executions before capture, but those executions would otherwise append
    dummy patches to the live cache.  Restoring the state afterwards leaves
    the graph ready to consume the first real online patch from precisely the
    same history as an eager engine.
    """
    eng = self.eng
    saved = None
    if preserve_state:
      saved = (
        eng.cache.key.clone(), eng.cache.value.clone(), eng.cache.slot_pos.clone(),
        eng.cache.write_ptr, eng.cache.next_pos, eng.cache.n_written,
        eng.stat_n.clone(), eng.stat_mu.clone(), eng.stat_sigma.clone(),
      )
    self.n.copy_(eng.stat_n)
    self.mu.copy_(eng.stat_mu)
    self.sigma.copy_(eng.stat_sigma)
    self.slots.fill_(eng.cache.write_ptr)
    self.positions.fill_(eng.cache.next_pos)

    # Warm-up on a side stream is required before capture; it mutates the
    # cache exactly as real steps would, so the state stays coherent.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
      for _ in range(self._warmup):
        self._body()
    torch.cuda.current_stream().wait_stream(stream)

    self.graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(self.graph):
      self.out = self._body()

    # Capture *records* kernels without running them, so the capture pass
    # leaves no side effect: only the warm-up iterations actually advanced the
    # ring.  Read the authoritative values back out of the device buffers
    # rather than trying to count passes.
    if saved is None:
      eng.cache.write_ptr = int(self.slots.item())
      eng.cache.next_pos = int(self.positions.item())
      eng.stat_n.copy_(self.n)
      eng.stat_mu.copy_(self.mu)
      eng.stat_sigma.copy_(self.sigma)
    else:
      (key, value, slot_pos, write_ptr, next_pos, n_written, n, mu, sigma) = saved
      eng.cache.key.copy_(key)
      eng.cache.value.copy_(value)
      eng.cache.slot_pos.copy_(slot_pos)
      eng.cache.write_ptr = write_ptr
      eng.cache.next_pos = next_pos
      eng.cache.n_written = n_written
      eng.stat_n.copy_(n)
      eng.stat_mu.copy_(mu)
      eng.stat_sigma.copy_(sigma)
      self.n.copy_(n)
      self.mu.copy_(mu)
      self.sigma.copy_(sigma)
      self.slots.fill_(write_ptr)
      self.positions.fill_(next_pos)

  # ------------------------------------------------------------ replay ----

  @torch.no_grad()
  def step(self, new_patch: torch.Tensor) -> torch.Tensor:
    """Replay the captured step on a new patch. Returns [B, H]."""
    if self.graph is None:
      raise RuntimeError("call capture() first")
    self.x.copy_(new_patch.view_as(self.x))
    self.graph.replay()
    self.eng.cache.advance(1)
    return self.out


class CudaGraphFullDecode:
  """CUDA-graph replay for fixed-shape TimesFM full recomputation.

  Upstream ``module.decode`` dynamically allocates its append-only decode cache,
  which CUDA Graph rejects.  This class instead invokes the rolling engine's
  T1-validated full-prefill forward on preallocated ring-buffer storage.  It
  writes every context token on every replay, so it is still a full recompute;
  it merely avoids the allocation that blocks graph capture.
  """

  def __init__(self, engine, rolling_target=None):
    if engine.num_decode_steps > 0:
      raise ValueError(
        "CUDA graph full decode supports horizon <= output_patch_len "
        f"({engine.o}); got horizon={engine.cfg.horizon}"
      )
    self.eng = engine
    self.rolling_target = rolling_target
    B, L, p = engine.cfg.batch_size, engine.cfg.context_length, engine.p
    self.x = torch.zeros(B, L, device=engine.cfg.device, dtype=engine.cfg.dtype)
    self.zero_mask = torch.zeros(B, engine.n_patches, p,
                                 device=engine.cfg.device, dtype=torch.bool)
    self.slots = torch.arange(engine.n_patches, device=engine.cfg.device)
    self.positions = torch.arange(engine.n_patches, device=engine.cfg.device)
    self.n = torch.zeros(B, device=engine.cfg.device)
    self.mu = torch.zeros(B, device=engine.cfg.device)
    self.sigma = torch.zeros(B, device=engine.cfg.device)
    self.graph: torch.cuda.CUDAGraph | None = None
    self.out: torch.Tensor | None = None

  def _body(self):
    eng = self.eng
    patches = self.x.view(eng.cfg.batch_size, eng.n_patches, eng.p)
    # Every slot is overwritten below, and validity tags are rebuilt from zero.
    eng.cache.slot_pos.fill_(-1)
    self.n.zero_()
    self.mu.zero_()
    self.sigma.zero_()
    n, mu, sigma = self.n, self.mu, self.sigma
    mus, sigmas = [], []
    for i in range(eng.n_patches):
      (n, mu, sigma), _ = update_running_stats(n, mu, sigma,
                                                patches[:, i], self.zero_mask[:, i])
      mus.append(mu)
      sigmas.append(sigma)
    mus, sigmas = torch.stack(mus, dim=1), torch.stack(sigmas, dim=1)
    normed = revin(patches, mus, sigmas, reverse=False).to(eng.cfg.dtype)
    emb = eng._encode_at(normed, self.zero_mask, self.slots, self.positions)
    out = eng._readout(emb[:, -1:, :], mus[:, -1], sigmas[:, -1])

    # Keep the final prefix statistics in persistent buffers.  If this graph
    # is being used as a periodic refresh, also install the complete refreshed
    # state into the rolling graph at fixed addresses.  These copies become
    # part of CUDA Graph replay, so switching back to rolling needs no eager
    # synchronization or recapture.
    self.n.copy_(n)
    self.mu.copy_(mu)
    self.sigma.copy_(sigma)
    target = self.rolling_target
    if target is not None:
      target.eng.cache.key.copy_(eng.cache.key)
      target.eng.cache.value.copy_(eng.cache.value)
      target.eng.cache.slot_pos.copy_(eng.cache.slot_pos)
      target.n.copy_(n)
      target.mu.copy_(mu)
      target.sigma.copy_(sigma)
      target.slots.zero_()
      target.positions.fill_(eng.n_patches)
    return out[:, : eng.cfg.horizon, eng.aridx]

  @torch.no_grad()
  def capture(self, warmup: int = 3, preserve_target: bool = True) -> None:
    target = self.rolling_target
    saved = None
    if target is not None and preserve_target:
      saved = (
          target.eng.cache.key.clone(), target.eng.cache.value.clone(),
          target.eng.cache.slot_pos.clone(), target.n.clone(), target.mu.clone(),
          target.sigma.clone(), target.slots.clone(), target.positions.clone(),
      )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
      for _ in range(warmup):
        self._body()
    torch.cuda.current_stream().wait_stream(stream)

    self.graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(self.graph):
      self.out = self._body()

    if saved is not None:
      key, value, slot_pos, n, mu, sigma, slots, positions = saved
      target.eng.cache.key.copy_(key)
      target.eng.cache.value.copy_(value)
      target.eng.cache.slot_pos.copy_(slot_pos)
      target.n.copy_(n)
      target.mu.copy_(mu)
      target.sigma.copy_(sigma)
      target.slots.copy_(slots)
      target.positions.copy_(positions)

  @torch.no_grad()
  def step(self, window: torch.Tensor) -> torch.Tensor:
    if self.graph is None:
      raise RuntimeError("call capture() first")
    self.x.copy_(window.view_as(self.x))
    self.graph.replay()
    if self.rolling_target is not None:
      cache = self.rolling_target.eng.cache
      cache.write_ptr = 0
      cache.next_pos = self.eng.n_patches
      cache.n_written = self.eng.n_patches
    return self.out
