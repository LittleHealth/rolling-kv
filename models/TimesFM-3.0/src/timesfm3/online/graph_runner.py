"""CUDA Graph capture of the TimesFM-3.0 rolling fast-update step.

Why this captures cleanly
-------------------------
A CUDA graph is a recorded sequence of kernel launches replayed with one host
call.  Capture requires every tensor to live at a fixed address and every
shape to be static.  TimesFM-3.0 is a dense model -- the FFN is a plain relu
block, there is no MoE and no data-dependent routing -- and the rolling fast
update always processes exactly one patch token against a fixed-size ring
buffer.  The per-step control flow (detrend on/off, variate attention on/off)
is decided by checkpoint config, i.e. it is a compile-time constant; the
data-dependent pieces (`trend_apply`, the Welford update) are tensor `where`s
and one fixed-shape call.  The horizon-scratch fork and CPM refinement never
run in the captured step (the guard requires a horizon short enough that the
forecast reads only the last context patch), so every shape is a compile-time
constant and the whole step captures cleanly.

Design
------
The graph is made fully self-contained: the slot index, absolute position and
the detrend time offset advance *inside* the graph as in-place tensor ops,
and the running normalization stats live in fixed buffers updated in place.
A replay therefore needs exactly one host-side action -- copying the new
patch into the input buffer -- instead of hundreds of kernel launches.

The detrend time offset is an int32 buffer advanced by `p` per step and
converted to float only where the frozen trend line is evaluated.  This
mirrors the eager engine's Python-int `t_offset` exactly: accumulating the
normalized offset `p/L` in floating point instead would drift away from the
eager path and break the T3 bit-exactness expectation.  (int32 -> fp32 is
exact up to 2^24 points since refresh, far beyond any refresh interval.)

The eviction is free as always: it is implied by the sliding-window term of
the position-tag mask, which is recomputed inside the graph from `slot_pos`.
"""

from __future__ import annotations

import torch

from .. import util

revin = util.revin
update_running_stats = util.update_running_stats


class CudaGraphRollingStep:
  """Captures `fast_update` + forecast readout into one replayable graph.

  Only valid when the engine needs no scratch tokens (horizon <= stitching
  extract_len, resp. <= output_patch_len without stitching), which is the
  case the graph is worth capturing for: the scratch fork re-enters the
  transformer with a different token count and is better served eagerly.

  Note: replays advance the engine's cache and (host-mirrored) pointers but
  not `last_embedding` / `token_history`; drive either the runner or the
  eager engine, not both interleaved.
  """

  def __init__(self, engine, warmup: int = 3):
    if engine.nh_scratch != 0:
      raise ValueError(
        "CUDA graph capture supports horizons that need no scratch tokens "
        f"(horizon <= {engine.extract_len if engine.use_stitching else engine.o}); "
        f"got horizon={engine.cfg.horizon}"
      )
    self.eng = engine
    cfg = engine.cfg
    B, V, p, o, H = cfg.batch_size, cfg.num_variates, engine.p, engine.o, cfg.horizon
    dev, dt = cfg.device, cfg.dtype

    # ---- fixed-address buffers -------------------------------------------
    self.x = torch.zeros(B, V, p, device=dev, dtype=dt)
    self.zero_mask = torch.zeros(B, V, p, device=dev, dtype=torch.bool)
    self.slots = torch.zeros(1, dtype=torch.long, device=dev)
    self.positions = torch.zeros(1, dtype=torch.long, device=dev)

    self.n = torch.zeros(B, V, device=dev)
    self.mu = torch.zeros(B, V, device=dev)
    self.sigma = torch.zeros(B, V, device=dev)

    self.trend_m = torch.zeros(B, V, 1, device=dev)
    self.trend_c = torch.zeros(B, V, 1, device=dev)
    self.trend_apply = torch.zeros(B, V, 1, dtype=torch.bool, device=dev)
    self.t_off = torch.zeros(1, dtype=torch.int32, device=dev)

    self.t_patch = torch.arange(1, p + 1, dtype=torch.float32, device=dev)
    self.t_hor = torch.arange(1, H + 1, dtype=torch.float32, device=dev)
    # Constant token tail past the normalized values: fully-masked future
    # covariates (zeros, o), value mask (zeros, p), fcov mask (ones, o).
    self.const_tail = torch.cat(
      [
        torch.zeros(o + p, device=dev, dtype=dt),
        torch.ones(o, device=dev, dtype=dt),
      ]
    ).view(1, 1, 1, o + p + o)

    self._cap = engine.cache.capacity
    self.graph: torch.cuda.CUDAGraph | None = None
    self.out: torch.Tensor | None = None
    self._warmup = warmup

  # -------------------------------------------------------------- body ----

  def _body(self):
    """One fast update + forecast, entirely in fixed-address tensor ops."""
    eng = self.eng
    B, V = eng.cfg.batch_size, eng.cfg.num_variates
    L = float(eng.cfg.context_length)

    # Frozen-trend detrend of the incoming patch (mirrors eager fast_update:
    # the patch's time offsets are computed BEFORE t_off advances).
    if eng.use_linear_detrending:
      t = (self.t_off.float() + self.t_patch) / L
      detrended = self.x - (self.trend_m * t[None, None, :] + self.trend_c)
      xd = torch.where(self.trend_apply, detrended, self.x)
    else:
      xd = self.x
    xd = torch.clamp(
      torch.nan_to_num(xd, nan=0.0), -eng.value_clip, eng.value_clip
    )

    # Welford step, in place so the buffers keep their addresses.
    n2, mu2, sg2 = update_running_stats(
      self.n, self.mu, self.sigma, xd, self.zero_mask
    )
    self.n.copy_(n2)
    self.mu.copy_(mu2)
    self.sigma.copy_(sg2)

    normed = revin(xd, self.mu, self.sigma, reverse=False).to(eng.cfg.dtype)
    tokens = torch.cat(
      [normed[:, :, None, :], self.const_tail.expand(B, V, 1, -1)], dim=-1
    )
    emb = eng._encode_at(tokens, self.slots, self.positions)
    out = eng._readout_patches(emb, self.mu[..., None], self.sigma[..., None])
    hor = out[:, :, 0, : eng.cfg.horizon, :]  # [B, V, H, nq]

    # Advance the point offset BEFORE evaluating the forecast trend, exactly
    # as the eager path does (fast_update advances t_offset, then forecast
    # reads the advanced value).
    self.t_off.add_(eng.p)
    if eng.use_linear_detrending:
      t_f = (self.t_off.float() + self.t_hor) / L
      trend = (
        self.trend_m[:, :, 0, None] * t_f[None, None, :]
        + self.trend_c[:, :, 0, None]
      )
      trend = torch.where(self.trend_apply[:, :, 0, None], trend, 0.0)
      hor = hor + trend[:, :, :, None]

    # Advance the ring inside the graph: replay then needs no host bookkeeping.
    self.slots.add_(1).remainder_(self._cap)
    self.positions.add_(1)

    return hor

  # ----------------------------------------------------------- capture ----

  @torch.no_grad()
  def capture(self, preserve_state: bool = False) -> None:
    """Sync host state into the buffers, warm up, then record the graph.

    ``preserve_state`` is for accuracy evaluation.  CUDA requires warm-up
    executions before capture, but those executions would otherwise append
    dummy patches to the live cache.  Restoring the state afterwards leaves
    the graph ready to consume the first real online patch from precisely
    the same history as an eager engine.
    """
    eng = self.eng
    saved = None
    if preserve_state:
      saved = (
        eng.cache.key.clone(), eng.cache.value.clone(), eng.cache.slot_pos.clone(),
        eng.cache.write_ptr, eng.cache.next_pos, eng.cache.n_written,
        eng.stat_n.clone(), eng.stat_mu.clone(), eng.stat_sigma.clone(),
        eng.t_offset,
      )
    self.n.copy_(eng.stat_n)
    self.mu.copy_(eng.stat_mu)
    self.sigma.copy_(eng.stat_sigma)
    self.trend_m.copy_(eng.trend_m)
    self.trend_c.copy_(eng.trend_c)
    self.trend_apply.copy_(eng.trend_apply)
    self.slots.fill_(eng.cache.write_ptr)
    self.positions.fill_(eng.cache.next_pos)
    self.t_off.fill_(eng.t_offset)

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
    # leaves no side effect: only the warm-up iterations actually advanced
    # the ring.  Read the authoritative values back out of the device buffers
    # rather than trying to count passes.
    if saved is None:
      prev_next_pos = eng.cache.next_pos
      eng.cache.write_ptr = int(self.slots.item())
      eng.cache.next_pos = int(self.positions.item())
      eng.cache.n_written += eng.cache.next_pos - prev_next_pos
      eng.stat_n.copy_(self.n)
      eng.stat_mu.copy_(self.mu)
      eng.stat_sigma.copy_(self.sigma)
      eng.t_offset = int(self.t_off.item())
    else:
      (key, value, slot_pos, write_ptr, next_pos, n_written,
       n, mu, sigma, t_offset) = saved
      eng.cache.key.copy_(key)
      eng.cache.value.copy_(value)
      eng.cache.slot_pos.copy_(slot_pos)
      eng.cache.write_ptr = write_ptr
      eng.cache.next_pos = next_pos
      eng.cache.n_written = n_written
      eng.stat_n.copy_(n)
      eng.stat_mu.copy_(mu)
      eng.stat_sigma.copy_(sigma)
      eng.t_offset = t_offset
      self.n.copy_(n)
      self.mu.copy_(mu)
      self.sigma.copy_(sigma)
      self.slots.fill_(write_ptr)
      self.positions.fill_(next_pos)
      self.t_off.fill_(t_offset)

  # ------------------------------------------------------------ replay ----

  @torch.no_grad()
  def step(self, new_patch: torch.Tensor) -> torch.Tensor:
    """Replay the captured step on a new patch. Returns [B, V, H, nq]."""
    if self.graph is None:
      raise RuntimeError("call capture() first")
    self.x.copy_(new_patch.view_as(self.x))
    self.graph.replay()
    self.eng.cache.advance(1)
    self.eng.t_offset += self.eng.p  # host mirror of the in-graph advance
    return self.out


class CudaGraphFullDecode:
  """CUDA-graph replay for fixed-shape TimesFM-3.0 full recomputation.

  Upstream `model.decode` dynamically allocates every intermediate on every
  call, which CUDA Graph rejects.  This class instead invokes the rolling
  engine's T1-validated full-prefill forward (detrend fit included) on
  preallocated ring-buffer storage.  It writes every context token on every
  replay, so it is still a full recompute; it merely avoids the allocations
  that block graph capture.

  When ``rolling_target`` is a `CudaGraphRollingStep`, each replay also
  installs the complete refreshed state -- KV, slot tags, stats, detrend
  coefficients and the zeroed time offset -- into the rolling graph's fixed
  buffers, so switching back to rolling needs no eager synchronization or
  recapture.
  """

  def __init__(self, engine, rolling_target=None):
    if engine.nh_scratch != 0:
      raise ValueError(
        "CUDA graph full decode supports horizons that need no scratch "
        f"tokens; got horizon={engine.cfg.horizon}"
      )
    self.eng = engine
    self.rolling_target = rolling_target
    cfg = engine.cfg
    B, V, L, p, H = cfg.batch_size, cfg.num_variates, cfg.context_length, engine.p, cfg.horizon
    dev = cfg.device

    self.x = torch.zeros(B, V, L, device=dev, dtype=cfg.dtype)
    self.zero_pmask = torch.zeros(B, V, p, device=dev, dtype=torch.bool)
    self.slots = torch.arange(engine.n_patches, device=dev)
    self.positions = torch.arange(engine.n_patches, device=dev)
    self.n = torch.zeros(B, V, device=dev)
    self.mu = torch.zeros(B, V, device=dev)
    self.sigma = torch.zeros(B, V, device=dev)
    self.trend_m = torch.zeros(B, V, 1, device=dev)
    self.trend_c = torch.zeros(B, V, 1, device=dev)
    self.trend_apply = torch.zeros(B, V, 1, dtype=torch.bool, device=dev)
    self.t_hor = torch.arange(1, H + 1, dtype=torch.float32, device=dev)
    self.graph: torch.cuda.CUDAGraph | None = None
    self.out: torch.Tensor | None = None

  def _body(self):
    eng = self.eng
    B, V = eng.cfg.batch_size, eng.cfg.num_variates
    L = float(eng.cfg.context_length)

    if eng.use_linear_detrending:
      m_trend, c_trend, apply_detrend, ctx = eng._fit_trend(self.x)
      self.trend_m.copy_(m_trend)
      self.trend_c.copy_(c_trend)
      self.trend_apply.copy_(apply_detrend)
    else:
      self.trend_m.zero_()
      self.trend_c.zero_()
      self.trend_apply.fill_(False)
      ctx = self.x
    patches = ctx.view(B, V, eng.n_patches, eng.p)
    xd = torch.clamp(
      torch.nan_to_num(patches, nan=0.0), -eng.value_clip, eng.value_clip
    )

    # Every slot is overwritten below, and validity tags are rebuilt from zero.
    eng.cache.slot_pos.fill_(-1)
    self.n.zero_()
    self.mu.zero_()
    self.sigma.zero_()
    n, mu, sigma = self.n, self.mu, self.sigma
    mus, sigmas = [], []
    for i in range(eng.n_patches):
      n, mu, sigma = update_running_stats(
        n, mu, sigma, xd[:, :, i], self.zero_pmask
      )
      mus.append(mu)
      sigmas.append(sigma)
    mus, sigmas = torch.stack(mus, dim=2), torch.stack(sigmas, dim=2)
    normed = revin(xd, mus, sigmas, reverse=False).to(eng.cfg.dtype)

    tokens = eng._build_tokens(normed)
    emb = eng._encode_at(tokens, self.slots, self.positions)
    out = eng._readout_patches(
      emb[:, :, -1:], mus[:, :, -1:], sigmas[:, :, -1:]
    )
    hor = out[:, :, 0, : eng.cfg.horizon, :]
    if eng.use_linear_detrending:
      t_f = self.t_hor / L  # refresh anchors the trend at t_offset = 0
      trend = (
        self.trend_m[:, :, 0, None] * t_f[None, None, :]
        + self.trend_c[:, :, 0, None]
      )
      trend = torch.where(self.trend_apply[:, :, 0, None], trend, 0.0)
      hor = hor + trend[:, :, :, None]

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
      target.trend_m.copy_(self.trend_m)
      target.trend_c.copy_(self.trend_c)
      target.trend_apply.copy_(self.trend_apply)
      target.t_off.zero_()
      target.slots.zero_()
      target.positions.fill_(eng.n_patches)
    return hor

  @torch.no_grad()
  def capture(self, warmup: int = 3, preserve_target: bool = True) -> None:
    """Warm up and record the graph, leaving the eager state untouched.

    The warm-up iterations run `_body()` for real, overwriting the engine's
    live cache (K/V, slot tags) with an encoding of the zero-filled input
    buffer -- and, with a ``rolling_target``, the target's fixed buffers too.
    The engine cache is therefore always snapshotted and restored here;
    ``preserve_target`` additionally restores the rolling target's fixed
    buffers (set it False only when a `step()` follows immediately, which
    rewrites everything anyway).
    """
    eng = self.eng
    target = self.rolling_target
    saved_cache = (
      eng.cache.key.clone(), eng.cache.value.clone(), eng.cache.slot_pos.clone(),
    )
    saved_target_cache = None
    saved_target = None
    if target is not None and preserve_target:
      saved_target = (
        target.n.clone(), target.mu.clone(), target.sigma.clone(),
        target.trend_m.clone(), target.trend_c.clone(),
        target.trend_apply.clone(), target.t_off.clone(),
        target.slots.clone(), target.positions.clone(),
      )
      if target.eng.cache is not eng.cache:
        saved_target_cache = (
          target.eng.cache.key.clone(), target.eng.cache.value.clone(),
          target.eng.cache.slot_pos.clone(),
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

    # `_body` never touches the Python-side ring pointers (it encodes at the
    # fixed slots/positions with no `advance`), so restoring the tensors alone
    # returns the cache to its pre-capture state.
    key, value, slot_pos = saved_cache
    eng.cache.key.copy_(key)
    eng.cache.value.copy_(value)
    eng.cache.slot_pos.copy_(slot_pos)
    if saved_target_cache is not None:
      key, value, slot_pos = saved_target_cache
      target.eng.cache.key.copy_(key)
      target.eng.cache.value.copy_(value)
      target.eng.cache.slot_pos.copy_(slot_pos)
    if saved_target is not None:
      (n, mu, sigma, trend_m, trend_c, trend_apply, t_off, slots,
       positions) = saved_target
      target.n.copy_(n)
      target.mu.copy_(mu)
      target.sigma.copy_(sigma)
      target.trend_m.copy_(trend_m)
      target.trend_c.copy_(trend_c)
      target.trend_apply.copy_(trend_apply)
      target.t_off.copy_(t_off)
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
      self.rolling_target.eng.t_offset = 0
    return self.out
