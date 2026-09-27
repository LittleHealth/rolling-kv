"""Mechanism experiments for the TimesFM-3.0 rolling KV cache (Exp1/2/3).

One driver computes, for every (target time T, channel, cache age k):

  Exp3  LADDER    roll -> R,abs^1row -> R,abs -> R,0 -> R,0+out -> R,0+new -> F' -> F
                  and path B  R,abs -> F,abs -> F'
  Exp2  FIX-ONE   per-layer / all-layer / all-but-one K/V swaps, cut-point h swaps,
                  positions (P2), STAT-in (P3), STAT-out (P4), TREND out (P5),
                  arm A (output end), arm B (new token, both ends), identity control
  Exp1  PROFILE   newest-token station metrics along the whole pipeline for the
                  pairs roll-vs-F', roll-vs-R,abs and R,abs^1row-vs-R,abs (NUM floor)

Fixed-target sampling: for each T and k the engine refreshes on the window that
ends k patches before T, then rolls k patches to T.  Every k therefore shares the
same current window W_T, the same references F'(T), F(T), and the same truth y.

Naming (see figures-draft/svg/mech/exp_design_timesfm3_zh.md):
  STALE  survivor K/V computed while evicted patches were still in the window
  POS    rolling positions k..k+N-1 vs full recompute 0..N-1
  STAT-in / STAT-kv / STAT-out   cumulative RevIN stats (new patch / survivors / output)
  TREND  frozen detrend line (new patch, survivors, output re-add)
  NUM    1-row vs N-row kernel numerics

All attention (rolling engine, references, model.decode) uses the explicit
softmax path (use_sdpa=False) in fp32 with TF32 disabled, eager mode.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(os.environ.get("ROLLKV_ROOT", Path(__file__).resolve().parents[3]))
CODE = ROOT / "code" if (ROOT / "code").exists() else Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE / "models" / "TimesFM-3.0" / "src"))

from timesfm3 import util  # noqa: E402
from timesfm3.model import TimesFM3Torch  # noqa: E402
from timesfm3.online import RollingConfig, RollingTimesFM3Engine  # noqa: E402

revin = util.revin
update_running_stats = util.update_running_stats

L_CTX = 8192
HORIZON = 64
KS_DEFAULT = (1, 4, 16, 64)

# Per-layer vector stations recorded for the newest token (row -1).
LAYER_STATIONS = (
    "x_in", "pre_seq_ln", "q_proj", "k_proj", "v_proj", "q_rope", "k_rope",
    "q_norm", "k_norm", "q_pds", "o", "wo", "u1", "h1", "var_pre_ln", "var_out",
    "u2", "h2", "ff_pre_ln", "ff0", "relu", "ff1", "u3", "h_out",
)
# Extra per-layer scalar metrics.
LAYER_EXTRA = ("logits_c", "tv_mean", "tv_max", "flip", "p_self_d", "p_old_d")
CROSS = ("c1", "c2", "c3")  # the three residual adds
# Input / output stations.
IO_STATIONS = ("detrend", "revin", "token", "z", "den", "yhat")
PAIRS = ("F", "Rabs", "floor")  # roll-vs-F', roll-vs-R,abs, R,abs^1row-vs-R,abs

LADDER = ("roll", "R1", "Rabs", "R0", "R0out", "R0new", "Fabs", "Fp")
ARMS_FIXED = ("identity", "P1all", "P2", "P3", "P4", "P5", "A", "B")


# ---------------------------------------------------------------------------
# Instrumented engine
# ---------------------------------------------------------------------------


class MechEngine(RollingTimesFM3Engine):
  """Rolling engine with station recording, h overrides and token overrides.

  Every op is identical to the parent (manual attention path); recording only
  clones the newest row after each op, so outputs are bit-identical with or
  without recording.
  """

  def __init__(self, model, cfg):
    super().__init__(model, cfg)
    B, V = cfg.batch_size, cfg.num_variates
    self.x_hist = torch.zeros(B, V, self.n_patches, self.p, device=cfg.device)
    self._xh_len = 0
    self.rec = None  # dict: 'L' -> list of per-layer dicts, 'h' -> list, 'io' -> dict
    self.h_override = None  # {layer_idx: [B, V, d]}; n_layers == final output
    self.token_override = None  # [B, V, 1, token_dim] replaces the new token
    for layer in self.layers:
      if layer.seq_attn.use_sdpa or layer.var_attn.use_sdpa:
        raise ValueError("MechEngine requires use_sdpa=False on all attention")

  # -- recording helpers ----------------------------------------------------
  def _R(self, layer_idx):
    if self.rec is None:
      return None
    return self.rec["L"][layer_idx]

  def start_rec(self):
    self.rec = {"L": [dict() for _ in range(self.n_layers)],
                "h": [None] * (self.n_layers + 1), "io": {}}

  def stop_rec(self):
    rec, self.rec = self.rec, None
    return rec

  # -- forward pass (verbatim math of the parent, manual attention) ---------
  def _attn_forward(self, layer_idx, attn, x, positions, mask, slots):
    R = self._R(layer_idx)
    BV, n, _ = x.shape
    query = attn.query_proj(x).view(BV, n, attn.num_heads, attn.head_dim)
    key = attn.key_proj(x).view(BV, n, attn.num_heads, attn.head_dim)
    value = attn.value_proj(x).view(BV, n, attn.num_heads, attn.head_dim)
    if R is not None:
      R["q_proj"] = query[:, -1].reshape(BV, -1).clone()
      R["k_proj"] = key[:, -1].reshape(BV, -1).clone()
      R["v_proj"] = value[:, -1].reshape(BV, -1).clone()
    if attn.rotary_position_embedding is not None:
      pos = positions[None, :]
      query = attn.rotary_position_embedding(query, pos)
      key = attn.rotary_position_embedding(key, pos)
    if R is not None:
      R["q_rope"] = query[:, -1].reshape(BV, -1).clone()
      R["k_rope"] = key[:, -1].reshape(BV, -1).clone()
    if attn.query_ln is not None:
      query = attn.query_ln(query)
    if attn.key_ln is not None:
      key = attn.key_ln(key)
    if R is not None:
      R["q_norm"] = query[:, -1].reshape(BV, -1).clone()
      R["k_norm"] = key[:, -1].reshape(BV, -1).clone()
    if attn.per_dim_scale is not None:
      query = attn.per_dim_scale(query)
    if R is not None:
      R["q_pds"] = query[:, -1].reshape(BV, -1).clone()
    if attn.value_ln is not None:
      value = attn.value_ln(value)

    self.cache.write_layer(layer_idx, slots, key, value)
    k_all = self.cache.key[layer_idx]
    v_all = self.cache.value[layer_idx]
    query = query.to(k_all.dtype)
    q_t = query.transpose(1, 2)
    k_t = k_all.transpose(1, 2)
    v_t = v_all.transpose(1, 2)
    attn_mask = mask.expand(BV, attn.num_heads, -1, -1)

    float_mask = torch.where(attn_mask, self._attn_zero, self._attn_neg)
    q_t = q_t * math.sqrt(attn.head_dim)
    if attn.rescale_logits:
      attn_logits = (
        torch.matmul(q_t, k_t.transpose(-2, -1)) / math.sqrt(attn.head_dim)
        + float_mask
      )
    else:
      attn_logits = torch.matmul(q_t, k_t.transpose(-2, -1)) + float_mask
    probs = F.softmax(attn_logits, dim=-1)
    out = torch.matmul(probs, v_t)
    if R is not None:
      R["logits"] = attn_logits[:, :, -1, :].clone()  # [BV, H, C]
      R["probs"] = probs[:, :, -1, :].clone()
      R["valid"] = mask[0, 0, -1, :].clone()  # [C]
      R["slot_pos"] = self.cache.slot_pos.clone()
      R["q_pos"] = int(positions[-1].item())

    out = out.transpose(1, 2).contiguous().view(BV, n, attn.in_features)
    if R is not None:
      R["o"] = out[:, -1].clone()
    y = attn.out_proj(out)
    if R is not None:
      R["wo"] = y[:, -1].clone()
    return y

  def _layer_forward(self, layer_idx, layer, x, positions, mask, slots):
    R = self._R(layer_idx)
    B, V, n, d = x.shape
    if R is not None:
      R["x_in"] = x[:, :, -1].reshape(B * V, d).clone()
    seq_in = layer.pre_seq_attn_ln(x).reshape(B * V, n, d)
    if R is not None:
      R["pre_seq_ln"] = seq_in[:, -1].clone()
    seq_out = self._attn_forward(layer_idx, layer.seq_attn, seq_in, positions, mask, slots)
    u1 = layer.post_seq_attn_ln(seq_out.view(B, V, n, d))
    h1 = u1 + x
    if R is not None:
      R["u1"] = u1[:, :, -1].reshape(B * V, d).clone()
      R["h1"] = h1[:, :, -1].reshape(B * V, d).clone()

    if layer.use_variate_attention:
      vpre = layer.pre_var_attn_ln(h1)
      var_in = vpre.permute(0, 2, 1, 3).reshape(B * n, V, d)
      var_out = self._var_attn_forward(layer.var_attn, var_in)
      var_out = var_out.view(B, n, V, d).permute(0, 2, 1, 3)
      u2 = layer.post_var_attn_ln(var_out)
      h2 = u2 + h1
      if R is not None:
        R["var_pre_ln"] = vpre[:, :, -1].reshape(B * V, d).clone()
        R["var_out"] = var_out[:, :, -1].reshape(B * V, d).clone()
        R["u2"] = u2[:, :, -1].reshape(B * V, d).clone()
    else:
      h2 = h1
    if R is not None:
      R["h2"] = h2[:, :, -1].reshape(B * V, d).clone()

    a = layer.pre_ff_ln(h2)
    f0 = layer.ff0(a)
    r = layer.activation(f0)
    f1 = layer.ff1(r)
    u3 = layer.post_ff_ln(f1)
    out = u3 + h2
    if R is not None:
      R["ff_pre_ln"] = a[:, :, -1].reshape(B * V, -1).clone()
      R["ff0"] = f0[:, :, -1].reshape(B * V, -1).clone()
      R["relu"] = r[:, :, -1].reshape(B * V, -1).clone()
      R["ff1"] = f1[:, :, -1].reshape(B * V, -1).clone()
      R["u3"] = u3[:, :, -1].reshape(B * V, d).clone()
      R["h_out"] = out[:, :, -1].reshape(B * V, d).clone()
    return out

  def _encode_at(self, tokens, slots, positions, window_anchor=None):
    self.cache.slot_pos.index_copy_(0, slots, positions)
    mask = self.cache.build_mask(positions, window_anchor)
    x = self.model.pre_transformer_resblock(tokens)
    if self.rec is not None:
      self.rec["h"][0] = x[:, :, -1].clone()
    ho = self.h_override
    for i, layer in enumerate(self.layers):
      if ho is not None and i in ho:
        x = x.clone()
        x[:, :, -1] = ho[i]
      x = self._layer_forward(i, layer, x, positions, mask, slots)
      if self.rec is not None:
        self.rec["h"][i + 1] = x[:, :, -1].clone()
    if ho is not None and self.n_layers in ho:
      x = x.clone()
      x[:, :, -1] = ho[self.n_layers]
    return x

  # -- ingestion with x history, io recording and token override ------------
  def _record_x(self, x):
    n_new = x.shape[2]
    W = self.n_patches
    if n_new >= W:
      self.x_hist.copy_(x[:, :, -W:])
      self._xh_len = W
    elif self._xh_len + n_new <= W:
      self.x_hist[:, :, self._xh_len:self._xh_len + n_new] = x
      self._xh_len += n_new
    else:
      keep = W - n_new
      self.x_hist[:, :, :keep] = self.x_hist[:, :, self._xh_len - keep:self._xh_len].clone()
      self.x_hist[:, :, keep:] = x
      self._xh_len = W

  def _ingest(self, patches):
    x = torch.clamp(torch.nan_to_num(patches, nan=0.0), -self.value_clip, self.value_clip)
    masks = torch.zeros_like(x, dtype=torch.bool)
    mu, sigma = self._advance_stats(x, masks)
    normed = revin(x, mu, sigma, reverse=False).to(self.cfg.dtype)
    tokens = self._build_tokens(normed)
    if self.token_override is not None:
      tokens = self.token_override
    self._record_tokens(tokens)
    self._record_x(x)
    if self.rec is not None:
      io = self.rec["io"]
      io["x"] = x[:, :, -1].clone()
      io["normed"] = normed[:, :, -1].clone()
      io["token"] = tokens[:, :, -1].clone()
    emb = self._encode(tokens)
    self.last_embedding = emb[:, :, -1:, :]
    self.last_mu, self.last_sigma = mu[..., -1], sigma[..., -1]
    return self._readout_patches(emb, mu, sigma)

  def full_refresh(self, raw_window):
    self._xh_len = 0
    return super().full_refresh(raw_window)


# ---------------------------------------------------------------------------
# State snapshot / restore (deep copies; never mark/rollback)
# ---------------------------------------------------------------------------

T_FIELDS = ("stat_n", "stat_mu", "stat_sigma", "trend_m", "trend_c", "trend_apply",
            "raw_buffer", "last_embedding", "last_mu", "last_sigma", "token_history",
            "x_hist")
I_FIELDS = ("t_offset", "_n_updates", "_hist_len", "_xh_len")


def snapshot(e):
  s = {f: getattr(e, f).clone() for f in T_FIELDS}
  s.update({f: getattr(e, f) for f in I_FIELDS})
  s["key"] = e.cache.key.clone()
  s["value"] = e.cache.value.clone()
  s["slot_pos"] = e.cache.slot_pos.clone()
  s["ptr"] = (e.cache.write_ptr, e.cache.next_pos, e.cache.n_written)
  return s


def restore(e, s):
  for f in T_FIELDS:
    setattr(e, f, s[f].clone())
  for f in I_FIELDS:
    setattr(e, f, s[f])
  e.cache.key.copy_(s["key"])
  e.cache.value.copy_(s["value"])
  e.cache.slot_pos = s["slot_pos"].clone()
  e.cache.write_ptr, e.cache.next_pos, e.cache.n_written = s["ptr"]
  e.rec = None
  e.h_override = None
  e.token_override = None


def encode_at(e, tokens, start, ring=False):
  """Encode tokens [B,V,n,192] into a fresh cache at positions start..start+n-1.

  ring=False: position start+i goes to slot i (logical order).
  ring=True : position P goes to slot P % capacity, the rolling engine's layout
              (fp32 attention sums run over slots in physical order, so the
              layout is part of NUM).
  """
  e.cache.reset()
  e.cache.next_pos = int(start)
  if ring:
    e.cache.write_ptr = int(start) % e.cache.capacity
  e._hist_len = 0
  return e._encode(tokens)


def fields_of(e):
  return dict(emb=e.last_embedding.clone(), mu=e.last_mu.clone(), sigma=e.last_sigma.clone(),
              tm=e.trend_m.clone(), tc=e.trend_c.clone(), ta=e.trend_apply.clone(),
              t_off=int(e.t_offset))


def readout(ro, emb, fl, mu=None, sigma=None, trend_from=None):
  """Forecast via the engine's own forecast() with the given readout fields.

  fl supplies mu/sigma/trend unless overridden; trend_from supplies (tm,tc,ta,t_off).
  Returns (yhat [B,V,H,nq], z [B,V,H,nq], T [B,V,H]).
  """
  tf = trend_from if trend_from is not None else fl
  emb = emb.contiguous()
  ro.last_embedding = emb
  ro.last_mu = fl["mu"] if mu is None else mu
  ro.last_sigma = fl["sigma"] if sigma is None else sigma
  ro.trend_m, ro.trend_c, ro.trend_apply, ro.t_offset = tf["tm"], tf["tc"], tf["ta"], tf["t_off"]
  yhat = ro.forecast()
  B, V = yhat.shape[:2]
  z = ro.model.output_head(emb).view(B, V, ro.o, ro.nq)[:, :, :HORIZON]
  t_f = (tf["t_off"] + torch.arange(1, HORIZON + 1, dtype=torch.float32, device=emb.device)) / float(L_CTX)
  T = tf["tm"][:, :, 0, None] * t_f[None, None, :] + tf["tc"][:, :, 0, None]
  T = torch.where(tf["ta"][:, :, 0, None], T, 0.0)
  return yhat, z, T


# ---------------------------------------------------------------------------
# K/V swaps and position shift
# ---------------------------------------------------------------------------


def survivor_map(e, k, N):
  """Physical slots of survivors (positions k..k+N-2) and their logical index p-k."""
  sp = e.cache.slot_pos
  surv = (sp >= k) & (sp <= k + N - 2)
  slots = torch.nonzero(surv).flatten()
  logical = sp[slots] - k
  return slots, logical


def swap_layers(e, layers, src_key, src_value, slots, logical):
  """src_* : [n_layers, BV, N, H, D] in logical order (index = position - k)."""
  for l in layers:
    e.cache.key[l][:, slots] = src_key[l][:, logical]
    e.cache.value[l][:, slots] = src_value[l][:, logical]


def shift_positions(e, k):
  """P2: move the live window to positions 0..N-1 (direct POS only)."""
  sp = e.cache.slot_pos
  e.cache.slot_pos = torch.where(sp >= 0, sp - k, sp)
  e.cache.next_pos -= k
  C = e.cache.capacity
  pos = torch.full((1, C), -float(k), device=sp.device)
  for l, layer in enumerate(e.layers):
    attn = layer.seq_attn
    w = attn.key_ln.weight
    K = e.cache.key[l]
    e.cache.key[l] = attn.rotary_position_embedding(K / w, pos) * w


# ---------------------------------------------------------------------------
# Station metrics
# ---------------------------------------------------------------------------


def _sq(a):
  return (a.double() ** 2).sum(-1)


def _logical(rec_l):
  valid = rec_l["valid"]
  sp = rec_l["slot_pos"]
  idx = torch.nonzero(valid).flatten()
  order = idx[torch.argsort(sp[idx])]
  return order


def pair_metrics(ra, rb, io_a, io_b, out_a, out_b, n_layers):
  """Metrics of run a against reference b for the newest token.

  Returns dict of numpy arrays:
    lay_num/lay_den [n_layers, S, BV]; extra [n_layers, E, BV];
    cross [n_layers, 3, 2, BV] (numerator 2<dh_b,du>, denominator |dh_a|^2);
    io_num/io_den [IO, BV].
  """
  S = len(LAYER_STATIONS)
  BV = ra["L"][0]["x_in"].shape[0]
  lay_num = torch.zeros(n_layers, S, BV, dtype=torch.float64)
  lay_den = torch.zeros(n_layers, S, BV, dtype=torch.float64)
  lay_dot = torch.zeros(n_layers, S, BV, dtype=torch.float64)
  lay_na = torch.zeros(n_layers, S, BV, dtype=torch.float64)
  extra = torch.zeros(n_layers, len(LAYER_EXTRA), BV, dtype=torch.float64)
  cross = torch.zeros(n_layers, 3, 2, BV, dtype=torch.float64)
  for l in range(n_layers):
    A, Bm = ra["L"][l], rb["L"][l]
    for si, st in enumerate(LAYER_STATIONS):
      a, b = A[st], Bm[st]
      lay_num[l, si] = _sq(a - b).cpu()
      lay_den[l, si] = _sq(b).cpu()
      lay_dot[l, si] = (a.double() * b.double()).sum(-1).cpu()
      lay_na[l, si] = _sq(a).cpu()
    oa, ob = _logical(A), _logical(Bm)
    if oa.numel() == ob.numel():
      la = A["logits"][:, :, oa].double()
      lb = Bm["logits"][:, :, ob].double()
      la = la - la.mean(-1, keepdim=True)
      lb = lb - lb.mean(-1, keepdim=True)
      extra[l, 0] = (_sq((la - lb).reshape(BV, -1)) / _sq(lb.reshape(BV, -1))).sqrt().cpu()
      pa = A["probs"][:, :, oa].double()
      pb = Bm["probs"][:, :, ob].double()
      tv = 0.5 * (pa - pb).abs().sum(-1)  # [BV, H]
      extra[l, 1] = tv.mean(-1).cpu()
      extra[l, 2] = tv.max(-1).values.cpu()
      extra[l, 4] = (pa[:, :, -1] - pb[:, :, -1]).mean(-1).cpu()  # share on self
      extra[l, 5] = (pa[:, :, 0] - pb[:, :, 0]).mean(-1).cpu()  # share on oldest
    else:
      extra[l, :3] = float("nan")
    extra[l, 3] = ((A["ff0"] > 0) != (Bm["ff0"] > 0)).double().mean(-1).cpu()
    for ci, (hb, du, ha) in enumerate((("x_in", "u1", "h1"), ("h1", "u2", "h2"), ("h2", "u3", "h_out"))):
      dhb = (A[hb] - Bm[hb]).double()
      ddu = (A[du] - Bm[du]).double()
      dha = (A[ha] - Bm[ha]).double()
      cross[l, ci, 0] = (2.0 * (dhb * ddu).sum(-1)).cpu()
      cross[l, ci, 1] = (dha ** 2).sum(-1).cpu()

  io_num = torch.zeros(len(IO_STATIONS), BV, dtype=torch.float64)
  io_den = torch.zeros(len(IO_STATIONS), BV, dtype=torch.float64)
  # detrend: |x_a - x_b| relative to |x_b - mu_b| (= sigma_b |normed_b|)
  xa, xb = io_a["x"].reshape(BV, -1), io_b["x"].reshape(BV, -1)
  mub = out_b["mu"].reshape(BV, 1)
  io_num[0], io_den[0] = _sq(xa - xb).cpu(), _sq(xb - mub).cpu()
  na, nb = io_a["normed"].reshape(BV, -1), io_b["normed"].reshape(BV, -1)
  io_num[1], io_den[1] = _sq(na - nb).cpu(), _sq(nb).cpu()
  ta, tb = io_a["token"].reshape(BV, -1), io_b["token"].reshape(BV, -1)
  io_num[2], io_den[2] = _sq(ta - tb).cpu(), _sq(tb[:, :nb.shape[-1]]).cpu()  # exclude constant mask dims
  za, zb = out_a["z"].reshape(BV, -1), out_b["z"].reshape(BV, -1)
  io_num[3], io_den[3] = _sq(za - zb).cpu(), _sq(zb).cpu()
  sb = out_b["sigma"].reshape(BV, 1).double()
  scale = (sb.squeeze(1) ** 2) * _sq(zb)  # (sigma_b |z_b|)^2
  dena = (out_a["yhat"] - out_a["T"][..., None]).reshape(BV, -1)
  denb = (out_b["yhat"] - out_b["T"][..., None]).reshape(BV, -1)
  io_num[4], io_den[4] = _sq(dena - denb).cpu(), scale.cpu()
  io_num[5], io_den[5] = _sq(out_a["yhat"].reshape(BV, -1) - out_b["yhat"].reshape(BV, -1)).cpu(), scale.cpu()
  return dict(lay_num=lay_num.numpy(), lay_den=lay_den.numpy(), lay_dot=lay_dot.numpy(),
              lay_na=lay_na.numpy(), extra=extra.numpy(),
              cross=cross.numpy(), io_num=io_num.numpy(), io_den=io_den.numpy())


def zero_io(io_a):
  """R-references reuse the rolling token, so input stations are identical."""
  return {k: v.clone() for k, v in io_a.items()}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def load_weather(path):
  import pandas as pd

  df = pd.read_csv(path)
  cols = [c for c in df.columns if c != "date"]
  vals = df[cols].astype(np.float64)
  vals = vals.mask(vals <= -9000.0)
  n_bad = int(vals.isna().sum().sum())
  vals = vals.interpolate(method="linear", limit_direction="both")
  arr = vals.to_numpy(dtype=np.float32).T.copy()  # [C, T]
  return arr, cols, n_bad


def target_times(n_points, k_max, p, spacing, count):
  t_min = L_CTX + k_max * p
  t_max = n_points - HORIZON
  ts = list(range(t_min, t_max + 1, spacing))
  if count and count < len(ts):
    ts = ts[:count]
  return ts


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run(args):
  torch.backends.cuda.matmul.allow_tf32 = False
  torch.backends.cudnn.allow_tf32 = False
  torch.set_grad_enabled(False)
  dev = torch.device(f"cuda:{args.gpu}")
  torch.cuda.set_device(dev)

  ckpt = Path(os.environ.get("ROLLKV_CKPT", ROOT / "checkpoints")) / "TimesFM-3.0"
  model = TimesFM3Torch.from_pretrained(str(ckpt))
  model.to(device=dev, dtype=torch.float32)
  model.eval()
  n_attn = 0
  for m in model.modules():
    if hasattr(m, "use_sdpa"):
      m.use_sdpa = False
      n_attn += 1

  data_path = Path(os.environ.get("ROLLKV_DATASETS", ROOT / "datasets")) / "weather" / "weather.csv"
  series, channels, n_bad = load_weather(data_path)
  C, n_points = series.shape
  B = C
  p = model.input_patch_len
  N = L_CTX // p
  ks = tuple(args.ks)
  k_max = max(ks)
  all_ts = target_times(n_points, k_max, p, args.spacing, args.count)
  my_ts = all_ts[args.shard::args.nshards]
  if args.limit:
    my_ts = my_ts[: args.limit]

  cfg = RollingConfig(context_length=L_CTX, horizon=HORIZON, full_refresh_every=0,
                      batch_size=B, num_variates=1, device=str(dev), dtype=torch.float32)
  main = MechEngine(model, cfg)
  work = MechEngine(model, cfg)
  ref = MechEngine(model, cfg)
  fref = MechEngine(model, cfg)
  assert main.nh_scratch == 0 and main.cache.capacity == N, (main.nh_scratch, main.cache.capacity)
  n_layers = main.n_layers
  nq = main.nq
  qm = main.median_q_idx
  key_ln_min = min(float(l.seq_attn.key_ln.weight.abs().min()) for l in main.layers)

  ser = torch.from_numpy(series).to(dev)

  nT, nK = len(my_ts), len(ks)
  n_arm_layer = n_layers
  arms_point = {a: np.zeros((nT, nK, B, HORIZON), np.float32) for a in ARMS_FIXED}
  for pre in ("P1l", "ABl"):
    arms_point[pre] = np.zeros((nT, nK, n_arm_layer, B, HORIZON), np.float32)
  arms_point["CUT"] = np.zeros((nT, nK, n_layers + 1, B, HORIZON), np.float32)
  arms_full = {a: np.zeros((nT, nK, B, HORIZON, nq), np.float32) for a in ("P1all", "P2", "P3", "P4", "P5", "A", "B")}
  ladder = {r: np.zeros((nT, nK, B, HORIZON, nq), np.float32) for r in LADDER}
  out4 = np.zeros((nT, nK, 4, B, HORIZON, nq), np.float32)  # net, scale, level, trend (vs F')
  Fdec = np.zeros((nT, B, HORIZON, nq), np.float32)
  Fp = np.zeros((nT, B, HORIZON, nq), np.float32)
  Y = np.zeros((nT, B, HORIZON), np.float32)
  win_std = np.zeros((nT, B), np.float32)
  stats = np.zeros((nT, nK, 6, B), np.float32)  # mu_R, sig_R, mu_F, sig_F, mu_ex, sig_ex
  gates = np.zeros((nT, nK, 2, B), np.bool_)  # apply_R (frozen), apply_F (refit)
  prof = {pr: dict(lay_num=np.zeros((nT, nK, n_layers, len(LAYER_STATIONS), B), np.float32),
                   lay_den=np.zeros((nT, nK, n_layers, len(LAYER_STATIONS), B), np.float32),
                   lay_dot=np.zeros((nT, nK, n_layers, len(LAYER_STATIONS), B), np.float64),
                   lay_na=np.zeros((nT, nK, n_layers, len(LAYER_STATIONS), B), np.float64),
                   extra=np.zeros((nT, nK, n_layers, len(LAYER_EXTRA), B), np.float32),
                   cross=np.zeros((nT, nK, n_layers, 3, 2, B), np.float64),
                   io_num=np.zeros((nT, nK, len(IO_STATIONS), B), np.float64),
                   io_den=np.zeros((nT, nK, len(IO_STATIONS), B), np.float64))
          for pr in PAIRS}
  ctrl = {
    "k0_roll_vs_Fp_maxabs": np.zeros(nT), "k0_Rabs_vs_Fp_maxabs": np.zeros(nT),
    "readout_vs_forecast_maxabs": np.zeros(nT),
    "identity_vs_roll_maxabs": np.zeros((nT, nK)), "cut_last_vs_Rabs_maxabs": np.zeros((nT, nK)),
    "p2_rot_roundtrip_rel": np.zeros((nT, nK)), "yscale_maxabs": np.zeros(nT),
    "roll_readout_vs_forecast_maxabs": np.zeros((nT, nK)),
    "p2_layer0_vs_R0_rel": np.zeros((nT, nK)),
  }
  timing = []

  t_run0 = time.time()
  for ti, T in enumerate(my_ts):
    t0 = time.time()
    window = ser[:, T - L_CTX:T].unsqueeze(1).contiguous()  # [B,1,L]
    Y[ti] = series[:, T:T + HORIZON]
    win_std[ti] = series[:, T - L_CTX:T].std(axis=1)

    # ---- references on the current window: F' (engine) and F (decode) -----
    fref.start_rec()
    fref.full_refresh(window)
    rec_F = fref.stop_rec()
    yFp_native = fref.forecast()
    fl_F = fields_of(fref)
    tok_F = fref.token_history.clone()
    yFp, zF, TF = readout(ref, fl_F["emb"], fl_F)
    ctrl["readout_vs_forecast_maxabs"][ti] = float((yFp - yFp_native).abs().max())
    out_F = dict(z=zF, T=TF, yhat=yFp, mu=fl_F["mu"], sigma=fl_F["sigma"])
    Fp[ti] = yFp[:, 0].cpu().numpy()
    Fdec[ti] = model.decode(window, horizon=HORIZON)[:, 0].cpu().numpy()
    ctrl["yscale_maxabs"][ti] = float(yFp.abs().max())

    # ---- k = 0 controls -------------------------------------------------------
    main.full_refresh(window)
    y0 = main.forecast()
    ctrl["k0_roll_vs_Fp_maxabs"][ti] = float((y0 - yFp_native).abs().max())
    emb0 = encode_at(ref, main.token_history.clone(), 0)[:, :, -1:]
    y0r, _, _ = readout(ref, emb0, fields_of(main))
    ctrl["k0_Rabs_vs_Fp_maxabs"][ti] = float((y0r - yFp).abs().max())

    for ki, k in enumerate(ks):
      # ---- rolling trajectory: refresh k patches before T, roll k patches --
      start = T - k * p
      main.full_refresh(ser[:, start - L_CTX:start].unsqueeze(1).contiguous())
      for j in range(k - 1):
        main.fast_update(ser[:, start + j * p:start + (j + 1) * p].unsqueeze(1))
      S_pre = snapshot(main)
      new_patch = ser[:, T - p:T].unsqueeze(1)
      main.start_rec()
      main.fast_update(new_patch)
      rec_roll = main.stop_rec()
      y_roll_native = main.forecast()
      fl_R = fields_of(main)
      tok_R = main.token_history.clone()
      y_roll, zR, TR = readout(ref, fl_R["emb"], fl_R)
      ctrl["roll_readout_vs_forecast_maxabs"][ti, ki] = float((y_roll - y_roll_native).abs().max())
      out_R = dict(z=zR, T=TR, yhat=y_roll, mu=fl_R["mu"], sigma=fl_R["sigma"])
      assert main.cache.next_pos == N + k
      assert torch.equal(main.raw_buffer, window)
      gates[ti, ki, 0] = fl_R["ta"][:, 0, 0].cpu().numpy()
      gates[ti, ki, 1] = fl_F["ta"][:, 0, 0].cpu().numpy()

      # ---- ladder references ------------------------------------------------
      ref.start_rec()
      e_abs = encode_at(ref, tok_R, k)
      rec_abs = ref.stop_rec()
      Rabs_key = ref.cache.key.clone()
      Rabs_val = ref.cache.value.clone()
      assert bool((ref.cache.slot_pos == torch.arange(k, k + N, device=dev)).all())
      y_abs, z_abs, T_abs = readout(ref, e_abs[:, :, -1:], fl_R)
      out_abs = dict(z=z_abs, T=T_abs, yhat=y_abs, mu=fl_R["mu"], sigma=fl_R["sigma"])

      encode_at(ref, tok_R[:, :, :N - 1], k, ring=True)
      ref.start_rec()
      e_1 = ref._encode(tok_R[:, :, N - 1:])
      rec_1 = ref.stop_rec()
      assert torch.equal(ref.cache.slot_pos, main.cache.slot_pos)  # same ring layout as roll
      y_1, z_1, T_1 = readout(ref, e_1[:, :, -1:], fl_R)
      out_1 = dict(z=z_1, T=T_1, yhat=y_1, mu=fl_R["mu"], sigma=fl_R["sigma"])

      e_0 = encode_at(ref, tok_R, 0)[:, :, -1:].clone()
      R0_key0 = ref.cache.key[0].clone()  # slots 0..N-1 <-> positions 0..N-1
      y_R0, z_R0, _ = readout(ref, e_0, fl_R)
      y_R0out, _, _ = readout(ref, e_0, fl_F)
      e_fabs = encode_at(ref, tok_F, k)[:, :, -1:].clone()
      y_Fabs, _, _ = readout(ref, e_fabs, fl_F)
      tok_mix = torch.cat([tok_R[:, :, :N - 1], tok_F[:, :, N - 1:]], dim=2)
      e_new = encode_at(ref, tok_mix, 0)[:, :, -1:].clone()
      y_R0new, _, _ = readout(ref, e_new, fl_F)

      for name, y in (("roll", y_roll), ("R1", y_1), ("Rabs", y_abs), ("R0", y_R0),
                      ("R0out", y_R0out), ("R0new", y_R0new), ("Fabs", y_Fabs), ("Fp", yFp)):
        ladder[name][ti, ki] = y[:, 0].cpu().numpy()

      # output-end 4-term split vs F'
      sR, sF = fl_R["sigma"][:, :, None, None], fl_F["sigma"][:, :, None, None]
      mR, mF = fl_R["mu"][:, :, None, None], fl_F["mu"][:, :, None, None]
      out4[ti, ki, 0] = (sR * (zR - zF))[:, 0].cpu().numpy()
      out4[ti, ki, 1] = ((sR - sF) * zF)[:, 0].cpu().numpy()
      out4[ti, ki, 2] = (mR - mF).expand_as(zF)[:, 0].cpu().numpy()
      out4[ti, ki, 3] = (TR - TF)[:, :, :, None].expand_as(zF)[:, 0].cpu().numpy()

      # ---- Exp1 station metrics ---------------------------------------------
      io_R = rec_roll["io"]
      for pr, (ra, rb, ia, ib, oa, ob) in (
          ("F", (rec_roll, rec_F, io_R, rec_F["io"], out_R, out_F)),
          ("Rabs", (rec_roll, rec_abs, io_R, zero_io(io_R), out_R, out_abs)),
          ("floor", (rec_1, rec_abs, io_R, zero_io(io_R), out_1, out_abs))):
        m = pair_metrics(ra, rb, ia, ib, oa, ob, n_layers)
        for key_, val in m.items():
          prof[pr][key_][ti, ki] = val

      # ---- Exp2 arms on copies of the pre-update state ----------------------
      def arm_forecast(setup=None, post=None):
        restore(work, S_pre)
        if setup is not None:
          setup(work)
        work.fast_update(new_patch)
        if post is not None:
          post(work)
        return readout(ref, work.last_embedding, fields_of(work))[0]

      restore(work, S_pre)
      slots, logical = survivor_map(work, k, N)
      assert slots.numel() == N - 1
      own_key = torch.zeros_like(Rabs_key)
      own_val = torch.zeros_like(Rabs_val)
      own_key[:, :, logical] = S_pre["key"][:, :, slots]
      own_val[:, :, logical] = S_pre["value"][:, :, slots]

      def med(y):
        return y[:, 0, :, qm].cpu().numpy()

      all_l = list(range(n_layers))
      y_id = arm_forecast(lambda e: swap_layers(e, all_l, own_key, own_val, slots, logical))
      ctrl["identity_vs_roll_maxabs"][ti, ki] = float((y_id - y_roll).abs().max())
      arms_point["identity"][ti, ki] = med(y_id)
      del own_key, own_val

      y_p1 = arm_forecast(lambda e: swap_layers(e, all_l, Rabs_key, Rabs_val, slots, logical))
      arms_point["P1all"][ti, ki] = med(y_p1)
      arms_full["P1all"][ti, ki] = y_p1[:, 0].cpu().numpy()
      for l in range(n_layers):
        y_l = arm_forecast(lambda e, l=l: swap_layers(e, [l], Rabs_key, Rabs_val, slots, logical))
        arms_point["P1l"][ti, ki, l] = med(y_l)
        others = [m_ for m_ in all_l if m_ != l]
        y_b = arm_forecast(lambda e, o=others: swap_layers(e, o, Rabs_key, Rabs_val, slots, logical))
        arms_point["ABl"][ti, ki, l] = med(y_b)
      for l in range(n_layers + 1):
        hl = rec_abs["h"][l]

        def set_cut(e, l=l, hl=hl):
          e.h_override = {l: hl}

        def clear(e):
          e.h_override = None

        y_c = arm_forecast(set_cut, clear)
        arms_point["CUT"][ti, ki, l] = med(y_c)
        if l == n_layers:
          ctrl["cut_last_vs_Rabs_maxabs"][ti, ki] = float((y_c - y_abs).abs().max())

      # P2 positions (direct POS)
      restore(work, S_pre)
      K0 = work.cache.key[0].clone()
      shift_positions(work, k)
      pos_back = torch.full((1, work.cache.capacity), float(k), device=dev)
      w0 = work.layers[0].seq_attn.key_ln.weight
      K0b = work.layers[0].seq_attn.rotary_position_embedding(work.cache.key[0] / w0, pos_back) * w0
      live = S_pre["slot_pos"] >= 0
      ctrl["p2_rot_roundtrip_rel"][ti, ki] = float(
        (K0b[:, live] - K0[:, live]).norm() / K0[:, live].norm())
      s2, lg2 = survivor_map(work, 0, N)  # after the shift survivors sit at 0..N-2
      ctrl["p2_layer0_vs_R0_rel"][ti, ki] = float(
        (work.cache.key[0][:, s2] - R0_key0[:, lg2]).norm() / R0_key0[:, lg2].norm())
      work.fast_update(new_patch)
      y_p2 = readout(ref, work.last_embedding, fields_of(work))[0]
      arms_point["P2"][ti, ki] = med(y_p2)
      arms_full["P2"][ti, ki] = y_p2[:, 0].cpu().numpy()

      # P3 STAT-in: new-patch stats over the current N patches (frozen frame)
      restore(work, S_pre)
      xs = S_pre["x_hist"][:, :, 1:]
      n_ = torch.zeros(B, 1, device=dev)
      mu_ = torch.zeros(B, 1, device=dev)
      sg_ = torch.zeros(B, 1, device=dev)
      zmask = torch.zeros(B, 1, p, dtype=torch.bool, device=dev)
      for i in range(N - 1):
        n_, mu_, sg_ = update_running_stats(n_, mu_, sg_, xs[:, :, i], zmask)
      work.stat_n, work.stat_mu, work.stat_sigma = n_, mu_, sg_
      work.fast_update(new_patch)
      mu_ex, sg_ex = work.last_mu.clone(), work.last_sigma.clone()
      y_p3 = readout(ref, work.last_embedding, fl_R)[0]
      arms_point["P3"][ti, ki] = med(y_p3)
      arms_full["P3"][ti, ki] = y_p3[:, 0].cpu().numpy()
      stats[ti, ki] = torch.stack([fl_R["mu"], fl_R["sigma"], fl_F["mu"], fl_F["sigma"],
                                   mu_ex, sg_ex])[:, :, 0].cpu().numpy()

      # P4 STAT-out, P5 TREND out, arm A: algebraic readouts of the rolling h_20
      y_p4, _, _ = readout(ref, fl_R["emb"], fl_R, mu=mu_ex, sigma=sg_ex)
      y_p5, _, _ = readout(ref, fl_R["emb"], fl_R, trend_from=fl_F)
      y_A, _, _ = readout(ref, fl_R["emb"], fl_F)
      # arm B: new token built the full-recompute way, readout with F's stats/trend
      restore(work, S_pre)
      work.token_override = tok_F[:, :, N - 1:].contiguous()
      work.fast_update(new_patch)
      work.token_override = None
      y_B, _, _ = readout(ref, work.last_embedding.clone(), fl_F)
      for name, y in (("P4", y_p4), ("P5", y_p5), ("A", y_A), ("B", y_B)):
        arms_point[name][ti, ki] = med(y)
        arms_full[name][ti, ki] = y[:, 0].cpu().numpy()

      del S_pre, Rabs_key, Rabs_val, rec_roll, rec_abs, rec_1, R0_key0
    torch.cuda.synchronize()
    dt = time.time() - t0
    timing.append(dt)
    print(f"[shard {args.shard}] T={T} ({ti + 1}/{nT}) {dt:.1f}s  "
          f"k0={ctrl['k0_roll_vs_Fp_maxabs'][ti]:.1e} id={ctrl['identity_vs_roll_maxabs'][ti].max():.1e} "
          f"cut={ctrl['cut_last_vs_Rabs_maxabs'][ti].max():.1e}", flush=True)

  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  save = dict(targets=np.array(my_ts), ks=np.array(ks), Y=Y, Fdec=Fdec, Fp=Fp, win_std=win_std,
              stats=stats, gates=gates, out4=out4, timing=np.array(timing))
  for r, v in ladder.items():
    save[f"lad_{r}"] = v
  for a, v in arms_point.items():
    save[f"armp_{a}"] = v
  for a, v in arms_full.items():
    save[f"armf_{a}"] = v
  for pr, d in prof.items():
    for key_, v in d.items():
      save[f"prof_{pr}_{key_}"] = v
  for c, v in ctrl.items():
    save[f"ctrl_{c}"] = v
  np.savez_compressed(out / f"shard{args.shard:02d}.npz", **save)
  meta = dict(channels=channels, n_bad_interpolated=n_bad, L=L_CTX, N=N, p=p, H=HORIZON,
              ks=list(ks), n_targets_total=len(all_ts), spacing=args.spacing,
              shard=args.shard, nshards=args.nshards, targets=my_ts, n_attn_modules=n_attn,
              key_ln_min_abs=key_ln_min, tf32=False, dtype="float32", attention="manual softmax",
              layer_stations=LAYER_STATIONS, layer_extra=LAYER_EXTRA, io_stations=IO_STATIONS,
              pairs=PAIRS, ladder=LADDER, arms_fixed=ARMS_FIXED,
              torch=torch.__version__, gpu=torch.cuda.get_device_name(dev),
              seconds=time.time() - t_run0)
  (out / f"shard{args.shard:02d}.json").write_text(json.dumps(meta, indent=1))
  print(f"[shard {args.shard}] done in {time.time() - t_run0:.0f}s", flush=True)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--gpu", type=int, default=1)
  ap.add_argument("--shard", type=int, default=0)
  ap.add_argument("--nshards", type=int, default=1)
  ap.add_argument("--ks", type=int, nargs="+", default=list(KS_DEFAULT))
  ap.add_argument("--spacing", type=int, default=144)
  ap.add_argument("--count", type=int, default=0, help="0 = all target times")
  ap.add_argument("--limit", type=int, default=0, help="max targets for this shard (smoke)")
  ap.add_argument("--out", default=str(Path(os.environ.get("ROLLKV_RESULTS", "results")) / "MECH_timesfm3"))
  run(ap.parse_args())


if __name__ == "__main__":
  main()
