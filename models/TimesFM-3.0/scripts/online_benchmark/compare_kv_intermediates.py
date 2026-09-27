"""Compare every TimesFM-3.0 layer's full-compute and rolling-cache K/V tensors.

This is the TimesFM-3.0 counterpart of
``models/TimesFM-2.5/scripts/online_benchmark/compare_kv_intermediates.py``
and follows its design one for one: one fixed synthetic series (same
generator, same seed), one refresh at cache age 0, then ``--evict-steps``
consecutive one-patch evictions; at every age the rolling cache is compared,
layer by layer and in logical (chronological) slot order, against a full
compute on the identical current window.

The two methods use different RoPE coordinate systems after the window moves:

* full compute renumbers the current window to positions ``0 .. N-1``;
* rolling cache retains monotone absolute positions ``age .. age+N-1``.

Consequently, raw cached keys are not directly comparable after an eviction.
This script reports both the raw difference (``k_direct``) and a position-
rebased difference (``k_rebased``).  TimesFM-3.0 caches
``key_ln(RoPE_p(k))`` where ``key_ln`` is an RMSNorm with a learned
elementwise weight ``w`` applied AFTER the rotation (PerDimScale touches the
queries only).  RoPE is split-half (dims ``i`` and ``i + head_dim/2`` form a
rotation pair) and preserves each head's sum of squares, so the RMS -- eps
included -- is the same at every position and

    stored(p) = w * RoPE_p(k) / sqrt(mean(k^2) + eps).

Rebasing therefore divides by ``w``, rotates by the position offset
``full - rolling`` (= ``-age``) and multiplies by ``w`` again (evaluated per
rotation pair, so a zero offset is the exact identity); this is exact up to
floating-point rounding, which ``validate_rebase_formula`` measures with the
same self-check as the TimesFM-2.5 probe.  Values carry no RoPE (and this
checkpoint has no value norm) and are compared directly.

What is "full compute" here
---------------------------
TimesFM-3.0 has no native DecodeCache prefill: upstream ``decode()`` is one
full-sequence forward that appends masked horizon (CPM) patches.  The K/V
reference is therefore a second ``RollingTimesFM3Engine`` that is fully
refreshed on the current window at every age -- the upstream-equivalent
prefill (fresh causal prefix RevIN stats, whole-window detrend refit,
positions ``0..N-1``) gated against ``decode()`` by T1a/T1b.  It shares the
rolling engine's code path and ring capacity, so at age 0 the two caches are
bitwise identical.  In addition, at every age the per-layer keys (output of
``seq_attn.key_ln``) and values (output of ``seq_attn.value_proj``) of the
context tokens are captured by forward hooks inside upstream
``TimesFM3Torch.decode`` and compared with the reference engine
(``full_reference_check``), which shows the reference is upstream up to
fp32 kernel-selection noise.

Forecast column
---------------
Median quantile (q = 0.5) over ``--horizon`` steps (default 128).  H = 128
exceeds TimesFM-3.0's stitching extract length (64), so:

* rolling = ``engine.forecast()``: the one-shot horizon-scratch fork encodes
  the 4 CPM patches inside a mark/rollback fork, runs the restarted CPM RevIN
  refinement, stitches, and re-adds the frozen trend;
* full    = upstream ``TimesFM3Torch.decode(window, horizon=128)``.

The JSON output contains per-age, per-layer, and per-scope metrics.  The CSV is
the flattened form intended for plotting or paper tables.

Example:
  python compare_kv_intermediates.py \
    --ckpt $ROLLKV_CKPT/TimesFM-3.0 --device cuda \
    --context-length 1024 --evict-steps 8 \
    --output-json results/timesfm3_kv_intermediates.json \
    --output-csv results/timesfm3_kv_intermediates.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from typing import Any

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from timesfm3.model import TimesFM3Torch  # noqa: E402
from timesfm3.online import RollingConfig, RollingTimesFM3Engine  # noqa: E402

MODEL_NAME = "TimesFM-3.0"

METRIC_FIELDS = (
  "mae",
  "rmse",
  "max_abs",
  "rel_l2",
  "normalized_mae",
  "cosine_distance",
)
SCOPE_ORDER = ("all", "survivors", "newest")
TENSOR_ORDER = ("k_direct", "k_rebased", "v")


def make_series(n: int, seed: int = 7) -> np.ndarray:
  """Deterministic multi-periodic signal with additive Gaussian noise.

  Identical to the TimesFM-2.5 probe's generator (same formula, same seed),
  so both models see the same synthetic series.
  """
  rng = np.random.RandomState(seed)
  t = np.arange(n, dtype=np.float32)
  return (
    np.sin(2 * np.pi * t / 96)
    + 0.5 * np.sin(2 * np.pi * t / 336)
    + 0.2 * rng.randn(n).astype(np.float32)
  ).astype(np.float32)


def _metric_tensor(rolling: torch.Tensor, full: torch.Tensor) -> torch.Tensor:
  """The TimesFM-2.5 probe's six metrics as one [6] tensor (no host sync).

  Same float32 ops, in the same order, as ``tensor_metrics`` of the 2.5
  probe; only the six ``.item()`` calls are deferred so a whole age can be
  transferred to the host in one copy.
  """
  rolling = rolling.detach().float()
  full = full.detach().float()
  diff = rolling - full
  eps = torch.finfo(torch.float32).eps
  diff_l2 = torch.linalg.vector_norm(diff)
  full_l2 = torch.linalg.vector_norm(full)
  rolling_l2 = torch.linalg.vector_norm(rolling)
  dot = torch.sum(rolling * full)
  return torch.stack(
    [
      diff.abs().mean(),
      diff.square().mean().sqrt(),
      diff.abs().max(),
      diff_l2 / full_l2.clamp_min(eps),
      diff.abs().mean() / full.abs().mean().clamp_min(eps),
      1.0 - dot / (rolling_l2 * full_l2).clamp_min(eps),
    ]
  )


def _as_metrics(values: list[float]) -> dict[str, float]:
  return {name: float(v) for name, v in zip(METRIC_FIELDS, values)}


def tensor_metrics(rolling: torch.Tensor, full: torch.Tensor) -> dict[str, float]:
  """Return scale-aware differences, treating ``full`` as the reference."""
  return _as_metrics(_metric_tensor(rolling, full).cpu().tolist())


def forecast_metrics(rolling: torch.Tensor, full: torch.Tensor) -> dict[str, float]:
  """``tensor_metrics`` plus the forecast MSE the paper prints in each cell."""
  metrics = tensor_metrics(rolling, full)
  diff = rolling.detach().float() - full.detach().float()
  metrics["mse"] = float(diff.square().mean().item())
  return metrics


def load_model(ckpt: str, device: str) -> tuple[TimesFM3Torch, str]:
  """Load from the checkpoint DIRECTORY so config.json is honoured."""
  ckpt_dir = ckpt
  if os.path.isfile(ckpt_dir):
    ckpt_dir = os.path.dirname(ckpt_dir)
  if not os.path.isfile(os.path.join(ckpt_dir, "config.json")):
    raise FileNotFoundError(
      f"{ckpt_dir} is not a TimesFM-3.0 checkpoint directory (no config.json)"
    )
  model = TimesFM3Torch.from_pretrained(ckpt_dir)
  model.to(device=device, dtype=torch.float32)
  model.eval()
  return model, ckpt_dir


def seq_attn_layers(model) -> list:
  return list(model.transformer_stack.layers)


def chronological_cache(engine: RollingTimesFM3Engine):
  """Read live slots in chronological rather than physical order.

  Returns key / value [n_layers, B*V, N, H, D] and the absolute positions [N].
  """
  cache = engine.cache
  lower = cache.next_pos - cache.window
  live = (cache.slot_pos >= lower) & (cache.slot_pos < cache.next_pos)
  slots = torch.nonzero(live, as_tuple=False).flatten()
  positions = cache.slot_pos.index_select(0, slots)
  order = torch.argsort(positions)
  slots = slots.index_select(0, order)
  positions = positions.index_select(0, order)
  if slots.numel() != engine.n_patches:
    raise RuntimeError(
      f"expected {engine.n_patches} live slots, found {slots.numel()}"
    )
  key = cache.key.index_select(2, slots)
  value = cache.value.index_select(2, slots)
  return key, value, positions


def _key_ln_scale(layer_idx: int, attn) -> torch.Tensor:
  if attn.key_ln is None or attn.rotary_position_embedding is None:
    raise RuntimeError(f"layer {layer_idx}: expected RoPE followed by key RMSNorm")
  scale = attn.key_ln.weight.detach().float()
  if torch.any(scale.abs() < 1e-12):
    raise RuntimeError(
      f"layer {layer_idx} key RMSNorm contains a near-zero weight; "
      "post-norm key rebasing is not invertible"
    )
  return scale


def rebase_post_norm_keys(
  layers,
  rolling_key: torch.Tensor,
  rolling_positions: torch.Tensor,
  full_positions: torch.Tensor,
) -> torch.Tensor:
  """Move post-RoPE/post-RMSNorm rolling K into full-compute coordinates.

  TimesFM-3.0's key RMSNorm (``seq_attn.key_ln``, torch ``nn.RMSNorm``)
  carries a learned per-dimension weight, which does not commute with RoPE
  (the weights are not symmetric within a rotation pair).  We therefore
  remove that weight, apply the relative rotation, and restore the weight
  for each layer, i.e. ``w * RoPE_delta(x / w)``.

  The rotation is the model's own split-half RoPE (same ``timescale`` buffer,
  same ``position.float() / timescale`` angles, pairs ``(i, i + D/2)``); it is
  written out per rotation pair so the weight enters as the ratio of the two
  pair weights:

      out_i     = x_i cos - (w_i / w_{i+D/2}) x_{i+D/2} sin
      out_{i+D/2} = x_{i+D/2} cos + (w_{i+D/2} / w_i) x_i sin

  which is ``w * RoPE_delta(x / w)`` re-associated.  At ``delta = 0``
  (cos = 1, sin = 0) it returns ``x`` bit for bit, where the literal
  ``(x / w) * w`` would leave 1-ulp residue; ``rebase_post_norm_keys_module``
  keeps the literal form and the self-check reports the gap between the two.

  rolling_key: [n_layers, B*V, N, H, D].
  """
  delta = full_positions - rolling_positions
  batch_leading = rolling_key.shape[1]
  position = delta[None, :].expand(batch_leading, -1)
  rebased = []
  for layer_idx, layer in enumerate(layers):
    attn = layer.seq_attn
    scale = _key_ln_scale(layer_idx, attn)
    key = rolling_key[layer_idx].float()
    rope = attn.rotary_position_embedding
    if rope.embedding_dims != key.shape[-1]:
      raise RuntimeError(f"layer {layer_idx}: RoPE dims != head_dim")
    timescale = rope.timescale.to(key.device)
    angle = position.unsqueeze(-1).unsqueeze(-1).float() / timescale.view(1, 1, 1, -1)
    sin, cos = torch.sin(angle), torch.cos(angle)
    w1, w2 = scale.chunk(2)
    x1, x2 = key.chunk(2, dim=-1)
    out1 = x1 * cos - (w1 / w2) * x2 * sin
    out2 = x2 * cos + (w2 / w1) * x1 * sin
    rebased.append(torch.cat([out1, out2], dim=-1))
  return torch.stack(rebased, dim=0)


def rebase_post_norm_keys_module(
  layers,
  rolling_key: torch.Tensor,
  rolling_positions: torch.Tensor,
  full_positions: torch.Tensor,
) -> torch.Tensor:
  """Literal TimesFM-2.5-probe form: ``RoPE_module(x / w, delta) * w``."""
  delta = full_positions - rolling_positions
  batch_leading = rolling_key.shape[1]
  position = delta[None, :].expand(batch_leading, -1)
  rebased = []
  for layer_idx, layer in enumerate(layers):
    attn = layer.seq_attn
    scale = _key_ln_scale(layer_idx, attn)
    unscaled = rolling_key[layer_idx].float() / scale.view(1, 1, 1, -1)
    rotated = attn.rotary_position_embedding(unscaled, position)
    rebased.append(rotated * scale.view(1, 1, 1, -1))
  return torch.stack(rebased, dim=0)


@torch.no_grad()
def validate_rebase_formula(model, device: str, rebase=None) -> float:
  """Numerically verify rebasing with one actual TimesFM-3.0 key-normalization.

  Same construction as the TimesFM-2.5 probe: three random raw keys encoded at
  positions 0..2 (full) and 5..7 (rolling) through layer 0's RoPE + key_ln;
  the rolling copy is rebased and compared with the full copy.
  """
  rebase = rebase_post_norm_keys if rebase is None else rebase
  layer = seq_attn_layers(model)[0]
  attn = layer.seq_attn
  generator = torch.Generator(device=device)
  generator.manual_seed(1234)
  raw = torch.randn(
    1, 3, attn.num_heads, attn.head_dim, device=device, generator=generator
  )
  full_pos = torch.arange(3, device=device)
  rolling_pos = full_pos + 5
  full = attn.key_ln(attn.rotary_position_embedding(raw, full_pos[None, :]))
  rolling = attn.key_ln(attn.rotary_position_embedding(raw, rolling_pos[None, :]))
  rebased = rebase([layer], rolling.unsqueeze(0), rolling_pos, full_pos)[0]
  return float((rebased - full).abs().max().item())


@torch.no_grad()
def validate_rebase_at_probe_shift(
  model, device: str, n_patches: int, shift: int
) -> dict[str, float]:
  """Extended self-check at the probe's own scale (all layers, N tokens).

  Random raw keys at positions 0..N-1 versus the same keys at
  ``shift..shift+N-1`` (the largest age of the run), rebased through every
  layer's key_ln weight.  Larger absolute positions mean larger fp32 RoPE
  angles, so this bounds the rebase noise over the whole probe.
  """
  layers = seq_attn_layers(model)
  attn0 = layers[0].seq_attn
  generator = torch.Generator(device=device)
  generator.manual_seed(4321)
  raw = torch.randn(
    1, n_patches, attn0.num_heads, attn0.head_dim, device=device, generator=generator
  )
  full_pos = torch.arange(n_patches, device=device)
  rolling_pos = full_pos + shift
  full_keys, rolling_keys = [], []
  for layer in layers:
    attn = layer.seq_attn
    full_keys.append(attn.key_ln(attn.rotary_position_embedding(raw, full_pos[None, :])))
    rolling_keys.append(
      attn.key_ln(attn.rotary_position_embedding(raw, rolling_pos[None, :]))
    )
  full = torch.stack(full_keys, dim=0)
  rolling = torch.stack(rolling_keys, dim=0)
  rebased = rebase_post_norm_keys(layers, rolling, rolling_pos, full_pos)
  literal = rebase_post_norm_keys_module(layers, rolling, rolling_pos, full_pos)
  diff = rebased - full
  diff_literal = literal - full
  identity = rebase_post_norm_keys(layers, full, full_pos, full_pos)
  return {
    "shift": int(shift),
    "n_tokens": int(n_patches),
    "n_layers": len(layers),
    "max_abs": float(diff.abs().max().item()),
    "rel_l2": float((diff.norm() / full.norm()).item()),
    "module_form_max_abs": float(diff_literal.abs().max().item()),
    "module_form_rel_l2": float((diff_literal.norm() / full.norm()).item()),
    "pair_vs_module_form_max_abs": float((rebased - literal).abs().max().item()),
    "module_form_self_check_max_abs": validate_rebase_formula(
      model, device, rebase_post_norm_keys_module
    ),
    "zero_shift_identity_max_abs": float((identity - full).abs().max().item()),
  }


class UpstreamKVCapture:
  """Forward hooks that record upstream ``decode()``'s per-layer K/V.

  Keys are the output of ``seq_attn.key_ln`` (post-RoPE, post-RMSNorm, the
  tensor upstream's DecodeCache would store); values are the output of
  ``seq_attn.value_ln`` when present, else of ``seq_attn.value_proj`` viewed
  as heads.  Capture is armed only around the decode call, because the
  rolling engine reuses the same modules.
  """

  def __init__(self, model):
    self.layers = seq_attn_layers(model)
    self.armed = False
    self.keys: list[torch.Tensor | None] = [None] * len(self.layers)
    self.values: list[torch.Tensor | None] = [None] * len(self.layers)
    self.handles = []
    for idx, layer in enumerate(self.layers):
      attn = layer.seq_attn
      self.handles.append(
        attn.key_ln.register_forward_hook(self._hook(idx, "key", attn))
      )
      value_module = attn.value_ln if attn.value_ln is not None else attn.value_proj
      self.handles.append(
        value_module.register_forward_hook(self._hook(idx, "value", attn))
      )

  def _hook(self, idx: int, kind: str, attn):
    def fn(_module, _inputs, output):
      if not self.armed:
        return
      out = output
      if out.dim() == 3:  # value_proj: [b*v, n, d] -> heads
        out = out.view(out.shape[0], out.shape[1], attn.num_heads, attn.head_dim)
      target = self.keys if kind == "key" else self.values
      target[idx] = out.detach().clone()

    return fn

  def stacked(self, n_context: int) -> tuple[torch.Tensor, torch.Tensor]:
    if any(k is None for k in self.keys) or any(v is None for v in self.values):
      raise RuntimeError("upstream K/V capture missed a layer")
    key = torch.stack([k[:, :n_context] for k in self.keys], dim=0)
    value = torch.stack([v[:, :n_context] for v in self.values], dim=0)
    self.keys = [None] * len(self.layers)
    self.values = [None] * len(self.layers)
    return key, value

  def remove(self) -> None:
    for handle in self.handles:
      handle.remove()


def compare_cache(
  layers,
  rolling_key: torch.Tensor,
  rolling_value: torch.Tensor,
  rolling_positions: torch.Tensor,
  full_key: torch.Tensor,
  full_value: torch.Tensor,
  full_positions: torch.Tensor,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
  """Compute global and per-layer cache metrics for one window."""
  n_layers = full_key.shape[0]
  n_patches = full_key.shape[2]
  rebased_key = rebase_post_norm_keys(
    layers, rolling_key, rolling_positions, full_positions
  )

  scopes: dict[str, slice] = {"all": slice(None)}
  if n_patches > 1:
    scopes.update({"survivors": slice(0, -1), "newest": slice(-1, None)})

  tensors = {
    "k_direct": (rolling_key, full_key),
    "k_rebased": (rebased_key, full_key),
    "v": (rolling_value, full_value),
  }
  # Evaluate everything on the device, then one host transfer per age.
  entries: list[tuple[str, str, int | None]] = []
  values: list[torch.Tensor] = []
  for scope_name, token_slice in scopes.items():
    for tensor_name, (rolling, full) in tensors.items():
      entries.append((scope_name, tensor_name, None))
      values.append(_metric_tensor(rolling[:, :, token_slice], full[:, :, token_slice]))
      for layer_idx in range(n_layers):
        entries.append((scope_name, tensor_name, layer_idx))
        values.append(
          _metric_tensor(
            rolling[layer_idx, :, token_slice], full[layer_idx, :, token_slice]
          )
        )
  host = torch.stack(values).cpu().tolist()

  global_metrics: dict[str, Any] = {}
  layer_metrics: list[dict[str, Any]] = [
    {"layer": layer_idx, "scopes": {}} for layer_idx in range(n_layers)
  ]
  for (scope_name, tensor_name, layer_idx), row in zip(entries, host):
    metrics = _as_metrics(row)
    if layer_idx is None:
      global_metrics.setdefault(scope_name, {})[tensor_name] = metrics
    else:
      layer_metrics[layer_idx]["scopes"].setdefault(scope_name, {})[
        tensor_name
      ] = metrics
  return global_metrics, layer_metrics


def flatten_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
  rows = []
  for step in results["steps"]:
    for layer in step["layers"]:
      for scope, tensors in layer["scopes"].items():
        for tensor_name, metrics in tensors.items():
          row = {
            "age": step["age"],
            "layer": layer["layer"],
            "scope": scope,
            "tensor": tensor_name,
          }
          row.update(metrics)
          rows.append(row)
  return rows


def save_outputs(results: dict[str, Any], json_path: str | None, csv_path: str | None):
  if json_path:
    json_path = os.path.abspath(json_path)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    tmp = f"{json_path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
      json.dump(results, f, indent=2)
    os.replace(tmp, json_path)
    print(f"saved JSON -> {json_path}")
  if csv_path:
    csv_path = os.path.abspath(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    rows = flatten_rows(results)
    tmp = f"{csv_path}.tmp.{os.getpid()}"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
      writer = csv.DictWriter(
        f, fieldnames=["age", "layer", "scope", "tensor", *METRIC_FIELDS]
      )
      writer.writeheader()
      writer.writerows(rows)
    os.replace(tmp, csv_path)
    print(f"saved CSV  -> {csv_path}")


def print_summary(results: dict[str, Any]) -> None:
  print(f"\n{'=' * 100}")
  print(f"{MODEL_NAME} full-compute vs rolling-cache intermediate K/V differences")
  print(f"{'=' * 100}")
  print(
    f"{'age':>4} {'forecast MAE':>13} {'forecast max':>13} {'fc relL2':>10} "
    f"{'Kdirect relL2':>15} {'Krebased relL2':>16} {'V relL2':>12}"
  )
  print("-" * 100)
  for step in results["steps"]:
    all_metrics = step["global"]["all"]
    print(
      f"{step['age']:>4} {step['forecast']['mae']:>13.4e} "
      f"{step['forecast']['max_abs']:>13.4e} "
      f"{step['forecast']['rel_l2']:>10.4e} "
      f"{all_metrics['k_direct']['rel_l2']:>15.4e} "
      f"{all_metrics['k_rebased']['rel_l2']:>16.4e} "
      f"{all_metrics['v']['rel_l2']:>12.4e}"
    )

  final = results["steps"][-1]
  print(f"\nPer-layer metrics at cache age k={final['age']} (scope=all tokens)")
  header = (
    f"{'layer':>5} {'Kdir relL2':>12} {'Kreb MAE':>11} {'Kreb max':>11} "
    f"{'Kreb relL2':>13} {'V MAE':>11} {'V max':>11} {'V relL2':>11}"
  )
  print(header)
  print("-" * len(header))
  for layer in final["layers"]:
    metrics = layer["scopes"]["all"]
    kd, kr, value = metrics["k_direct"], metrics["k_rebased"], metrics["v"]
    print(
      f"{layer['layer']:>5} {kd['rel_l2']:>12.4e} "
      f"{kr['mae']:>11.4e} {kr['max_abs']:>11.4e} {kr['rel_l2']:>13.4e} "
      f"{value['mae']:>11.4e} {value['max_abs']:>11.4e} "
      f"{value['rel_l2']:>11.4e}"
    )
  first = results["steps"][0]
  worst_k0 = max(
    first["global"][scope][tensor]["max_abs"]
    for scope in first["global"]
    for tensor in first["global"][scope]
  )
  print(
    f"\nk=0 agreement: max_abs over all K/V tensors = {worst_k0:.3e}; "
    f"forecast rel_l2 = {first['forecast']['rel_l2']:.3e}"
  )
  checks = [s["full_reference_check"] for s in results["steps"]]
  if checks and "key_max_abs" in checks[0]:
    print(
      "full reference (engine full refresh) vs upstream decode, worst over ages: "
      f"K max_abs {max(c['key_max_abs'] for c in checks):.3e}, "
      f"V max_abs {max(c['value_max_abs'] for c in checks):.3e}, "
      f"forecast max_abs {max(c['forecast_max_abs'] for c in checks):.3e}"
    )
  print(f"{'=' * 100}\n")


def _normalization(engine, reference) -> dict[str, Any]:
  """2.5's six RevIN fields plus TimesFM-3.0's frozen linear-detrend state."""
  r_mu = engine.last_mu[0, 0]
  f_mu = reference.last_mu[0, 0]
  r_sigma = engine.last_sigma[0, 0]
  f_sigma = reference.last_sigma[0, 0]
  r_m = engine.trend_m[0, 0, 0]
  f_m = reference.trend_m[0, 0, 0]
  r_c = engine.trend_c[0, 0, 0]
  f_c = reference.trend_c[0, 0, 0]
  vals = torch.stack(
    [
      r_mu, f_mu, (r_mu - f_mu).abs(), r_sigma, f_sigma, (r_sigma - f_sigma).abs(),
      r_m, f_m, (r_m - f_m).abs(), r_c, f_c, (r_c - f_c).abs(),
      engine.trend_apply[0, 0, 0].float(), reference.trend_apply[0, 0, 0].float(),
    ]
  ).cpu().tolist()
  return {
    "rolling_mu": vals[0],
    "full_mu": vals[1],
    "mu_abs_diff": vals[2],
    "rolling_sigma": vals[3],
    "full_sigma": vals[4],
    "sigma_abs_diff": vals[5],
    "rolling_trend_m": vals[6],
    "full_trend_m": vals[7],
    "trend_m_abs_diff": vals[8],
    "rolling_trend_c": vals[9],
    "full_trend_c": vals[10],
    "trend_c_abs_diff": vals[11],
    "rolling_detrend_applied": bool(vals[12]),
    "full_detrend_applied": bool(vals[13]),
    "rolling_trend_t_offset": int(engine.t_offset),
  }


@torch.no_grad()
def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--ckpt", required=True,
    help="TimesFM-3.0 checkpoint directory (config.json + model.safetensors); "
    "a model.safetensors path is accepted and its directory used",
  )
  parser.add_argument(
    "--device", default="cuda" if torch.cuda.is_available() else "cpu"
  )
  parser.add_argument("--context-length", type=int, default=1024)
  parser.add_argument("--horizon", type=int, default=128)
  parser.add_argument("--evict-steps", type=int, default=8)
  parser.add_argument("--seed", type=int, default=7)
  parser.add_argument(
    "--dtype", choices=["float32", "bfloat16"], default="float32",
    help="bfloat16 is refused: the vendored TimesFM-3.0 forward is fp32-only",
  )
  parser.add_argument(
    "--skip-upstream-kv-check", action="store_true",
    help="do not hook upstream decode() to cross-check the reference K/V",
  )
  parser.add_argument(
    "--progress-every", type=int, default=64,
    help="print one progress line every this many ages (0 = never)",
  )
  parser.add_argument("--output-json")
  parser.add_argument("--output-csv")
  args = parser.parse_args()

  if args.context_length < 32 or args.context_length % 32:
    raise ValueError("--context-length must be >=32 and divisible by 32")
  if args.horizon < 1 or args.horizon > 128:
    raise ValueError("this diagnostic requires 1 <= --horizon <= 128")
  if args.evict_steps < 1:
    raise ValueError("--evict-steps must be >= 1")
  if args.dtype != "float32":
    raise ValueError(
      "--dtype bfloat16 unsupported: the vendored TimesFM-3.0 forward is "
      "fp32-only (see test_exactness.py); run with --dtype float32"
    )
  if str(args.device).startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

  # True fp32 everywhere (these are torch's defaults for matmul; pinned so the
  # probe does not depend on the caller's global state).
  torch.backends.cuda.matmul.allow_tf32 = False
  torch.backends.cudnn.allow_tf32 = False
  torch.set_float32_matmul_precision("highest")

  dtype = torch.float32
  torch.manual_seed(0)
  model, ckpt_dir = load_model(args.ckpt, args.device)
  layers = seq_attn_layers(model)
  n_layers = len(layers)
  p = model.input_patch_len
  if args.context_length % p:
    raise ValueError(f"--context-length must be divisible by the patch size {p}")
  n_patches = args.context_length // p

  rebase_check = validate_rebase_formula(model, args.device)
  tolerance = 2e-5
  if rebase_check > tolerance:
    raise RuntimeError(
      f"key rebase self-check failed: max_abs={rebase_check:.3e}, "
      f"tolerance={tolerance:.3e}"
    )
  rebase_check_shift = validate_rebase_at_probe_shift(
    model, args.device, n_patches, args.evict_steps
  )

  length = args.context_length + args.evict_steps * p
  series = make_series(length, seed=args.seed)
  window = torch.as_tensor(
    series[: args.context_length], device=args.device, dtype=dtype
  )[None, None, :]  # [B=1, V=1, L]

  config = RollingConfig(
    context_length=args.context_length,
    horizon=args.horizon,
    full_refresh_every=0,
    batch_size=1,
    num_variates=1,
    device=args.device,
    dtype=dtype,
  )
  engine = RollingTimesFM3Engine(model, config)
  reference = RollingTimesFM3Engine(model, config)
  if engine.n_layers != n_layers:
    raise RuntimeError("engine / model layer count mismatch")
  median = engine.median_q_idx
  engine.full_refresh(window)

  capture = None if args.skip_upstream_kv_check else UpstreamKVCapture(model)
  full_positions = torch.arange(n_patches, device=args.device)

  device_name = args.device
  if str(args.device).startswith("cuda"):
    device_name = torch.cuda.get_device_name(torch.device(args.device))
  attn0 = layers[0].seq_attn
  results: dict[str, Any] = {
    "model": MODEL_NAME,
    "checkpoint": os.path.abspath(ckpt_dir),
    "config": {
      "context_length": args.context_length,
      "num_patches": n_patches,
      "patch_length": p,
      "horizon": args.horizon,
      "evict_steps": args.evict_steps,
      "seed": args.seed,
      "dtype": args.dtype,
      "device": "cuda" if str(args.device).startswith("cuda") else args.device,
    },
    "environment": {
      "python": platform.python_version(),
      "torch": torch.__version__,
      "device_name": device_name,
    },
    "protocol": {
      "full_compute": (
        "reference RollingTimesFM3Engine.full_refresh on the identical current "
        "window at every age: the upstream-equivalent prefill (fresh causal "
        "prefix RevIN stats, whole-window linear-detrend refit, positions "
        "0..N-1, same ring capacity), gated against TimesFM3Torch.decode by "
        "T1a/T1b; cross-checked every age against the K/V captured by forward "
        "hooks inside upstream decode() (full_reference_check in the raw JSON)"
      ),
      "rolling": (
        "ring-buffer cache refreshed once at age 0, then one-patch fast updates; "
        "RevIN prefix statistics keep accumulating since the refresh (frozen at "
        "encode) and the linear-detrend line (m, c, apply gate) stays frozen at "
        "the refresh fit; monotone absolute positions age..age+N-1"
      ),
      "alignment": (
        "live rolling slots sorted by absolute position and matched chronologically"
      ),
      "k_direct": (
        "raw cached keys: key_ln(RoPE(k)), i.e. post-RoPE/post-key-RMSNorm "
        "(RMSNorm with learned elementwise weight, applied after the rotation)"
      ),
      "k_rebased": (
        "rolling keys analytically rotated into full-compute relative RoPE positions "
        "(split-half RoPE, offset full - rolling = -age); learned key-RMSNorm weight "
        "is removed before rotation and restored after (w * RoPE(x / w), evaluated per "
        "rotation pair so offset 0 is the exact identity); RoPE preserves each head's "
        "sum of squares, so the RMS (eps included) is position independent and the "
        "transform is exact up to fp32 rounding"
      ),
      "v": "raw cached values (value_proj output, no value norm); no position transform is needed",
      "scopes": {
        "all": "all N tokens in the current cache",
        "survivors": "the first N-1 cached tokens reused by rolling",
        "newest": "the newly encoded final token",
      },
      "forecast": (
        f"median quantile (q=0.5) over H={args.horizon}; rolling = engine.forecast() "
        f"({engine.nh_scratch} horizon-scratch CPM patches encoded in one mark/rollback "
        "fork, restarted CPM RevIN refine, stitching, frozen-trend re-add); full = "
        f"upstream TimesFM3Torch.decode(window, horizon={args.horizon}) on the "
        "identical current window"
      ),
      "normalization": (
        "rolling_* = rolling engine state (RevIN stats of the newest token, frozen "
        "detrend line); full_* = reference full refresh on the current window"
      ),
    },
    "metric_definitions": {
      "mae": "mean(abs(rolling - full))",
      "rmse": "sqrt(mean((rolling - full)^2))",
      "max_abs": "max(abs(rolling - full))",
      "rel_l2": "||rolling - full||_2 / ||full||_2",
      "normalized_mae": "MAE / mean(abs(full))",
      "cosine_distance": "1 - cosine_similarity(rolling, full)",
      "mse": "mean((rolling - full)^2) (forecast only)",
    },
    "rebase_self_check_max_abs": rebase_check,
    "rebase_self_check_at_probe_shift": rebase_check_shift,
    "runtime": {
      "cuda": torch.version.cuda,
      "platform": platform.platform(),
      "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
      "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
      "attention": "sdpa" if attn0.use_sdpa else "manual softmax",
      "rescale_logits": bool(attn0.rescale_logits),
      "n_layers": n_layers,
      "num_heads": attn0.num_heads,
      "head_dim": attn0.head_dim,
      # nn.RMSNorm(eps=None) uses finfo(dtype).eps; record the value in force
      "key_ln_eps": (
        attn0.key_ln.eps
        if attn0.key_ln.eps is not None
        else torch.finfo(torch.float32).eps
      ),
      "key_ln_weight_min_abs": min(
        float(layer.seq_attn.key_ln.weight.abs().min()) for layer in layers
      ),
      "ring_capacity": engine.cache.capacity,
      "horizon_scratch_patches": engine.nh_scratch,
      "median_quantile_index": median,
      "upstream_kv_check": capture is not None,
    },
    "steps": [],
  }

  t_start = time.perf_counter()
  step_seconds: list[float] = []
  for age in range(args.evict_steps + 1):
    t_age = time.perf_counter()
    if age > 0:
      lo = args.context_length + (age - 1) * p
      new_patch = torch.as_tensor(
        series[lo : lo + p], device=args.device, dtype=dtype
      )[None, None, :]
      engine.fast_update(new_patch)
      window = torch.cat([window[:, :, p:], new_patch], dim=2)

    reference.full_refresh(window)
    full_key, full_value, ref_positions = chronological_cache(reference)
    if not torch.equal(ref_positions, full_positions):
      raise RuntimeError("reference full refresh is not at positions 0..N-1")
    rolling_key, rolling_value, positions = chronological_cache(engine)
    if int(positions[0].item()) != age or int(positions[-1].item()) != age + n_patches - 1:
      raise RuntimeError(f"age {age}: rolling positions {positions[0]}..{positions[-1]}")
    if not torch.equal(engine.raw_buffer, window):
      raise RuntimeError(f"age {age}: rolling raw buffer diverged from the window")
    global_metrics, layer_metrics = compare_cache(
      layers,
      rolling_key,
      rolling_value,
      positions,
      full_key,
      full_value,
      full_positions,
    )

    rolling_forecast = engine.forecast()[:, 0, :, median]  # [B, H]
    reference_forecast = reference.forecast()[:, 0, :, median]
    if capture is not None:
      capture.armed = True
    try:
      full_forecast = model.decode(target=window, horizon=args.horizon)[:, 0, :, median]
    finally:
      if capture is not None:
        capture.armed = False

    check: dict[str, Any] = {
      "forecast_max_abs": float((reference_forecast - full_forecast).abs().max().item()),
      "forecast_rel_l2": float(
        ((reference_forecast - full_forecast).norm() / full_forecast.norm()).item()
      ),
      "rolling_vs_full_refresh_forecast_max_abs": float(
        (rolling_forecast - reference_forecast).abs().max().item()
      ),
    }
    if capture is not None:
      up_key, up_value = capture.stacked(n_patches)
      check.update(
        {
          "key_max_abs": float((full_key - up_key).abs().max().item()),
          "key_rel_l2": float(((full_key - up_key).norm() / up_key.norm()).item()),
          "value_max_abs": float((full_value - up_value).abs().max().item()),
          "value_rel_l2": float(
            ((full_value - up_value).norm() / up_value.norm()).item()
          ),
        }
      )
      del up_key, up_value

    results["steps"].append(
      {
        "age": age,
        "rolling_position_first": int(positions[0].item()),
        "rolling_position_last": int(positions[-1].item()),
        "forecast": forecast_metrics(rolling_forecast, full_forecast),
        "normalization": _normalization(engine, reference),
        "full_reference_check": check,
        "global": global_metrics,
        "layers": layer_metrics,
      }
    )
    step_seconds.append(time.perf_counter() - t_age)
    if args.progress_every and (age % args.progress_every == 0 or age == args.evict_steps):
      g = global_metrics["all"]
      print(
        f"[probe] age {age:>5}/{args.evict_steps}  {step_seconds[-1]:.3f}s  "
        f"elapsed {time.perf_counter() - t_start:.1f}s  "
        f"Kreb {g['k_rebased']['rel_l2']:.4e}  V {g['v']['rel_l2']:.4e}  "
        f"fc {results['steps'][-1]['forecast']['rel_l2']:.4e}",
        flush=True,
      )

  if capture is not None:
    capture.remove()
  results["runtime"]["seconds_total"] = time.perf_counter() - t_start
  results["runtime"]["seconds_per_age_mean"] = float(np.mean(step_seconds))
  results["runtime"]["seconds_per_age_median"] = float(np.median(step_seconds))
  print_summary(results)
  save_outputs(results, args.output_json, args.output_csv)


if __name__ == "__main__":
  main()
