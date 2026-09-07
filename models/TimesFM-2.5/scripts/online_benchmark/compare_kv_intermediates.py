"""Compare every TimesFM layer's full-compute and rolling-cache K/V tensors.

The two methods use different RoPE coordinate systems after the window moves:

* full compute renumbers the current window to positions ``0 .. N-1``;
* rolling cache retains monotone absolute positions ``age .. age+N-1``.

Consequently, raw cached keys are not directly comparable after an eviction.
This script reports both the raw difference (``k_direct``) and a position-
rebased difference (``k_rebased``).  Rebasing analytically moves each rolling
key into the full-compute RoPE coordinates while preserving TimesFM's learned
per-dimension key RMSNorm scale.  Values do not contain RoPE and are compared
directly.

The JSON output contains per-age, per-layer, and per-scope metrics.  The CSV is
the flattened form intended for plotting or paper tables.

Example:
  python compare_kv_intermediates.py \
    --ckpt /path/to/model.safetensors --device cuda \
    --context-length 1024 --evict-steps 8 \
    --output-json results/timesfm_kv_intermediates.json \
    --output-csv results/timesfm_kv_intermediates.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
from typing import Any

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from timesfm.online import RollingConfig, RollingTimesFMEngine  # noqa: E402
from timesfm.timesfm_2p5.timesfm_2p5_torch import (  # noqa: E402
  TimesFM_2p5_200M_torch_module,
)
from timesfm.torch import util  # noqa: E402

revin = util.revin

METRIC_FIELDS = (
  "mae",
  "rmse",
  "max_abs",
  "rel_l2",
  "normalized_mae",
  "cosine_distance",
)


def make_series(n: int, seed: int = 7) -> np.ndarray:
  """Deterministic multi-periodic signal with additive Gaussian noise."""
  rng = np.random.RandomState(seed)
  t = np.arange(n, dtype=np.float32)
  return (
    np.sin(2 * np.pi * t / 96)
    + 0.5 * np.sin(2 * np.pi * t / 336)
    + 0.2 * rng.randn(n).astype(np.float32)
  ).astype(np.float32)


def tensor_metrics(rolling: torch.Tensor, full: torch.Tensor) -> dict[str, float]:
  """Return scale-aware differences, treating ``full`` as the reference."""
  rolling = rolling.detach().float()
  full = full.detach().float()
  diff = rolling - full
  eps = torch.finfo(torch.float32).eps
  diff_l2 = torch.linalg.vector_norm(diff)
  full_l2 = torch.linalg.vector_norm(full)
  rolling_l2 = torch.linalg.vector_norm(rolling)
  dot = torch.sum(rolling * full)
  return {
    "mae": float(diff.abs().mean().item()),
    "rmse": float(diff.square().mean().sqrt().item()),
    "max_abs": float(diff.abs().max().item()),
    "rel_l2": float((diff_l2 / full_l2.clamp_min(eps)).item()),
    "normalized_mae": float(
      (diff.abs().mean() / full.abs().mean().clamp_min(eps)).item()
    ),
    "cosine_distance": float(
      (1.0 - dot / (rolling_l2 * full_l2).clamp_min(eps)).item()
    ),
  }


def prefix_stats(
  patches: torch.Tensor, masks: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
  """Run the exact serial prefix-statistics loop used by TimesFM.decode."""
  batch_size = patches.shape[0]
  n = torch.zeros(batch_size, device=patches.device)
  mu = torch.zeros(batch_size, device=patches.device)
  sigma = torch.zeros(batch_size, device=patches.device)
  mus, sigmas = [], []
  for i in range(patches.shape[1]):
    (n, mu, sigma), _ = util.update_running_stats(
      n, mu, sigma, patches[:, i], masks[:, i]
    )
    mus.append(mu)
    sigmas.append(sigma)
  return torch.stack(mus, dim=1), torch.stack(sigmas, dim=1)


@torch.no_grad()
def native_full_compute(module, window: torch.Tensor, horizon: int):
  """Run native TimesFM prefill and return its real per-layer DecodeCaches."""
  batch_size = window.shape[0]
  patches = window.view(batch_size, -1, module.p)
  masks = torch.zeros_like(patches, dtype=torch.bool)
  mu, sigma = prefix_stats(patches, masks)
  normed = revin(patches, mu, sigma, reverse=False).to(window.dtype)

  cache_size = patches.shape[1]
  decode_caches = [
    util.DecodeCache(
      next_index=torch.zeros(
        batch_size, dtype=torch.int32, device=window.device
      ),
      num_masked=torch.zeros(
        batch_size, dtype=torch.int32, device=window.device
      ),
      key=torch.zeros(
        batch_size,
        cache_size,
        module.h,
        module.hd,
        dtype=window.dtype,
        device=window.device,
      ),
      value=torch.zeros(
        batch_size,
        cache_size,
        module.h,
        module.hd,
        dtype=window.dtype,
        device=window.device,
      ),
    )
    for _ in range(module.x)
  ]

  (_, _, normed_output, _), decode_caches = module(normed, masks, decode_caches)
  output = revin(normed_output, mu, sigma, reverse=True)
  point_forecast = output.reshape(
    batch_size, -1, module.o, module.q
  )[:, -1, :horizon, module.aridx]
  key = torch.stack([cache.key for cache in decode_caches], dim=0)
  value = torch.stack([cache.value for cache in decode_caches], dim=0)
  return key, value, point_forecast, mu[:, -1], sigma[:, -1]


def chronological_rolling_cache(engine: RollingTimesFMEngine):
  """Read live rolling slots in chronological rather than physical order."""
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


def rebase_post_norm_keys(
  module,
  rolling_key: torch.Tensor,
  rolling_positions: torch.Tensor,
  full_positions: torch.Tensor,
) -> torch.Tensor:
  """Move post-RoPE/post-RMSNorm rolling K into full-compute coordinates.

  TimesFM's key RMSNorm includes a learned per-dimension scale, which does not
  generally commute with RoPE.  We therefore remove that scale, apply the
  relative rotation, and restore the scale for each layer.
  """
  delta = full_positions - rolling_positions
  batch_size = rolling_key.shape[1]
  position = delta[None, :].expand(batch_size, -1)
  rebased = []
  for layer_idx, layer in enumerate(module.stacked_xf):
    key = rolling_key[layer_idx].float()
    scale = layer.attn.key_ln.scale.detach().float()
    if torch.any(scale.abs() < 1e-12):
      raise RuntimeError(
        f"layer {layer_idx} key RMSNorm contains a near-zero scale; "
        "post-norm key rebasing is not invertible"
      )
    unscaled = key / scale.view(1, 1, 1, -1)
    rotated = layer.attn.rotary_position_embedding(unscaled, position)
    rebased.append(rotated * scale.view(1, 1, 1, -1))
  return torch.stack(rebased, dim=0)


@torch.no_grad()
def validate_rebase_formula(module, device: str) -> float:
  """Numerically verify rebasing with one actual TimesFM key-normalization."""
  layer = module.stacked_xf[0]
  generator = torch.Generator(device=device)
  generator.manual_seed(1234)
  raw = torch.randn(
    1, 3, module.h, module.hd, device=device, generator=generator
  )
  full_pos = torch.arange(3, device=device)
  rolling_pos = full_pos + 5
  full = layer.attn.key_ln(
    layer.attn.rotary_position_embedding(raw, full_pos[None, :])
  )
  rolling = layer.attn.key_ln(
    layer.attn.rotary_position_embedding(raw, rolling_pos[None, :])
  )
  rebased = rebase_post_norm_keys(
    _SingleLayerModule(layer), rolling.unsqueeze(0), rolling_pos, full_pos
  )[0]
  return float((rebased - full).abs().max().item())


class _SingleLayerModule:
  """Minimal adapter used only by ``validate_rebase_formula``."""

  def __init__(self, layer):
    self.stacked_xf = [layer]


def compare_cache(
  module,
  rolling_key: torch.Tensor,
  rolling_value: torch.Tensor,
  rolling_positions: torch.Tensor,
  full_key: torch.Tensor,
  full_value: torch.Tensor,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
  """Compute global and per-layer cache metrics for one window."""
  n_patches = full_key.shape[2]
  full_positions = torch.arange(n_patches, device=full_key.device)
  rebased_key = rebase_post_norm_keys(
    module, rolling_key, rolling_positions, full_positions
  )

  scopes: dict[str, slice] = {"all": slice(None)}
  if n_patches > 1:
    scopes.update({"survivors": slice(0, -1), "newest": slice(-1, None)})

  global_metrics: dict[str, Any] = {}
  layer_metrics: list[dict[str, Any]] = [
    {"layer": layer_idx, "scopes": {}} for layer_idx in range(module.x)
  ]
  tensors = {
    "k_direct": (rolling_key, full_key),
    "k_rebased": (rebased_key, full_key),
    "v": (rolling_value, full_value),
  }
  for scope_name, token_slice in scopes.items():
    global_metrics[scope_name] = {}
    for tensor_name, (rolling, full) in tensors.items():
      global_metrics[scope_name][tensor_name] = tensor_metrics(
        rolling[:, :, token_slice], full[:, :, token_slice]
      )
      for layer_idx in range(module.x):
        layer_metrics[layer_idx]["scopes"].setdefault(scope_name, {})[
          tensor_name
        ] = tensor_metrics(
          rolling[layer_idx, :, token_slice], full[layer_idx, :, token_slice]
        )
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
    with open(json_path, "w", encoding="utf-8") as f:
      json.dump(results, f, indent=2)
    print(f"saved JSON -> {json_path}")
  if csv_path:
    csv_path = os.path.abspath(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    rows = flatten_rows(results)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
      writer = csv.DictWriter(
        f, fieldnames=["age", "layer", "scope", "tensor", *METRIC_FIELDS]
      )
      writer.writeheader()
      writer.writerows(rows)
    print(f"saved CSV  -> {csv_path}")


def print_summary(results: dict[str, Any]) -> None:
  print(f"\n{'=' * 100}")
  print("TimesFM-2.5 full-compute vs rolling-cache intermediate K/V differences")
  print(f"{'=' * 100}")
  print(
    f"{'age':>4} {'forecast MAE':>13} {'forecast max':>13} "
    f"{'Kdirect relL2':>15} {'Krebased relL2':>16} {'V relL2':>12}"
  )
  print("-" * 100)
  for step in results["steps"]:
    all_metrics = step["global"]["all"]
    print(
      f"{step['age']:>4} {step['forecast']['mae']:>13.4e} "
      f"{step['forecast']['max_abs']:>13.4e} "
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
  print(f"{'=' * 100}\n")


@torch.no_grad()
def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--ckpt", required=True)
  parser.add_argument(
    "--device", default="cuda" if torch.cuda.is_available() else "cpu"
  )
  parser.add_argument("--context-length", type=int, default=1024)
  parser.add_argument("--horizon", type=int, default=128)
  parser.add_argument("--evict-steps", type=int, default=8)
  parser.add_argument("--seed", type=int, default=7)
  parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
  parser.add_argument("--output-json")
  parser.add_argument("--output-csv")
  args = parser.parse_args()

  if args.context_length < 32 or args.context_length % 32:
    raise ValueError("--context-length must be >=32 and divisible by 32")
  if args.horizon < 1 or args.horizon > 128:
    raise ValueError("this diagnostic requires 1 <= --horizon <= 128")
  if args.evict_steps < 1:
    raise ValueError("--evict-steps must be >= 1")
  if args.device == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

  dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
  torch.manual_seed(0)
  module = TimesFM_2p5_200M_torch_module()
  module.device = torch.device(args.device)
  module.load_checkpoint(args.ckpt)
  module.eval()
  if dtype != torch.float32:
    module.to(dtype)

  rebase_check = validate_rebase_formula(module, args.device)
  tolerance = 2e-5 if dtype == torch.float32 else 2e-2
  if rebase_check > tolerance:
    raise RuntimeError(
      f"key rebase self-check failed: max_abs={rebase_check:.3e}, "
      f"tolerance={tolerance:.3e}"
    )

  length = args.context_length + args.evict_steps * module.p
  series = make_series(length, seed=args.seed)
  window = torch.as_tensor(
    series[: args.context_length], device=args.device, dtype=dtype
  )[None, :]

  config = RollingConfig(
    context_length=args.context_length,
    horizon=args.horizon,
    full_refresh_every=0,
    batch_size=1,
    device=args.device,
    dtype=dtype,
  )
  engine = RollingTimesFMEngine(module, config)
  engine.full_refresh(window)

  device_name = args.device
  if args.device == "cuda":
    device_name = torch.cuda.get_device_name(0)
  results: dict[str, Any] = {
    "model": "TimesFM-2.5-200M",
    "checkpoint": os.path.abspath(args.ckpt),
    "config": {
      "context_length": args.context_length,
      "num_patches": args.context_length // module.p,
      "patch_length": module.p,
      "horizon": args.horizon,
      "evict_steps": args.evict_steps,
      "seed": args.seed,
      "dtype": args.dtype,
      "device": args.device,
    },
    "environment": {
      "python": platform.python_version(),
      "torch": torch.__version__,
      "device_name": device_name,
    },
    "protocol": {
      "full_compute": "native TimesFM forward with one DecodeCache per layer",
      "rolling": "ring-buffer cache with frozen-at-encode prefix statistics",
      "alignment": (
        "live rolling slots sorted by absolute position and matched chronologically"
      ),
      "k_direct": "raw post-RoPE/post-key-RMSNorm cached keys",
      "k_rebased": (
        "rolling keys analytically rotated into full-compute relative RoPE positions; "
        "learned key-RMSNorm scale is removed before rotation and restored after"
      ),
      "v": "raw cached values; no position transform is needed",
      "scopes": {
        "all": "all N tokens in the current cache",
        "survivors": "the first N-1 cached tokens reused by rolling",
        "newest": "the newly encoded final token",
      },
    },
    "metric_definitions": {
      "mae": "mean(abs(rolling - full))",
      "rmse": "sqrt(mean((rolling - full)^2))",
      "max_abs": "max(abs(rolling - full))",
      "rel_l2": "||rolling - full||_2 / ||full||_2",
      "normalized_mae": "MAE / mean(abs(full))",
      "cosine_distance": "1 - cosine_similarity(rolling, full)",
    },
    "rebase_self_check_max_abs": rebase_check,
    "steps": [],
  }

  for age in range(args.evict_steps + 1):
    if age > 0:
      lo = args.context_length + (age - 1) * module.p
      new_patch = torch.as_tensor(
        series[lo : lo + module.p], device=args.device, dtype=dtype
      )[None, :]
      engine.fast_update(new_patch)
      window = torch.cat([window[:, module.p :], new_patch], dim=1)

    full_key, full_value, full_forecast, full_mu, full_sigma = native_full_compute(
      module, window, args.horizon
    )
    rolling_key, rolling_value, positions = chronological_rolling_cache(engine)
    global_metrics, layer_metrics = compare_cache(
      module,
      rolling_key,
      rolling_value,
      positions,
      full_key,
      full_value,
    )
    rolling_forecast = engine.forecast()
    results["steps"].append(
      {
        "age": age,
        "rolling_position_first": int(positions[0].item()),
        "rolling_position_last": int(positions[-1].item()),
        "forecast": tensor_metrics(rolling_forecast, full_forecast),
        "normalization": {
          "rolling_mu": float(engine.last_mu[0].item()),
          "full_mu": float(full_mu[0].item()),
          "mu_abs_diff": float((engine.last_mu - full_mu).abs()[0].item()),
          "rolling_sigma": float(engine.last_sigma[0].item()),
          "full_sigma": float(full_sigma[0].item()),
          "sigma_abs_diff": float(
            (engine.last_sigma - full_sigma).abs()[0].item()
          ),
        },
        "global": global_metrics,
        "layers": layer_metrics,
      }
    )

  print_summary(results)
  save_outputs(results, args.output_json, args.output_csv)


if __name__ == "__main__":
  main()
