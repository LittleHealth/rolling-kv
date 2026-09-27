"""Supplementary breakdowns of the TimesFM-3.0 mechanism campaign for the report.

Reads the raw shards written by mech_campaign.py and writes breakdown.json:
  * accuracy against the ground truth: pooled MAE change of each rung vs F (%),
    with 57-day block-bootstrap CIs
  * distribution of per-window gap_pct (p10/p50/p90/p99) per k
  * gap as a function of lead time h (pooled per h)
  * per-channel gap_pct, MAE_F/std, signed shares (STALE, STAT-kv) and segment sizes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mech_aggregate import QM, boot_idx, load, ratio


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--root", required=True)
  ap.add_argument("--block", type=int, default=57)
  ap.add_argument("--n_boot", type=int, default=2000)
  args = ap.parse_args()
  d, metas = load(args.root)
  ks = d["ks"].tolist()
  channels = metas[0]["channels"]
  nT, nK, B = d["lad_roll"].shape[:3]
  idx = boot_idx(nT, args.n_boot, block=args.block)
  s = d["win_std"].astype(np.float64)
  F = d["Fdec"].astype(np.float64)
  Fq = F[..., QM]
  Y = d["Y"].astype(np.float64)
  out = {"ks": ks, "channels": channels, "n_targets": int(nT)}

  # ---- accuracy vs ground truth -------------------------------------------
  maeF = np.abs(Fq - Y).mean(-1) / s  # [T,B]
  maeF_t = np.repeat(maeF.sum(1)[:, None], nK, 1)
  acc = {}
  for r in ("roll", "Rabs", "R0", "Fp"):
    Xq = d["lad_" + r][..., QM].astype(np.float64)
    mae = (np.abs(Xq - Y[:, None]).mean(-1) / s[:, None]).sum(2)  # [T,K]
    est, lo, hi = ratio(100 * (mae - maeF_t), maeF_t, idx)
    acc[r] = {"est": est.round(3).tolist(), "lo": lo.round(3).tolist(), "hi": hi.round(3).tolist()}
  out["mae_change_vs_F_pct"] = acc
  # per-window: fraction of windows where rolling is more accurate than F
  Xq = d["lad_roll"][..., QM].astype(np.float64)
  mae_roll_w = np.abs(Xq - Y[:, None]).mean(-1)  # [T,K,B]
  mae_F_w = np.abs(Fq - Y).mean(-1)[:, None]
  out["frac_windows_roll_better"] = (mae_roll_w < mae_F_w).mean((0, 2)).round(3).tolist()
  out["mae_F_over_std_pooled"] = float(maeF.mean())

  # ---- distribution of per-window gap_pct --------------------------------
  gw = 100 * np.abs(Xq - Fq[:, None]).mean(-1) / np.maximum(np.abs(Fq - Y).mean(-1)[:, None], 1e-12)
  out["gap_window_pct_quantiles"] = {q: np.percentile(gw, int(q[1:]), axis=(0, 2)).round(3).tolist()
                                     for q in ("p10", "p50", "p90", "p99")}
  # scale-free version: per-window gap / std (robust to tiny MAE_F)
  gs = np.abs(Xq - Fq[:, None]).mean(-1) / s[:, None]
  out["gap_over_std_quantiles"] = {q: np.percentile(gs, int(q[1:]), axis=(0, 2)).round(5).tolist()
                                   for q in ("p10", "p50", "p90", "p99")}

  # ---- gap vs lead time ---------------------------------------------------
  num_h = (np.abs(Xq - Fq[:, None]) / s[:, None, :, None]).sum((0, 2))  # [K,H]
  den = maeF.sum()
  lead = 100 * num_h / den
  out["gap_by_lead_pct"] = {"h": [1, 8, 16, 32, 48, 64],
                            "values": lead[:, [0, 7, 15, 31, 47, 63]].round(3).tolist()}

  # ---- per channel --------------------------------------------------------
  def vec(name):
    return d["lad_" + name].astype(np.float64) / s[:, None, :, None, None]

  Fv = (F / s[:, :, None, None])[:, None]
  Dv = vec("roll") - Fv
  st = vec("roll") - vec("R1")
  ab = vec("Rabs")
  sv = vec("R0new") - vec("Fp")
  per = []
  for c in range(B):
    DD = (Dv[:, :, c] ** 2).sum((0, 2, 3))
    g_roll = np.abs(Xq[:, :, c] - Fq[:, None, c]).mean(-1) / s[:, None, c]
    g_abs = np.abs(d["lad_Rabs"][:, :, c, :, QM] - Fq[:, None, c]).mean(-1) / s[:, None, c]
    mf = maeF[:, c].sum()
    per.append({
      "channel": channels[c],
      "maeF_over_std": round(float(maeF[:, c].mean()), 4),
      "gap_roll_pct": (100 * g_roll.sum(0) / mf).round(2).tolist(),
      "gap_Rabs_pct": (100 * g_abs.sum(0) / mf).round(2).tolist(),
      "share_STALE": ((st[:, :, c] * Dv[:, :, c]).sum((0, 2, 3)) / DD).round(3).tolist(),
      "share_STATkv": ((sv[:, :, c] * Dv[:, :, c]).sum((0, 2, 3)) / DD).round(3).tolist(),
      "size_STALE": np.sqrt((st[:, :, c] ** 2).sum((0, 2, 3)) / DD).round(3).tolist(),
      "size_STATkv": np.sqrt((sv[:, :, c] ** 2).sum((0, 2, 3)) / DD).round(3).tolist(),
    })
  out["per_channel"] = per
  del ab
  # ---- direction of the rolling deviation relative to F's error ----------
  # delta = yhat_roll - yhat_F, e = y - yhat_F. cos > 0: rolling leans toward the truth.
  def cos_ci(a, b):  # a, b: [T,K,B,H] already /s
    ab = (a * b).sum((2, 3)); aa = (a ** 2).sum((2, 3)); bb = (b ** 2).sum((2, 3))
    est = ab.sum(0) / np.sqrt(aa.sum(0) * bb.sum(0))
    bs = np.stack([ab[i].sum(0) / np.sqrt(aa[i].sum(0) * bb[i].sum(0)) for i in idx])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    return {"est": est.round(4).tolist(), "lo": lo.round(4).tolist(), "hi": hi.round(4).tolist()}

  e_v = np.broadcast_to(((Y - Fq) / s[:, :, None])[:, None], Xq.shape)
  direction = {}
  mirror = {}
  for r in ("roll", "Rabs"):
    Rq = d["lad_" + r][..., QM].astype(np.float64)
    dlt = (Rq - Fq[:, None]) / s[:, None, :, None]
    direction[r] = cos_ci(dlt, e_v)
    # mirror control: forecast F - delta; if delta leans toward the truth, the mirror is worse
    mae_m = (np.abs(Fq[:, None] - (Rq - Fq[:, None]) - Y[:, None]).mean(-1) / s[:, None]).sum(2)
    est, lo, hi = ratio(100 * (mae_m - maeF_t), maeF_t, idx)
    mirror[r] = {"est": est.round(3).tolist(), "lo": lo.round(3).tolist(), "hi": hi.round(3).tolist()}
  out["cos_delta_vs_error"] = direction
  out["mae_change_mirror_pct"] = mirror
  # paired asymmetry: MAE(F - delta) - MAE(F + delta). Zero in expectation if delta is
  # independent of F's error (symmetric in delta); > 0 if delta leans toward the truth.
  asym = {}
  for r in ("roll", "Rabs"):
    Rq = d["lad_" + r][..., QM].astype(np.float64)
    m_plus = (np.abs(Rq - Y[:, None]).mean(-1) / s[:, None]).sum(2)
    m_minus = (np.abs(2 * Fq[:, None] - Rq - Y[:, None]).mean(-1) / s[:, None]).sum(2)
    est, lo, hi = ratio(100 * (m_minus - m_plus), maeF_t, idx)
    asym[r] = {"est": est.round(3).tolist(), "lo": lo.round(3).tolist(), "hi": hi.round(3).tolist()}
  out["mirror_minus_actual_mae_pct"] = asym

  # ---- detrend gate on every refresh window and every N+k window ------------
  from mech_campaign import L_CTX, load_weather, target_times
  import os
  raw, _, _ = load_weather(Path(os.environ.get("ROLLKV_DATASETS", "datasets")) / "weather" / "weather.csv")
  p = 32
  worst, n_win, n_on = np.inf, 0, 0
  for T in d["targets"].tolist():
    for k in ks:
      for a, b in ((T - L_CTX - k * p, T), (T - L_CTX - k * p, T - k * p)):  # N+k window, refresh window
        x = raw[:, a:b].astype(np.float64)
        Ln = x.shape[1]
        t = np.arange(-(Ln - 1), 1) / Ln
        A = np.vstack([t, np.ones_like(t)]).T
        coef = np.linalg.lstsq(A, x.T, rcond=None)[0]
        det = x - (coef[0][:, None] * t + coef[1][:, None])
        ratio_ = det.std(1) / x.std(1)
        worst = min(worst, float(ratio_.min()))
        n_win += ratio_.size
        n_on += int((ratio_ < 0.5).sum())
  out["gate_check_NplusK_and_refresh"] = {"windows": n_win, "gate_on": n_on, "min_std_ratio": round(worst, 4),
                                           "threshold": 0.5}

  # ---- probabilistic accuracy: mean pinball loss over the 9 quantiles ------
  qs = np.linspace(0.1, 0.9, 9)

  def pinball(Xf):  # Xf [T,K,B,H,9] or [T,B,H,9]
    err = Y[..., None] - Xf if Xf.ndim == 4 else Y[:, None, ..., None] - Xf
    return np.maximum(qs * err, (qs - 1) * err).mean((-1, -2))

  pl_F = pinball(F) / s  # [T,B]
  pl_F_t = np.repeat(pl_F.sum(1)[:, None], nK, 1)
  wql = {}
  for r in ("roll", "Rabs", "R0"):
    pl = (pinball(d["lad_" + r].astype(np.float64)) / s[:, None]).sum(2)
    est, lo, hi = ratio(100 * (pl - pl_F_t), pl_F_t, idx)
    wql[r] = {"est": est.round(3).tolist(), "lo": lo.round(3).tolist(), "hi": hi.round(3).tolist()}
  out["pinball_change_vs_F_pct"] = wql

  # ---- accuracy of the Exp2 arms (median forecasts) ------------------------
  arm_acc = {}
  for a in ("P1all", "P2", "P3", "P4", "A", "B"):
    Aq = d["armp_" + a].astype(np.float64)
    mae = (np.abs(Aq - Y[:, None]).mean(-1) / s[:, None]).sum(2)
    est, lo, hi = ratio(100 * (mae - maeF_t), maeF_t, idx)
    arm_acc[a] = {"est": est.round(3).tolist(), "lo": lo.round(3).tolist(), "hi": hi.round(3).tolist()}
  out["arm_mae_change_vs_F_pct"] = arm_acc

  # ---- gap per lead relative to the per-lead MAE of F ---------------------
  maeF_h = (np.abs(Fq - Y) / s[:, :, None]).sum((0, 1))  # [H]
  rel_h = 100 * num_h / maeF_h[None, :]
  out["gap_by_lead_rel_pct"] = {"h": [1, 8, 16, 32, 48, 64],
                                "values": rel_h[:, [0, 7, 15, 31, 47, 63]].round(3).tolist(),
                                "maeF_h_rel_to_mean": (maeF_h / maeF_h.mean())[[0, 7, 15, 31, 47, 63]].round(3).tolist()}

  # ---- sigma check of the P3 control (gate off everywhere) -----------------
  st_ = d["stats"].astype(np.float64)
  off = (~d["gates"][:, :, 0]) & (~d["gates"][:, :, 1])
  out["p3_sigma_vs_F_gateoff_maxrel"] = float((np.abs(st_[:, :, 5] - st_[:, :, 3]) / st_[:, :, 3])[off].max())
  out["p3_mu_vs_F_gateoff_maxrel"] = float((np.abs(st_[:, :, 4] - st_[:, :, 2]) / st_[:, :, 3])[off].max())

  Path(args.root, "breakdown.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
  print(json.dumps({k: v for k, v in out.items() if k != "per_channel"}, indent=1))


if __name__ == "__main__":
  main()
