"""Aggregate mech_campaign.py shards into the numbers the three figures plot.

Pooling rule (all panels): every window's forecast errors are divided by the
standard deviation of its own context window (raw values) before pooling, and
ratios are ratio-of-sums over all (target, channel) windows.  95% confidence
intervals use a moving-block bootstrap over target times (block = 10 targets,
i.e. 10 days), resampling all channels of a target together.

Writes summary.npz (arrays) and summary.json (headline numbers + controls).
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

QM = 4  # median quantile index (q = 0.5)
H = 64


def load(root):
  files = sorted(glob.glob(str(Path(root) / "shard*.npz")))
  metas = [json.loads(Path(f[:-4] + ".json").read_text()) for f in files]
  parts = [dict(np.load(f)) for f in files]
  keys = parts[0].keys()
  d = {}
  for k in keys:
    if k == "ks":
      d[k] = parts[0][k]
      continue
    d[k] = np.concatenate([p[k] for p in parts], axis=0)
  order = np.argsort(d["targets"])
  for k in keys:
    if k != "ks":
      d[k] = d[k][order]
  return d, metas


def boot_idx(nT, n_boot=2000, block=10, seed=0):
  rng = np.random.default_rng(seed)
  block = max(1, min(block, nT))
  n_blocks = int(np.ceil(nT / block))
  starts = rng.integers(0, nT - block + 1, size=(n_boot, n_blocks))
  idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :nT]
  return idx


def ratio(num_t, den_t, idx):
  """num_t, den_t: [nT, ...] per-target sums. Returns est, lo, hi."""
  est = num_t.sum(0) / den_t.sum(0)
  bs = np.stack([num_t[i].sum(0) / den_t[i].sum(0) for i in idx])
  lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
  return est, lo, hi


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--root", required=True)
  ap.add_argument("--n_boot", type=int, default=2000)
  ap.add_argument("--block", type=int, default=57,
                  help="bootstrap block in targets (days); 57 ~ one 8192-point context")
  args = ap.parse_args()
  d, metas = load(args.root)
  meta = metas[0]
  ks = d["ks"].tolist()
  nT, nK = d["lad_roll"].shape[:2]
  B = d["lad_roll"].shape[2]
  idx = boot_idx(nT, args.n_boot, block=args.block)
  idx10 = boot_idx(nT, args.n_boot, block=10, seed=1)
  s = d["win_std"].astype(np.float64)  # [T,B]
  S = {}
  J = {"n_targets": int(nT), "n_channels": int(B), "ks": ks, "channels": meta["channels"],
       "n_windows_per_k": int(nT * B), "pooling": "per-window /std(context), ratio of sums",
       "ci": "moving-block bootstrap over targets, block %d, n=%d" % (args.block, args.n_boot)}

  F = d["Fdec"].astype(np.float64)  # [T,B,H,nq]
  Fq = F[..., QM]
  Y = d["Y"].astype(np.float64)
  maeF = (np.abs(Fq - Y).mean(-1) / s)  # [T,B]
  maeF_t = maeF.sum(1)  # [T]

  def G(Xq):  # [T,K,B,H] -> per-window gap / std
    return np.abs(Xq - Fq[:, None]).mean(-1) / s[:, None]

  # ---------------- Exp3 (a): gap_pct per rung -------------------------------
  rungs = ["roll", "R1", "Rabs", "R0", "R0out", "R0new", "Fabs", "Fp"]
  gap = {}
  for r in rungs:
    g_t = G(d["lad_" + r][..., QM].astype(np.float64)).sum(2)  # [T,K]
    est, lo, hi = ratio(100 * g_t, np.repeat(maeF_t[:, None], nK, 1), idx)
    gap[r] = (est, lo, hi)
    S[f"gap_{r}"] = np.stack([est, lo, hi])
  J["gap_pct"] = {r: [float(x) for x in gap[r][0]] for r in rungs}
  J["mae_F_over_std"] = float(maeF.mean())

  # ---------------- Exp3 (b,c): signed shares --------------------------------
  def vec(name):
    return d["lad_" + name].astype(np.float64) / s[:, None, :, None, None]

  Fv = (F / s[:, :, None, None])[:, None]
  Dv = vec("roll") - Fv
  DD_t = (Dv ** 2).sum((2, 3, 4))  # [T,K]

  def share_chain(chain, names, tag):
    X = [vec(c) if c != "F" else np.broadcast_to(Fv, Dv.shape) for c in chain]
    out = {}
    for i, n in enumerate(names):
      di = X[i] - X[i + 1]
      num_t = (di * Dv).sum((2, 3, 4))
      est, lo, hi = ratio(num_t, DD_t, idx)
      nrm = np.sqrt(np.stack(ratio((di ** 2).sum((2, 3, 4)), DD_t, idx)))
      out[n] = (est, lo, hi)
      S[f"share{tag}_{n}"] = np.stack([est, lo, hi])
      S[f"share{tag}_{n}_b10"] = np.stack(ratio(num_t, DD_t, idx10))
      S[f"norm{tag}_{n}"] = nrm[0]
      S[f"normci{tag}_{n}"] = nrm
    return out

  namesA = ["STALE", "NUM", "POS", "STATout", "STATnew", "STATsurv", "NUMdec"]
  shA = share_chain(["roll", "R1", "Rabs", "R0", "R0out", "R0new", "Fp", "F"], namesA, "A")
  # pooled pairwise cosines between the main path-A segments (per k)
  chainA = ["roll", "R1", "Rabs", "R0", "R0out", "R0new", "Fp"]
  segs = {n: vec(chainA[i]) - vec(chainA[i + 1]) for i, n in enumerate(namesA[:-1])}
  cn = ["STALE", "POS", "STATout", "STATnew", "STATsurv"]
  cosm = np.zeros((nK, len(cn), len(cn)))
  for i, a_ in enumerate(cn):
    for j, b_ in enumerate(cn):
      num = (segs[a_] * segs[b_]).sum((0, 2, 3, 4))
      den = np.sqrt((segs[a_] ** 2).sum((0, 2, 3, 4)) * (segs[b_] ** 2).sum((0, 2, 3, 4)))
      cosm[:, i, j] = num / np.maximum(den, 1e-300)
  S["seg_cos"] = cosm

  def cos_ci(a, b):
    ab = (a * b).sum((2, 3, 4)); aa = (a ** 2).sum((2, 3, 4)); bb = (b ** 2).sum((2, 3, 4))  # [T,K]
    est = ab.sum(0) / np.sqrt(aa.sum(0) * bb.sum(0))
    bs = np.stack([ab[i].sum(0) / np.sqrt(aa[i].sum(0) * bb[i].sum(0)) for i in idx])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    return np.stack([est, lo, hi])

  stat_all = segs["STATout"] + segs["STATnew"] + segs["STATsurv"]
  S["cosci_STALE_STATsurv"] = cos_ci(segs["STALE"], segs["STATsurv"])
  S["cosci_STATout_STATnew"] = cos_ci(segs["STATout"], segs["STATnew"])
  S["cosci_STALE_STATall"] = cos_ci(segs["STALE"], stat_all)
  pair = segs["STALE"] + segs["STATsurv"]
  S["norm_STALEplusSTATsurv"] = np.sqrt(np.stack(ratio((pair ** 2).sum((2, 3, 4)), DD_t, idx)))
  pair2 = segs["STATout"] + segs["STATnew"]
  S["norm_STAToutplusnew"] = np.sqrt(np.stack(ratio((pair2 ** 2).sum((2, 3, 4)), DD_t, idx)))
  S["norm_STATall"] = np.sqrt(np.stack(ratio((stat_all ** 2).sum((2, 3, 4)), DD_t, idx)))
  del pair, pair2, stat_all
  J["cos_ci"] = {k_: S["cosci_" + k_].round(3).tolist() for k_ in ("STALE_STATsurv", "STATout_STATnew", "STALE_STATall")}
  J["norm_pairs"] = {k_: S[k_].round(3).tolist() for k_ in ("norm_STALEplusSTATsurv", "norm_STAToutplusnew", "norm_STATall")}
  J["seg_cos_names"] = cn
  J["seg_cos_STALE_STATsurv"] = [float(x) for x in cosm[:, 0, 4]]
  J["seg_cos_STATout_STATnew"] = [float(x) for x in cosm[:, 2, 3]]
  J["seg_cos_with_D"] = {n: [float(x) for x in ((segs[n] * Dv).sum((0, 2, 3, 4)) / np.sqrt((segs[n] ** 2).sum((0, 2, 3, 4)) * DD_t.sum(0)))] for n in cn}
  del segs
  namesB = ["STALE", "NUM", "STATTREND", "POS", "NUMdec"]
  shB = share_chain(["roll", "R1", "Rabs", "Fabs", "Fp", "F"], namesB, "B")
  J["sharesA"] = {n: [float(x) for x in shA[n][0]] for n in namesA}
  J["sharesB"] = {n: [float(x) for x in shB[n][0]] for n in namesB}
  # Shapley / interaction in the STALE-free plane
  statA_t = sum(((vec(x) - vec(y)) * Dv).sum((2, 3, 4)) for x, y in (("R0", "R0out"), ("R0out", "R0new"), ("R0new", "Fp")))
  posA_t = ((vec("Rabs") - vec("R0")) * Dv).sum((2, 3, 4))
  statB_t = ((vec("Rabs") - vec("Fabs")) * Dv).sum((2, 3, 4))
  posB_t = ((vec("Fabs") - vec("Fp")) * Dv).sum((2, 3, 4))
  # fix-one = remove the source while the other one is present; break-one = the
  # source alone (design doc 3.5): POS fix R,abs->R,0, break F,abs->F';
  # STAT fix R,abs->F,abs, break R,0->F'.  Path A = (POS fix, STAT break),
  # path B = (STAT fix, POS break).
  for name, num in (("POS_fix", posA_t), ("POS_break", posB_t), ("POS_shapley", 0.5 * (posA_t + posB_t)),
                    ("ST_fix", statB_t), ("ST_break", statA_t), ("ST_shapley", 0.5 * (statA_t + statB_t)),
                    ("POS_pathA", posA_t), ("POS_pathB", posB_t), ("ST_pathA", statA_t), ("ST_pathB", statB_t),
                    ("interaction", posA_t - posB_t)):
    S["shap_" + name] = np.stack(ratio(num, DD_t, idx))
  J["shapley"] = {n: [float(x) for x in S["shap_" + n][0]] for n in
                  ("POS_fix", "POS_break", "POS_shapley", "ST_fix", "ST_break", "ST_shapley", "interaction")}
  J["shapley_ci"] = {n: S["shap_" + n].round(4).tolist() for n in
                     ("POS_fix", "POS_break", "POS_shapley", "ST_fix", "ST_break", "ST_shapley", "interaction")}

  # ---------------- Exp2 (a): fix-one arms, dG/G -----------------------------
  g_roll = G(d["lad_roll"][..., QM].astype(np.float64))  # [T,K,B]
  g_roll_t = g_roll.sum(2)
  J["armG"] = {}
  J["arm_worse_frac"] = {}
  for a in ("identity", "P1all", "P2", "P3", "P4", "P5", "A", "B"):
    g_a = G(d["armp_" + a].astype(np.float64))
    est, lo, hi = ratio(100 * (g_roll - g_a).sum(2), g_roll_t, idx)
    S["armG_" + a] = np.stack([est, lo, hi])
    S["armG_" + a + "_b10"] = np.stack(ratio(100 * (g_roll - g_a).sum(2), g_roll_t, idx10))
    J["armG"][a] = [float(x) for x in est]
    J["arm_worse_frac"][a] = [float(x) for x in (g_a > g_roll * (1 + 1e-9)).mean((0, 2))]

  # ---------------- Exp2 (b): per-layer, relative to the staleness gap -------
  Rq = d["lad_Rabs"][..., QM].astype(np.float64)

  def Gs(Xq):  # gap to R,abs (staleness-only reference)
    return np.abs(Xq - Rq).mean(-1) / s[:, None]

  s_roll = Gs(d["lad_roll"][..., QM].astype(np.float64))  # [T,K,B]
  s_roll_t = s_roll.sum(2)
  P1l = d["armp_P1l"].astype(np.float64)  # [T,K,L,B,H]
  ABl = d["armp_ABl"].astype(np.float64)
  CUT = d["armp_CUT"].astype(np.float64)
  nL = P1l.shape[2]
  fix_l, brk_l, cut_l, fixF_l = [], [], [], []
  for l in range(nL):
    fix_l.append(ratio(100 * (s_roll - Gs(P1l[:, :, l])).sum(2), s_roll_t, idx))
    brk_l.append(ratio(100 * Gs(ABl[:, :, l]).sum(2), s_roll_t, idx))
    fixF_l.append(ratio(100 * (g_roll - G(P1l[:, :, l])).sum(2), g_roll_t, idx))
  for l in range(nL + 1):
    cut_l.append(ratio(100 * Gs(CUT[:, :, l]).sum(2), s_roll_t, idx))
  S["layer_fix"] = np.array(fix_l).transpose(1, 0, 2)  # [3, L, K]
  S["layer_break"] = np.array(brk_l).transpose(1, 0, 2)
  S["layer_fixF"] = np.array(fixF_l).transpose(1, 0, 2)
  S["cut"] = np.array(cut_l).transpose(1, 0, 2)  # [3, L+1, K]
  J["staleness_gap_over_G"] = [float(x) for x in (s_roll_t.sum(0) / g_roll_t.sum(0))]
  J["P1all_vs_Rabs_rel"] = float(np.abs(d["armp_P1all"] - d["lad_Rabs"][..., QM]).max() / np.abs(d["lad_Rabs"]).max())

  # ---------------- Exp2 (d): controls ---------------------------------------
  ysc = float(d["ctrl_yscale_maxabs"].max())
  Dq = d["lad_roll"][..., QM].astype(np.float64) - Fq[:, None]
  DDq = ((Dq / s[:, None, :, None]) ** 2).sum()

  def rms_rel(Xq, Rq_):
    return float(np.sqrt((((Xq - Rq_) / s[:, None, :, None]) ** 2).sum() / DDq))

  lay0 = d["armp_P1l"][:, :, 0].astype(np.float64)
  chain = ["roll", "R1", "Rabs", "R0", "R0out", "R0new", "Fp"]
  X32 = [d["lad_" + c].astype(np.float32) for c in chain] + [np.broadcast_to(d["Fdec"][:, None], d["lad_roll"].shape).astype(np.float32)]
  resid = sum((X32[i] - X32[i + 1]) for i in range(len(X32) - 1)) - (X32[0] - X32[-1])
  st = d["stats"].astype(np.float64)
  gates = d["gates"]
  off = (~gates[:, :, 0]) & (~gates[:, :, 1])
  ctrl = {
    "k0_roll_vs_full_maxrel": float(d["ctrl_k0_roll_vs_Fp_maxabs"].max() / ysc),
    "k0_Rabs_vs_full_maxrel": float(d["ctrl_k0_Rabs_vs_Fp_maxabs"].max() / ysc),
    "identity_swap_maxrel": float(d["ctrl_identity_vs_roll_maxabs"].max() / ysc),
    "cut20_vs_Rabs_maxrel": float(d["ctrl_cut_last_vs_Rabs_maxabs"].max() / ysc),
    "readout_vs_forecast_maxrel": float(d["ctrl_readout_vs_forecast_maxabs"].max() / ysc),
    "roll_readout_vs_forecast_maxrel": float(d["ctrl_roll_readout_vs_forecast_maxabs"].max() / ysc),
    "R1_vs_Rabs_maxrel": float(np.abs(d["lad_R1"] - d["lad_Rabs"]).max() / ysc),
    "Fp_vs_decode_maxrel": float(np.abs(d["lad_Fp"][:, 0] - d["Fdec"]).max() / ysc),
    "layer0_swap_rms_over_D": rms_rel(lay0, d["lad_roll"][..., QM].astype(np.float64)),
    "allswap_vs_Rabs_rms_over_D": rms_rel(d["armp_P1all"].astype(np.float64), d["lad_Rabs"][..., QM].astype(np.float64)),
    "num_floor_rms_over_D": rms_rel(d["lad_R1"][..., QM].astype(np.float64), d["lad_Rabs"][..., QM].astype(np.float64)),
    "num_dec_rms_over_D": rms_rel(d["lad_Fp"][..., QM].astype(np.float64), np.broadcast_to(Fq[:, None], d["lad_Fp"].shape[:-1])),
    "ladder_resid_maxrel": float(np.abs(resid).max() / ysc),
    "p2_rot_roundtrip_max": float(d["ctrl_p2_rot_roundtrip_rel"].max()),
    "p2_layer0_vs_R0_max": float(d["ctrl_p2_layer0_vs_R0_rel"].max()),
    "p3_stats_vs_F_gateoff_max": float((np.abs(st[:, :, 4] - st[:, :, 2]) / st[:, :, 3])[off].max()) if off.any() else None,
    "stale_k1_rms_over_D": None,
  }
  k1 = ks.index(1) if 1 in ks else 0
  Dk = Dq[:, k1]
  ctrl["stale_k1_rms_over_D"] = float(np.sqrt((((d["lad_roll"][:, k1, ..., QM] - d["lad_R1"][:, k1, ..., QM]) / s[:, :, None]) ** 2).sum()
                                            / ((Dk / s[:, :, None]) ** 2).sum()))
  J["controls"] = ctrl
  J["gate_apply_frac_roll"] = [float(x) for x in gates[:, :, 0].mean((0, 2))]
  J["gate_apply_frac_full"] = float(gates[:, 0, 1].mean())
  J["gate_flip_frac"] = [float(x) for x in (gates[:, :, 0] != gates[:, :, 1]).mean((0, 2))]
  J["key_ln_min_abs"] = meta["key_ln_min_abs"]
  J["_layer_stations"] = list(meta["layer_stations"])
  J["_layer_extra"] = list(meta["layer_extra"])
  J["n_bad_interpolated"] = meta["n_bad_interpolated"]
  J["seconds_per_shard"] = [m["seconds"] for m in metas]

  # ---------------- Exp1 profiles ---------------------------------------------
  LS = meta["layer_stations"]
  LX = meta["layer_extra"]
  IO = meta["io_stations"]
  for pr in meta["pairs"]:
    num = d[f"prof_{pr}_lay_num"].astype(np.float64)
    den = d[f"prof_{pr}_lay_den"].astype(np.float64)
    rel = np.sqrt(num / np.maximum(den, 1e-300))  # [T,K,L,S,B]
    io = np.sqrt(d[f"prof_{pr}_io_num"] / np.maximum(d[f"prof_{pr}_io_den"], 1e-300))  # [T,K,IO,B]
    # pipeline stations: detrend, revin, token, h_0..h_20, z, den, yhat
    h0 = rel[:, :, 0, LS.index("x_in")][:, :, None]
    hl = rel[:, :, :, LS.index("h_out")]
    pipe = np.concatenate([io[:, :, :3], h0, hl, io[:, :, 3:]], axis=2)  # [T,K,27,B]
    S[f"pipe_{pr}"] = np.stack([np.percentile(pipe, q, axis=(0, 3)) for q in (25, 50, 75)])  # [3,K,27]
    S[f"pipe_{pr}_zerofrac"] = (pipe == 0).mean(axis=(0, 3))
    if f"prof_{pr}_lay_dot" in d:
      dot = d[f"prof_{pr}_lay_dot"]
      na = d[f"prof_{pr}_lay_na"]
      oc = 1.0 - dot / np.sqrt(np.maximum(na * den, 1e-300))  # [T,K,L,S,B]
      hc = np.concatenate([oc[:, :, 0, LS.index("x_in")][:, :, None], oc[:, :, :, LS.index("h_out")]], axis=2)
      S[f"cos_{pr}"] = np.stack([np.percentile(hc, q, axis=(0, 3)) for q in (25, 50, 75)])  # [3,K,21]
    S[f"lay_{pr}"] = np.stack([np.percentile(rel, q, axis=(0, 4)) for q in (25, 50, 75)])  # [3,K,L,S]
    ex = d[f"prof_{pr}_extra"].astype(np.float64)
    S[f"ext_{pr}"] = np.stack([np.nanpercentile(ex, q, axis=(0, 4)) for q in (25, 50, 75)])  # [3,K,L,E]
    S[f"ext_{pr}_mean"] = np.nanmean(ex, axis=(0, 4))
    cr = d[f"prof_{pr}_cross"].astype(np.float64)  # [T,K,L,3,2,B]
    S[f"cross_{pr}"] = cr[..., 0, :].sum((0, 4)) / np.maximum(cr[..., 1, :].sum((0, 4)), 1e-300)  # [K,L,3]
    # bounded version: 2<dh,du> / (|dh|^2 + |du|^2) in [-1, 1]; |dh|^2+|du|^2 = |dh'|^2 - 2<dh,du>
    S[f"crossb_{pr}"] = cr[..., 0, :].sum((0, 4)) / np.maximum((cr[..., 1, :] - cr[..., 0, :]).sum((0, 4)), 1e-300)
    S[f"lay_{pr}_zerofrac"] = (rel == 0).mean(axis=(0, 4))

  # output-end 4-term split vs F' (64x9, per-window /std)
  o4 = d["out4"].astype(np.float64) / s[:, None, None, :, None, None]  # [T,K,4,B,H,nq]
  Dp = (d["lad_roll"].astype(np.float64) - d["lad_Fp"].astype(np.float64)) / s[:, None, :, None, None]
  DDp = (Dp ** 2).sum((2, 3, 4))
  o4s = []
  for j in range(4):
    o4s.append(ratio((o4[:, :, j] * Dp).sum((2, 3, 4)), DDp, idx))
  S["out4"] = np.array(o4s).transpose(1, 0, 2)  # [3, 4, K]
  S["out4_norm"] = np.array([np.sqrt((o4[:, :, j] ** 2).sum((0, 2, 3, 4)) / DDp.sum(0)) for j in range(4)])  # [4, K]
  J["out4_norm"] = S["out4_norm"].round(3).tolist()
  res = Dp - o4.sum(2)
  J["out4_resid_rel"] = float(np.sqrt((res ** 2).sum() / (Dp ** 2).sum()))
  J["out4_shares"] = {n: [float(x) for x in S["out4"][0, j]] for j, n in enumerate(["net", "scale", "level", "trend"])}

  # sigma/mu drift diagnostics
  J["log_sigmaR_over_sigmaF_median"] = [float(x) for x in np.median(np.abs(np.log(st[:, :, 1] / st[:, :, 3])), axis=(0, 2))]
  J["dmu_over_sigmaF_median"] = [float(x) for x in np.median(np.abs(st[:, :, 0] - st[:, :, 2]) / st[:, :, 3], axis=(0, 2))]

  S["ks"] = np.array(ks)
  out = Path(args.root)
  np.savez(out / "summary.npz", **S)
  (out / "summary.json").write_text(json.dumps(J, indent=1))
  print(json.dumps(J, indent=1))


if __name__ == "__main__":
  main()
