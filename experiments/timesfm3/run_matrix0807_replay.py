"""TimesFM-3.0 replay of the 0807 CUDA-graph refresh matrix (paper §4.5 prose).

The TimesFM-2.5 numbers in §4.5 ("K=4 at L=512 vs L=8192", "K=16 ... worst-
dataset D", "never refreshing reaches ...", the c_N calibration) come from the
0807 matrix: one fixed window per dataset, L in {512, 2048, 8192},
K in {1, 4, 16, 64, 0}, 64 updates of one 32-point patch, H = 64, latency
measured inline per update, both paths as CUDA-graph replays.  This launcher
runs the SAME cells for TimesFM-3.0 with
``models/TimesFM-3.0/scripts/online_benchmark/eval_graph_refresh.py`` (the
line-for-line port of the 2.5 driver), one fresh process per cell exactly like
the original ``run_graph_refresh_matrix.py`` (CUDA-graph memory pools of a
cell are released before the next one).

Cell table (csv, column, start_index, naive_period) is the original 0807
runner's ``DATASETS`` table; the reference ``series_length`` / ``naive_mae``
are the values recorded in the TimesFM-2.5 cell JSONs and are checked before
any GPU work, so a different copy of a dataset cannot slip in.

Outputs (same names and schema as the 2.5 tree, model = "TimesFM-3.0"):

  $ROLLKV_RESULTS/MATRIX_0807/timesfm3/<dataset>_L<L>.json   one per cell
  $ROLLKV_RESULTS/MATRIX_0807/timesfm3/summary.json          flattened rows
  $ROLLKV_RESULTS/MATRIX_0807/timesfm3/_meta/<cell>.json     GPU/versions/verify
  $ROLLKV_RESULTS/MATRIX_0807/timesfm3/_logs/<cell>.log      per-cell stdout

Idempotent: a cell whose JSON already exists and is complete is skipped
(``--force`` recomputes).  ``summary.json`` is rebuilt after every run from
all canonical cells present on disk, in the 2.5 row order.

Usage (server):
  source /data/zxj/rolling-kv-cache/env.sh
  python $ROLLKV_ROOT/code/experiments/run_matrix0807_timesfm3.py --gpu 5
  python .../run_matrix0807_timesfm3.py --gpu 5 --datasets weather --lengths 512,2048
  python .../run_matrix0807_timesfm3.py --summary-only
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE_ROOT = HERE.parents[1]

MODEL_KEY = "timesfm3"
MODEL_NAME = "TimesFM-3.0"

# name -> (relative csv, column, start_index, naive_period); verbatim from the
# original 0807 launcher (results/_server_snapshot_20260908/scripts/
# run_graph_refresh_matrix.py).  Order = 2.5 summary.json row order.
DATASETS = {
    "ETTh1": ("ETT-small/ETTh1.csv", "OT", 11520, 24),
    "ETTh2": ("ETT-small/ETTh2.csv", "OT", 11520, 24),
    "ETTm1": ("ETT-small/ETTm1.csv", "OT", 46080, 96),
    "ETTm2": ("ETT-small/ETTm2.csv", "OT", 46080, 96),
    "weather": ("weather/weather.csv", "T (degC)", 36864, 144),
    "electricity": ("electricity/electricity.csv", "0", 18432, 24),
    "traffic": ("traffic/traffic.csv", "0", 12288, 24),
}

# (series_length, naive_mae) recorded by the TimesFM-2.5 0807 cell JSONs
# (plot/2026VLDB/sections/55-models/data/raw/graph_refresh_matrix_0807/
# timesfm/<name>_L*.json).  naive_mae is recomputed here with the driver's own
# float32 expression, so a bit-identical series reproduces it exactly.
REFERENCE = {
    "ETTh1": (17420, 2.4079220294952393),
    "ETTh2": (17420, 3.0536415576934814),
    "ETTm1": (69680, 2.411865472793579),
    "ETTm2": (69680, 3.0484611988067627),
    "weather": (52696, 2.87797474861145),
    "electricity": (26304, 7.840775966644287),
    "traffic": (17544, 0.015443452633917332),
}

CONTEXT_LENGTHS = (512, 2048, 8192)
REFRESH_LENGTHS = "1,4,16,64,0"
STEPS = 64
HORIZON = 64
PATCH = 32


def env_path(name: str, fallback: Path) -> Path:
  value = os.environ.get(name)
  return Path(value) if value else fallback


def split_csv(text: str, cast=str) -> list:
  return [cast(x.strip()) for x in text.split(",") if x.strip()]


def now() -> str:
  return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def cell_name(name: str, context: int) -> str:
  return f"{name}_L{context}"


def write_json_atomic(path: Path, value) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  tmp = path.with_name(path.name + ".partial")
  with open(tmp, "w") as f:
    json.dump(value, f, indent=2)
  os.replace(tmp, path)


def preflight_dataset(dataset_root: Path, name: str) -> Path:
  """Check the server copy reproduces the 2.5 series before spending GPU."""
  import numpy as np
  import pandas as pd

  rel, column, start, period = DATASETS[name]
  csv_path = dataset_root / rel
  if not csv_path.is_file():
    raise FileNotFoundError(f"{name}: {csv_path} missing")
  series = pd.to_numeric(pd.read_csv(csv_path)[column], errors="raise").to_numpy(
      np.float32
  )
  naive = float(np.abs(series[period:start] - series[:start - period]).mean())
  ref_len, ref_naive = REFERENCE[name]
  if len(series) != ref_len or not math.isclose(naive, ref_naive, rel_tol=1e-6):
    raise ValueError(
        f"{name}: {csv_path} gives series_length={len(series)} naive_mae={naive!r}; "
        f"the TimesFM-2.5 0807 cell recorded {ref_len} / {ref_naive!r}"
    )
  max_needed = start + STEPS * PATCH + HORIZON
  if max_needed > len(series):
    raise ValueError(f"{name}: needs {max_needed} points, has {len(series)}")
  print(f"[preflight] {name:<11} {csv_path}  n={len(series)}  "
        f"naive_mae={naive:.6g}  (matches 2.5)", flush=True)
  return csv_path


def cell_complete(path: Path, context: int, refresh: list[int], steps: int,
                  horizon: int) -> bool:
  if not path.is_file():
    return False
  try:
    with open(path) as f:
      result = json.load(f)
  except (OSError, json.JSONDecodeError):
    return False
  return (
      result.get("model") == MODEL_NAME
      and result.get("context_length") == context
      and result.get("prediction_length") == horizon
      and all(
          str(k) in result.get("methods", {})
          and result["methods"][str(k)].get("steps") == steps
          for k in refresh
      )
  )


def run_cell(args, csv_path: Path, name: str, context: int, out_dir: Path) -> None:
  _, column, start, period = DATASETS[name]
  cell = cell_name(name, context)
  out = out_dir / f"{cell}.json"
  tmp = out_dir / f"{cell}.json.partial"
  meta = out_dir / "_meta" / f"{cell}.json"
  log = out_dir / "_logs" / f"{cell}.log"
  log.parent.mkdir(parents=True, exist_ok=True)
  command = [
      sys.executable,
      str(args.script),
      "--ckpt", str(args.ckpt),
      "--csv", str(csv_path),
      "--column", column,
      "--start-index", str(start),
      "--steps", str(args.steps),
      "--context-length", str(context),
      "--horizon", str(args.horizon),
      "--refresh-lengths", args.refresh_lengths,
      "--naive-period", str(period),
      "--output", str(tmp),
      "--meta-output", str(meta),
  ]
  if args.verify:
    command.append("--verify")
  env = os.environ.copy()
  env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  if args.gpu is not None:
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
  print(f"\n=== {MODEL_KEY} {name} L={context}  [{now()}]"
        f"  gpu={env.get('CUDA_VISIBLE_DEVICES')} ===", flush=True)
  t0 = time.perf_counter()
  with open(log, "w") as fh:
    fh.write(" ".join(command) + "\n")
    fh.flush()
    proc = subprocess.Popen(command, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
      sys.stdout.write(line)
      fh.write(line)
    proc.wait()
  if proc.returncode != 0:
    raise RuntimeError(f"{cell} failed (exit {proc.returncode}); see {log}")
  with open(tmp) as f:
    result = json.load(f)
  ref_len, ref_naive = REFERENCE[name]
  ds = result["dataset"]
  if ds["series_length"] != ref_len or not math.isclose(
      ds["naive_mae"], ref_naive, rel_tol=1e-6):
    raise RuntimeError(f"{cell}: dataset metadata diverged from the 2.5 cell: {ds}")
  os.replace(tmp, out)
  print(f"[done] {cell} in {time.perf_counter() - t0:.1f}s -> {out}", flush=True)
  if args.verify and meta.is_file():
    with open(meta) as f:
      ver = (json.load(f).get("verification") or {})
    worst = max(
        (v.get("max_rel_diff_vs_eager", 0.0) for k, v in ver.items() if k != "wall_s"),
        default=0.0,
    )
    if worst > args.verify_tol:
      print(f"[WARN] {cell}: graph vs eager max rel diff {worst:.3e} > "
            f"{args.verify_tol:.0e}", flush=True)


def build_summary(out_dir: Path) -> list[dict]:
  """Flatten every present canonical cell into the 2.5 summary row schema."""
  rows = []
  for name in DATASETS:
    for context in CONTEXT_LENGTHS:
      path = out_dir / f"{cell_name(name, context)}.json"
      if not path.is_file():
        continue
      with open(path) as f:
        result = json.load(f)
      for refresh, metrics in result["methods"].items():
        rows.append({
            "model": result["model"],
            "dataset": os.path.basename(result["dataset"]["csv"]).removesuffix(".csv"),
            "column": result["dataset"]["column"],
            "context_length": result["context_length"],
            "horizon": result["prediction_length"],
            "refresh_length": int(refresh),
            **metrics,
        })
  write_json_atomic(out_dir / "summary.json",
                    {"execution": "cuda_graph_only", "rows": rows})
  return rows


def print_digest(rows: list[dict]) -> None:
  """Console digest in the §4.5 aggregation (per-dataset ratios, then median).

  speedup = mean latency(K=1) / mean latency(K); G = gap / MAE(K=1);
  D = MAE delta / MAE(K=1); both in percent; worst D = max over datasets.
  """
  if not rows:
    print("no cells present yet")
    return
  base = {(r["dataset"], r["context_length"]): r
          for r in rows if r["refresh_length"] == 1}
  cells = {}
  for r in rows:
    b = base.get((r["dataset"], r["context_length"]))
    if b is None:
      continue
    cells.setdefault((r["context_length"], r["refresh_length"]), []).append((
        r["dataset"],
        b["mean_latency_ms"] / r["mean_latency_ms"],
        100.0 * r["prediction_gap_mae_vs_full_k1"] / b["mae"],
        100.0 * r["forecast_mae_delta_vs_full_k1"] / b["mae"],
        r["mean_latency_ms"],
    ))
  label = ", ".join(sorted({r["model"] for r in rows}))
  print(f"\n{label} 0807 matrix digest (median over datasets present)")
  print(f"{'L':>6} {'K':>4} {'n':>2} {'lat_ms':>8} {'speedup':>8} {'G%':>7} "
        f"{'D%med':>7} {'D%max':>7}  worst")
  for (L, K) in sorted(cells, key=lambda x: (x[0], x[1] == 0, x[1])):
    v = cells[(L, K)]
    worst = max(v, key=lambda x: x[3])
    print(f"{L:>6} {K:>4} {len(v):>2} {statistics.median(x[4] for x in v):>8.3f} "
          f"{statistics.median(x[1] for x in v):>7.2f}x "
          f"{statistics.median(x[2] for x in v):>7.2f} "
          f"{statistics.median(x[3] for x in v):>7.2f} {worst[3]:>7.2f}  {worst[0]}")
  for L in CONTEXT_LENGTHS:
    ceil = cells.get((L, 0))
    if not ceil:
      continue
    s_inf = statistics.median(x[1] for x in ceil)
    c_n = 1.0 / s_inf
    parts = []
    for K in (4, 16, 64):
      if (L, K) in cells:
        pred = K / (1 + (K - 1) * c_n)
        meas = statistics.median(x[1] for x in cells[(L, K)])
        parts.append(f"K={K}: pred {pred:.2f}x / meas {meas:.2f}x")
    print(f"c_N calibration L={L}: ceiling {s_inf:.2f}x -> c_N={c_n:.4f}; "
          + "; ".join(parts))


def main(argv=None) -> int:
  root = env_path("ROLLKV_ROOT", CODE_ROOT.parent)
  models = env_path("ROLLKV_MODELS", CODE_ROOT / "models")
  results = env_path("ROLLKV_RESULTS", root / "results")
  parser = argparse.ArgumentParser(description=__doc__,
                                   formatter_class=argparse.RawDescriptionHelpFormatter)
  parser.add_argument("--gpu", type=int, default=None,
                      help="physical GPU index (PCI order); default: inherit "
                      "CUDA_VISIBLE_DEVICES")
  parser.add_argument("--datasets", default=",".join(DATASETS))
  parser.add_argument("--lengths", "--context-lengths", dest="lengths",
                      default=",".join(map(str, CONTEXT_LENGTHS)))
  parser.add_argument("--refresh-lengths", default=REFRESH_LENGTHS)
  parser.add_argument("--steps", type=int, default=STEPS)
  parser.add_argument("--horizon", type=int, default=HORIZON)
  parser.add_argument("--output-dir", type=Path,
                      default=results / "MATRIX_0807" / MODEL_KEY)
  parser.add_argument("--ckpt", type=Path,
                      default=env_path("ROLLKV_CKPT", root / "checkpoints") / "TimesFM-3.0")
  parser.add_argument("--dataset-root", type=Path,
                      default=env_path("ROLLKV_DATASETS", root / "datasets"))
  parser.add_argument("--script", type=Path,
                      default=models / "TimesFM-3.0" / "scripts" / "online_benchmark"
                      / "eval_graph_refresh.py")
  parser.add_argument("--force", action="store_true", help="recompute done cells")
  parser.add_argument("--no-verify", dest="verify", action="store_false",
                      help="skip the post-timing eager/decode cross-check")
  parser.add_argument("--verify-tol", type=float, default=1e-5)
  parser.add_argument("--dry-run", action="store_true")
  parser.add_argument("--summary-only", action="store_true",
                      help="only rebuild summary.json + digest from existing cells")
  args = parser.parse_args(argv)

  datasets = split_csv(args.datasets)
  lengths = split_csv(args.lengths, int)
  refresh = split_csv(args.refresh_lengths, int)
  unknown = [x for x in datasets if x not in DATASETS]
  if unknown:
    raise SystemExit(f"unknown datasets {unknown}; choose from {list(DATASETS)}")
  if 1 not in refresh:
    raise SystemExit("--refresh-lengths must include 1 (full-recompute reference)")
  off_grid = [x for x in lengths if x not in CONTEXT_LENGTHS]
  if off_grid:
    print(f"[note] lengths {off_grid} are outside the 0807 grid "
          f"{CONTEXT_LENGTHS}; they are run but not included in summary.json")
  out_dir = args.output_dir.resolve()
  out_dir.mkdir(parents=True, exist_ok=True)

  if not args.summary_only:
    todo = [(n, L) for n in datasets for L in lengths
            if args.force or not cell_complete(
                out_dir / f"{cell_name(n, L)}.json", L, refresh, args.steps,
                args.horizon)]
    skipped = len(datasets) * len(lengths) - len(todo)
    print(f"[plan] {MODEL_NAME}: {len(todo)} cells to run, {skipped} already done; "
          f"out={out_dir}  script={args.script}  ckpt={args.ckpt}", flush=True)
    if args.dry_run:
      for n, L in todo:
        print(f"  {cell_name(n, L)}")
      return 0
    if not args.script.is_file():
      raise SystemExit(f"driver not found: {args.script}")
    csv_paths = {n: preflight_dataset(args.dataset_root, n)
                 for n in dict.fromkeys(n for n, _ in todo)}
    t0 = time.perf_counter()
    for n, L in todo:
      run_cell(args, csv_paths[n], n, L, out_dir)
    print(f"\n[plan] ran {len(todo)} cells in {time.perf_counter() - t0:.1f}s", flush=True)

  rows = build_summary(out_dir)
  present = len({(r["dataset"], r["context_length"]) for r in rows})
  total = len(DATASETS) * len(CONTEXT_LENGTHS)
  print(f"\nsummary: {out_dir / 'summary.json'}  ({len(rows)} rows, "
        f"{present}/{total} cells{'' if present == total else ', INCOMPLETE'})")
  print_digest(rows)
  return 0


if __name__ == "__main__":
  sys.exit(main())
