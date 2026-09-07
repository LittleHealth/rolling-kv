# Rolling KV Cache for Online Time-Series Forecasting — Artifact

Reference implementation and experiment harness for reusing survivor KV state
across sliding-window updates in time-series foundation models, instead of
recomputing the full context at every step.

The idea is simple: when an online forecaster slides its context window by a few
points, almost all of the attention state it needs is state it already computed.
A rolling cache keeps the survivors, evicts what fell out of the window, and
encodes only the new tokens.

This artifact ships the implementation and the scripts that measure it. It
contains no measurement data and no paper figures — run the harness to generate
your own.

## Layout

```
artifact/
├── models/        nine vendored model repos + our rolling-cache engines
├── experiments/   the measurement harness (EXP-0 … EXP-5)
└── requirements.txt
```

The vendored repositories are reduced to what the harness actually imports —
1.8 MB across nine models rather than the full upstream trees.
`models/README.md` records what was kept and dropped per repository, and
`models/UPSTREAM.json` pins each one's origin and upstream commit.

## Upstream provenance

Each vendored tree was re-downloaded from its upstream repository on
2026-09-08 and verified file-by-file against the revision below. The same
pin, plus the download date, is recorded in that model's vendor commit
(`git log --oneline -- models/<name>` shows the vendor/implementation pair;
`git show` on the implementation commit is the full diff of our additions).

| Directory | Upstream repository | Pinned revision | Rev. date |
| --- | --- | --- | --- |
| `models/TimesFM-2.5` | <https://github.com/google-research/timesfm> | `3dae50b20d7a724981e8ea36cda75578f80dd2dc` | 2026-07-13 |
| `models/TimesFM-3.0` | <https://github.com/google-research/timesfm> | `0df95ae62085a6ac0d0afd1ad40dee2e6c1356ab` | 2026-09-04 |
| `models/Time-MoE` | <https://github.com/Time-MoE/Time-MoE> | `915bfda4c78a544d62a2bec6ab22948423059236` | 2026-03-22 |
| `models/Lag-Llama` | <https://github.com/time-series-foundation-models/lag-llama> | `df7531a83a19b3c6a0222d703ca9bf59ef7a6ab9` | 2025-06-06 |
| `models/Toto` | <https://github.com/DataDog/toto> | `44ea4e88852228039564aa3e76fac26aafac0803` | 2026-06-03 |
| `models/OpenLTM` | <https://github.com/thuml/OpenLTM> | `0b3005099d380ecc00d512f72553c7fe8fccca99` | 2026-03-22 |
| `models/Timer-HF` | <https://huggingface.co/thuml/timer-base-84m> | `70077a71acce1b4c00d98332fcaabc694255d8e5` | 2025-08-03 |
| `models/Sundial-HF` | <https://huggingface.co/thuml/sundial-base-128m> | `3212e42564493f520593e5414af4367fc4b49226` | 2026-03-09 |
| `models/Timer-S1` | <https://huggingface.co/thuml/Timer-S1> | `dd92ce51c691454aa71709c0385155c2b780337d` | 2026-05-09 |

Toto and OpenLTM were originally vendored as unversioned snapshots; their pins
were recovered by matching file contents against upstream history. Timer-HF,
Sundial-HF, and Timer-S1 have no source repository — the pin is the HuggingFace
checkpoint repository revision whose code files are vendored (weights excluded).
TimesFM-3.0 pins master past the `v3.0.0` tag to pick up the upstream KV-cache
batch-indexing fix (`03675bf`); the first seven models were verified on
2026-09-08, the two newest were downloaded and vendored the same day.

## The implementation

Each model gets a rolling engine that owns three things: a KV ring buffer, an
eviction policy, and a position-remapping rule. `models/README.md` maps each
engine to its file. Two design points drive most of the behaviour:

- **Eviction is a mask update, not a memcpy.** The TimesFM ring buffer tags each
  slot with its absolute position and rebuilds the attention mask from the tags,
  so evicting a token costs one integer update. Time-MoE's `slice_cache`
  physically copies instead, and that copy costs 1.6–1.7% of each step.
- **CUDA Graph capture is where most of the win lives, and it does not apply
  uniformly.** It works on TimesFM and is bit-exact, taking a step from 26.0 ms
  to 4.7 ms. It does not work on Time-MoE: MoE routing uses `torch.where`, so
  expert GEMM shapes are data-dependent and uncapturable. The only substantive
  upstream edit in this artifact adds a graph-safe dense dispatcher for exactly
  that reason (the one other upstream edit, in Toto, is a two-line Python 3.10
  import shim).

## Setup

Measurements were taken on an NVIDIA A100-SXM4-40GB (driver 535.261.03) running
Ubuntu 22.04.5, Python 3.10.20, PyTorch 2.5.1+cu121 (CUDA 12.1, cuDNN 9.1.0),
transformers 4.45.2, numpy 1.25.2, pandas 2.1.1, gluonts 0.14.4.
`requirements.txt` is that environment's exact `pip freeze`.

```bash
conda create -n rolling-kv python=3.10 -y
conda activate rolling-kv
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Two version pins matter, and one non-obvious dependency set:

- **transformers 4.45.2.** `models/Time-MoE/time_moe/online/cache_utils.py`
  imports `DynamicLayer` defensively because that symbol does not exist before
  transformers 4.57. Both branches are exercised; a newer transformers works.
- **gluonts 0.14.4.** Lag-Llama's published checkpoint does not load against
  newer GluonTS. Toto's inference-only path uses APIs already present in 0.14.4,
  so one pin serves both.
- Beyond a standard PyTorch install the harness also needs `gluonts[torch]`,
  `openpyxl`, `unit-scaling`, and `jaxtyping`.

The harness reads four paths from the environment. Only the two unshipped ones
normally need setting; the others default correctly inside a checkout.

| Variable | Default | Meaning |
| --- | --- | --- |
| `ROLLKV_CKPT` | `checkpoints/` | model weights (not shipped) |
| `ROLLKV_DATASETS` | `datasets/` | forecasting datasets (not shipped) |
| `ROLLKV_MODELS` | `models/` | vendored model repositories |
| `ROLLKV_RESULTS` | `results/` | where runs write; created on first write |

### Checkpoints

Roughly 21 GB in total (3.4 GB for the original seven; TimesFM-3.0 adds 1.3 GB
and Timer-S1 16 GB). Download into `checkpoints/` using exactly these directory
names — `experiments/common.py` resolves each model's weights by name, and the
two newest models' gate scripts read the same layout:

| Directory | Source | Used by |
| --- | --- | --- |
| `TimesFM-2.5-200M/model.safetensors` | HF `google/timesfm-2.5-200m-pytorch` | `timesfm` |
| `TimeMoE-50M/` | HF `Maple728/TimeMoE-50M` | `timemoe` |
| `Sundial-base-128M/` | HF `thuml/sundial-base-128m` | `sundial` |
| `Timer-base-84M/` | HF `thuml/timer-base-84m` | `timer` |
| `Toto-2.0-313m/` | HF `Datadog/Toto-2.0-313m` | `toto2` |
| `Timer-XL-67M/checkpoint.pth` | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/01c35ca13f474176be7b/), linked from `models/OpenLTM/README.md` | `timerxl` |
| `Lag-Llama/lag-llama.ckpt` | HF `time-series-foundation-models/Lag-Llama` | `lagllama` |
| `TimesFM-3.0/` | HF `google/timesfm-3.0-pytorch` (config + safetensors) | `TimesFM-3.0` gates |
| `Timer-S1/` | HF `thuml/Timer-S1` (full snapshot, code + 4 shards) | `Timer-S1` gates |

`Sundial-base-128M/`, `Timer-base-84M/`, and `Timer-S1/` must be full HuggingFace
snapshots, not just the weights: these models ship their modelling code inside
the checkpoint and are loaded with `trust_remote_code=True`. The vendor commits
under `models/Timer-HF`, `models/Sundial-HF`, and `models/Timer-S1` mirror those
code files at the pinned revision for reference; at runtime the code is loaded
from the checkpoint snapshot, not from `models/`.

### Datasets

```
datasets/
├── ETT-small/{ETTh1,ETTh2,ETTm1,ETTm2}.csv
├── electricity/electricity.csv
├── traffic/traffic.csv
└── weather/weather.csv
```

These are the files distributed by
[thuml/Time-Series-Library](https://github.com/thuml/Time-Series-Library). The
target column each experiment reads is `OT` for the ETT family, column `0` for
Electricity and Traffic, and `T (degC)` for Weather; `experiments/common.py`
holds the registry, and each run re-records it in its own `manifest.json`.

## Running it

```bash
export PYTHONPATH=experiments
export ROLLKV_CKPT=$PWD/checkpoints
export ROLLKV_DATASETS=$PWD/datasets

# correctness gates first — nothing downstream is interpretable until these pass
python experiments/exp0_w1.py  --model timesfm    # W1 models
python experiments/exp0_all.py --model timerxl    # W2/W3 models

# one EXP-1 grid cell: latency, then quality
python experiments/exp1_w1.py --mode timing  --model timesfm --L 8192
python experiments/exp1_w1.py --mode quality --model timesfm \
    --dataset ETTh1 --window 0 --L 8192
```

The quality mode reads its evaluation window starts from `results/manifest.json`,
which the queue runner writes on first launch; run `launch_all.py --resume` once
(or call `common.create_manifest`) before driving `exp1_w1.py` by hand.

`experiments/README.md` documents the disconnect-safe queue runner that drives a
whole campaign, including its durable cursor and `--resume` semantics. Reruns are
safe: JSONL writers are append-only and every task skips keys it has already
completed.

Results are written as JSON Lines under `$ROLLKV_RESULTS`, one object per line,
each carrying `schema`, `run_id`, `model`, `git_sha`, `ts`, and `status`.
Infeasible grid cells are recorded as `status="unsupported"` rather than dropped,
so a missing row and a failed row stay distinguishable.

### The gate that matters

EXP-0's **T2 growing-window test** is the mandatory one. With append-only
updates and no eviction, rolling and full recompute are mathematically
identical, so any mismatch is an implementation bug rather than a cache
approximation gap. Every fp32 engine passes it at or below 1.3e-06, which is
fp32 noise. Do not interpret any measured quality gap before T2 passes.

All seven gate suites were run against this tree on an A100-SXM4-40GB. Every
gate passes; worst case across every context length, `rel_err`:

| Model | dtype | T1 forward | T2 growing window | T3 graph | T4 remap | T5 gap range |
| --- | --- | --- | --- | --- | --- | --- |
| `timesfm` | fp32 | 4.1e-07 | 1.3e-06 | 0.0 | 0.0 | 0.9 – 14.3% |
| `timemoe` | bf16 | 2.9e-02 | 1.4e-02 | 0.0 | 0.0 | 6.7 – 96.1% |
| `sundial` | fp32 | 0.0 | 3.5e-07 | 0.0 | 9.9e-15 | 0.6 – 22.9% |
| `timer` | fp32 | 0.0 | 3.0e-07 | 0.0 | 9.4e-15 | 1.2 – 22.8% |
| `toto2` | fp32 | 9.4e-07 | 5.1e-07 | 0.0 | 1.3e-15 | 1.0 – 30.4% |
| `timerxl` | fp32 | 3.2e-07 | 1.0e-07 | 0.0 | 9.9e-15 | 1.1 – 31.2% |
| `lagllama` | fp32 | 0.0 | 5.2e-07 | 0.0 | 0.0 | 1.6 – 1204% |

Thresholds are 1e-5 for fp32 and 2e-2 for bf16 on T1/T2, 1e-6 on T3, and 1e-12
on T4. Time-MoE's T1 compares the graph-safe dense dispatcher against the
dynamic one rather than against upstream, so it is recorded without a threshold.
T3 is exactly 0.0 everywhere: graph replay is bit-exact against eager.

T5 is diagnostic, not a gate — it measures the cache gap as a function of cache
age and is deliberately unasserted. Its range is wide, and it is **non-monotone
in cache age**, so an error bound over ages 1..K has to take the max over k
rather than the value at K. Lag-Llama's upper end is an outlier worth knowing
about before trusting its adaptive-policy numbers.

### The two newest models (added 2026-09-08)

`TimesFM-3.0` and `Timer-S1` are not wired into the EXP-0…EXP-5 harness yet;
their gates are standalone scripts, run and passed on an A100-SXM4-40GB:

```bash
python models/TimesFM-3.0/scripts/online_benchmark/test_exactness.py --device cuda:0
python models/Timer-S1/scripts/online_benchmark/test_exactness.py    --device cuda:0
```

| Model | dtype | T1 | T2 growing window | T3 graph | T4 | cache gap (measured) |
| --- | --- | --- | --- | --- | --- | --- |
| `TimesFM-3.0` | fp32 | 1.0e-07 | 5.0e-07 | 0.0 | 1.2e-07 (ring machinery) | 0.4 – 0.8% |
| `Timer-S1` | bf16 | 0.0 | 1.5e-02 | — (eager v1) | 2.0e-07 (RoPE rebase algebra) | 1.2 – 3.0% |

Two per-model caveats, both documented in the gate scripts: the vendored
TimesFM-3.0 forward is fp32-only (it promotes resblock inputs to fp32, which
torch 2.5.1 rejects against bf16 weights), so its gates refuse
`--dtype bfloat16`; and neither new model asserts recompute equivalence under
eviction — TimesFM-3.0's post-RoPE affine qk-RMSNorm/PerDimScale and Timer-S1's
per-dim `q_scale`/`k_scale` make attention depend on absolute positions, so the
eviction-age gap is measured (last column), not asserted, exactly like T5.
Timer-S1's rolling speedup in eager v1 is modest (~1.35–1.40x, launch-bound
MoE); the CUDA-graph path needs a Time-MoE-style static dispatch and is left as
future work.

## Experiments

| Block | Question | Entry point |
| --- | --- | --- |
| EXP-0 | Correctness gates T1–T5: custom-vs-upstream forward, growing window, graph-vs-eager, position-remap algebra, cache gap vs age | `exp0_w1.py`, `exp0_all.py` |
| EXP-1 | Fixed-policy sweep over model × dataset × window × context length × refresh interval K × position remap | `exp1_w1.py` |
| EXP-2 | Stage-level time decomposition (norm / embed / attention / FFN / head / cache) | `exp2_stages.py` |
| EXP-3 | Adaptive refresh: drift threshold × max cache age × error budget × calibration | `exp3_adaptive.py` |
| EXP-4 | Context length vs accuracy — derived from EXP-1's K=1 runs, never run separately | `aggregate.py` |
| EXP-5 | Appendix: eager vs graph, batch scaling, kernel-class breakdown | `exp5_appendix.py` |

All seven models pass the EXP-0 correctness gates on this tree (table above).
Beyond that, `sundial` and `timer` have not been run through the full EXP-1/2/3
protocol, so their sweep behaviour is unmeasured.

## Two baseline traps

Both are cases where a broken baseline manufactures a speedup. If you rerun the
sweeps, you will hit both:

- **TimesFM's upstream `decode()` spends up to 82% of its time in a serial
  Python Welford loop** over prefix normalization statistics — not in the
  transformer. For a fully observed window that loop is a prefix sum and
  collapses to two `cumsum` calls (statistics match to 3.6e-07). Any
  rolling-vs-full speedup must be reported against both the native baseline and
  this vectorized fair baseline. At L=16384, batch=1: **12.67× native, 1.21×
  fair.** The native number credits a Python-loop artefact to the KV cache.
- **Time-MoE's L=8192 numbers are not trustworthy.** The K=1 full-recompute
  baseline MAE jumps 4–5× going from L=4096 to L=8192 on every dataset (ETTh1
  1.84 → 10.17, Electricity 5.99 → 23.81, Weather 1.30 → 8.72) while
  L=512/2048/4096 baselines sit flat near 2.4. Because both arms are broken
  there, rolling looks nearly free. It is a broken baseline, not a result.
  TimesFM's baselines stay flat across L.

Structurally, rolling step latency is a **constant** (~31 ms Time-MoE, ~25 ms
TimesFM) regardless of context length and batch size, because it sits on the
kernel-launch floor — so speedup exists only where full recompute climbs above
that floor. Absolute latencies are hardware-specific; the ratios are the claims.

## License

Our code is MIT (`LICENSE`). Each vendored repository under `models/` retains
its own upstream license, listed in `models/UPSTREAM.json`; those licenses
govern their respective directories.
