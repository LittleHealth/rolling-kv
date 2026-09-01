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
├── models/        seven vendored model repos + our rolling-cache engines
├── experiments/   the measurement harness (EXP-0 … EXP-5)
└── requirements.txt
```

The vendored repositories are reduced to what the harness actually imports —
1.1 MB across seven models rather than the full upstream trees.
`models/README.md` records what was kept and dropped per repository, and
`models/UPSTREAM.json` pins each one's origin and upstream commit.

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
  expert GEMM shapes are data-dependent and uncapturable. The one upstream file
  we modify anywhere in this artifact adds a graph-safe dense dispatcher for
  exactly that reason.

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

Roughly 3.4 GB in total. Download into `checkpoints/` using exactly these
directory names — `experiments/common.py` resolves each model's weights by name:

| Directory | Source | Used by |
| --- | --- | --- |
| `TimesFM-2.5-200M/model.safetensors` | HF `google/timesfm-2.5-200m-pytorch` | `timesfm` |
| `TimeMoE-50M/` | HF `Maple728/TimeMoE-50M` | `timemoe` |
| `Sundial-base-128M/` | HF `thuml/sundial-base-128m` | `sundial` |
| `Timer-base-84M/` | HF `thuml/timer-base-84m` | `timer` |
| `Toto-2.0-313m/` | HF `Datadog/Toto-2.0-313m` | `toto2` |
| `Timer-XL-67M/checkpoint.pth` | [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/01c35ca13f474176be7b/), linked from `models/OpenLTM/README.md` | `timerxl` |
| `Lag-Llama/lag-llama.ckpt` | HF `time-series-foundation-models/Lag-Llama` | `lagllama` |

`Sundial-base-128M/` and `Timer-base-84M/` must be full HuggingFace snapshots,
not just the weights: both models ship their modelling code inside the checkpoint
and are loaded with `trust_remote_code=True`. That is also why those two
directories under `models/` carry no upstream source of their own.

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
