# Model repositories

Nine third-party time-series foundation models, each carrying our rolling
KV-cache implementation. Every repository keeps its own `LICENSE` (for the
HuggingFace checkpoints, the model card declares it); `UPSTREAM.json`
records the origin URL and the exact upstream revision each vendored tree
was verified against.

The git history separates provenance from contribution: each model has a
vendor commit (byte-identical upstream files, with the source URL, revision,
and download date in the commit message) followed by an implementation commit
(our files, plus any upstream edit). `git log --oneline -- models/<name>`
shows the pair; `git show` on the implementation commit is the full diff of
what we added.

Vendoring rather than cloning keeps the artifact runnable on an air-gapped
machine — the benchmark host itself has no outbound network.

## What is ours

| Repository | Rolling-cache engine | Benchmark scripts |
| --- | --- | --- |
| `TimesFM-2.5` | `src/timesfm/online/` | `scripts/online_benchmark/` |
| `TimesFM-3.0` | `src/timesfm3/online/` | `scripts/online_benchmark/` |
| `Timer-S1` | `timers1_online/` (eager-only v1: MoE routing blocks CUDA Graph, as in Time-MoE) | `scripts/online_benchmark/` |
| `Time-MoE` | `time_moe/online/` | `scripts/online_benchmark/` |
| `Timer-HF` | `timer_online/` | `scripts/online_benchmark/` |
| `Sundial-HF` | `sundial_online/` | `scripts/online_benchmark/` |
| `Toto` | `toto2/toto2/online_rolling.py` | `scripts/` |
| `OpenLTM` | `timer_xl_online/` | `scripts/eval_timer_xl_refresh.py`, `scripts/bench_timer_xl_rolling.py` |
| `Lag-Llama` | `lag_llama/online/` | `scripts/bench_online_rolling.py`, `scripts/eval_online_refresh.py` |

Exactly two upstream source files are edited anywhere in the tree:

- `Time-MoE/time_moe/models/modeling_time_moe.py` — the substantive one. The
  MoE layer routes tokens with `torch.where`, which makes expert GEMM shapes
  data-dependent and therefore uncapturable by CUDA Graph; the edit adds a
  dense fallback path the graph runner can capture. It is confined to
  `TimeMoeSparseExpertsLayer` and marked in place by a comment at the top of
  the added branch.
- `Toto/toto2/toto2/model.py` — a two-line Python 3.10 compatibility shim:
  `typing.NotRequired` exists only from Python 3.11, so the import falls back
  to `typing_extensions`. No functional change.

Every other file listed above is new. To see either edit as a diff, `git show`
the model's implementation commit, or diff against the pinned upstream revision
from `UPSTREAM.json`.

## These are reduced to the import closure

Each directory holds only what the harness actually loads, not the full upstream
tree. What that leaves per repository:

| Repository | Kept | Dropped |
| --- | --- | --- |
| `TimesFM-2.5` | `src/timesfm/` torch path | TimesFM 1.0 (`v1/`), the JAX/Flax backend, examples, upstream tests |
| `TimesFM-3.0` | `src/timesfm3/` (self-contained torch package) | its `*_test.py` files, the 1.0/2.x packages, the Flax backend |
| `Timer-S1` | HF repo code files | weights (4 safetensors shards, ~16 GB) |
| `Time-MoE` | `time_moe/models/`, `time_moe/online/` | trainer, dataset tooling, training entry points |
| `Toto` | `toto2/`, `dd_unit_scaling/` | Toto 1.0 (`toto/`), the BOOM benchmark dataset, CI config |
| `OpenLTM` | Timer-XL and the four layer modules it imports | six other model architectures, training pipeline, shell recipes |
| `Lag-Llama` | `lag_llama/model/`, `gluon_utils/` | GluonTS estimator wrapper, training scripts, data, images |
| `Timer-HF`, `Sundial-HF` | HF repo code files | weights (`model.safetensors`) |

`Timer-HF`, `Sundial-HF`, and `Timer-S1` have no source repository: these models
are loaded with `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`,
so their modelling code ships inside the HuggingFace checkpoint repository. Their
vendor commits carry that repository's code files (model card, configs,
`modeling_*.py`) so the engines can be read against the model code they drive;
the weights are excluded — download the checkpoint per the main `README.md`.

A vendored directory is therefore not byte-identical to its upstream commit —
it is a subset of it, plus our files. Nothing our code imports was removed: the
reduction was derived from the import graph and then checked by running the EXP-0
correctness gates for all seven models against this tree.

## Directory layout and the harness

`experiments/common.py` resolves model repositories through `MODELS_ROOT`,
which defaults to this directory and falls back to the repository root (the
layout the original development tree used). Override with `ROLLKV_MODELS` if you
place them elsewhere.
