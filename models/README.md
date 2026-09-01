# Model repositories

Seven third-party time-series foundation models, each carrying our rolling
KV-cache implementation. Every repository keeps its own `LICENSE`;
`UPSTREAM.json` records the origin URL and, where the working tree was a git
checkout, the exact upstream commit.

Vendoring rather than cloning keeps the artifact runnable on an air-gapped
machine — the benchmark host itself has no outbound network.

## What is ours

| Repository | Rolling-cache engine | Benchmark scripts |
| --- | --- | --- |
| `TimesFM-2.5` | `src/timesfm/online/` | `scripts/online_benchmark/` |
| `Time-MoE` | `time_moe/online/` | `scripts/online_benchmark/` |
| `Timer-HF` | `timer_online/` | `scripts/online_benchmark/` |
| `Sundial-HF` | `sundial_online/` | `scripts/online_benchmark/` |
| `Toto` | `toto2/toto2/online_rolling.py` | `scripts/` |
| `OpenLTM` | `timer_xl_online/` | `scripts/eval_timer_xl_refresh.py`, `scripts/bench_timer_xl_rolling.py` |
| `Lag-Llama` | `lag_llama/online/` | `scripts/bench_online_rolling.py`, `scripts/eval_online_refresh.py` |

Exactly one upstream source file is edited anywhere in the tree:
`Time-MoE/time_moe/models/modeling_time_moe.py`. The MoE layer routes tokens
with `torch.where`, which makes expert GEMM shapes data-dependent and therefore
uncapturable by CUDA Graph; the edit adds a dense fallback path the graph runner
can capture. It is confined to `TimeMoeSparseExpertsLayer` and marked in place by
a comment at the top of the added branch. Every other file listed above is new.

To see it as a diff, check out the pinned commit from `UPSTREAM.json` and compare:

```bash
git clone https://github.com/Time-MoE/Time-MoE.git upstream-timemoe
cd upstream-timemoe && git checkout 915bfda4c78a544d62a2bec6ab22948423059236
diff -u time_moe/models/modeling_time_moe.py \
        ../models/Time-MoE/time_moe/models/modeling_time_moe.py
```

## These are reduced to the import closure

Each directory holds only what the harness actually loads, not the full upstream
tree. What that leaves per repository:

| Repository | Kept | Dropped |
| --- | --- | --- |
| `TimesFM-2.5` | `src/timesfm/` torch path | TimesFM 1.0 (`v1/`), the JAX/Flax backend, examples, upstream tests |
| `Time-MoE` | `time_moe/models/`, `time_moe/online/` | trainer, dataset tooling, training entry points |
| `Toto` | `toto2/`, `dd_unit_scaling/` | Toto 1.0 (`toto/`), the BOOM benchmark dataset, CI config |
| `OpenLTM` | Timer-XL and the four layer modules it imports | six other model architectures, training pipeline, shell recipes |
| `Lag-Llama` | `lag_llama/model/`, `gluon_utils/` | GluonTS estimator wrapper, training scripts, data, images |
| `Timer-HF`, `Sundial-HF` | our engine only | see below |

`Timer-HF` and `Sundial-HF` contain **no upstream source at all**. Both models
are loaded with `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`,
so their modelling code ships inside the HuggingFace checkpoint directory rather
than in a source repository. Whatever those directories would have held is
redundant with the checkpoint you download per the main `README.md`.

A vendored directory is therefore not byte-identical to its upstream commit —
it is a subset of it, plus our files. Nothing our code imports was removed: the
reduction was derived from the import graph and then checked by running the EXP-0
correctness gates for all seven models against this tree.

## Directory layout and the harness

`experiments/common.py` resolves model repositories through `MODELS_ROOT`,
which defaults to this directory and falls back to the repository root (the
layout the original development tree used). Override with `ROLLKV_MODELS` if you
place them elsewhere.
