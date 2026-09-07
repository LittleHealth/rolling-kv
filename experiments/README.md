# Experiment harness

| File | Role |
| --- | --- |
| `common.py` | model/dataset registry, path resolution, append-only JSONL I/O, provenance |
| `adapters.py` | uniform CUDA-Graph interface over all seven models |
| `exp0_w1.py`, `exp0_all.py` | correctness gates T1–T5 |
| `exp1_w1.py` | fixed-policy sweep |
| `exp2_stages.py` | stage-level time decomposition |
| `exp3_adaptive.py` | adaptive refresh policies |
| `exp5_appendix.py` | eager vs graph, batch scaling |
| `aggregate.py` | derived quantities, including all of EXP-4 |
| `validate_all.py` | acceptance checks over a finished run |
| `launch_all.py`, `launch_wave.py` | disconnect-safe queue runners |

## Path resolution

`common.py` resolves four roots, each overridable by environment variable:
`ROLLKV_MODELS` (default `models/`, falling back to the repository root),
`ROLLKV_CKPT`, `ROLLKV_DATASETS`, and `ROLLKV_RESULTS`. This is the only
difference between the harness here and the copy that ran on the benchmark host,
where the model repositories sat directly at the repository root.

## A known inefficiency in the EXP-3 grid

The grid is `7 thetas × 4 max_ages × 4 budgets × 2 calibs` = 224 policies per
cell, and roughly 98.5% of that is duplicate work. `budget_pct` only enters
through `realized_cap()` when `calib="per_model"`, so with `calib="none"` the
four budgets are identical runs. Small thetas make the drift signal fire every
step, degenerating the policy to full recompute. Measured, the 224 policies
collapse to 10–14 distinct `(n_refresh, MAE)` signatures per cell.

This matters for cost: EXP-3 on Time-MoE runs about 6.3 h per
(dataset, window, L) cell. Dedupe by realized refresh schedule before running
the full grid.

## The complete queue

`launch_all.py` is the single disconnect-safe entry point for every experiment.
It runs the fixed W1, W2, and W3 waves in order, including EXP-0, EXP-1, EXP-2,
EXP-3, EXP-5, deterministic aggregation, and final acceptance validation. EXP-4
is derived by `aggregate.py` from EXP-1 rather than run separately.

The queue records its durable cursor in `$ROLLKV_RESULTS/all_queue_state.json`
after every command and uses `all_queue.lock` in the same directory to prevent
two launchers from running concurrently. A restart with `--resume` continues from
the next durable task. Individual JSONL writers are append-only and each task
also skips already complete keys, so rerunning a failed task is safe.

Full campaign, detached so it survives a disconnect:

```bash
export PYTHONPATH=$PWD/experiments
export ROLLKV_RESULTS=$PWD/results
setsid -f env CUDA_VISIBLE_DEVICES=2 \
  python -u experiments/launch_all.py \
  --gpu 2 --windows 5 --retries 2 --resume \
  >> "$ROLLKV_RESULTS/all_queue.log" 2>&1 < /dev/null
```

One wave can also run independently on another GPU without triggering aggregate
or validation:

```bash
CUDA_VISIBLE_DEVICES=3 python -u experiments/launch_wave.py \
  --wave W3 --gpu 3 --windows 5 --retries 2 --resume
```

The wave queue uses its own `w3_queue.lock`, `w3_queue_state.json`, and log.
When the full queue later reaches that wave, completed grid cells are skipped.

The queue is long: 3444 tasks across the three waves, dominated by EXP-3. The
EXP-3 grid redundancy noted above is worth removing before committing to it.

Completion requires `manifest.json` to contain a non-null
`all_experiments_finished_at` and `all_validation_passed: true`. The detailed
acceptance report is written to `validation_all.json` alongside it.
