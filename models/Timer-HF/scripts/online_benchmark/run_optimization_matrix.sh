#!/usr/bin/env bash
set -euo pipefail

CKPT=${CKPT:-checkpoints/Timer-base-84M}
DATA_ROOT=${DATA_ROOT:-${ROLLKV_DATASETS:-datasets}}
OUTPUT_ROOT=${OUTPUT_ROOT:-results/timer_optimization_0811}
CONTEXT_LENGTHS=${CONTEXT_LENGTHS:-2880}
STEPS=${STEPS:-48}
REFRESH_LENGTHS=${REFRESH_LENGTHS:-1,2,3,4,16,0}
ADAPTIVE_THRESHOLDS=${ADAPTIVE_THRESHOLDS:-}
ADAPTIVE_MIN_REFRESH=${ADAPTIVE_MIN_REFRESH:-2}
ADAPTIVE_MAX_REFRESH=${ADAPTIVE_MAX_REFRESH:-16}
MODE=${MODE:-baseline}

case "$MODE" in
  baseline) rope_args=(--no-rope-rebase) ;;
  rope_rebase) rope_args=(--rope-rebase) ;;
  *) echo "MODE must be baseline or rope_rebase" >&2; exit 2 ;;
esac

datasets=(
  "ETTh1|ETT-small/ETTh1.csv|OT|11520|24"
  "ETTh2|ETT-small/ETTh2.csv|OT|11520|24"
  "ETTm1|ETT-small/ETTm1.csv|OT|46080|96"
  "ETTm2|ETT-small/ETTm2.csv|OT|46080|96"
  "electricity|electricity/electricity.csv|0|18432|24"
  "traffic|traffic/traffic.csv|0|12288|24"
  "weather|weather/weather.csv|T (degC)|36864|144"
)

mkdir -p "$OUTPUT_ROOT/$MODE"
for context_length in $CONTEXT_LENGTHS; do
  for spec in "${datasets[@]}"; do
    IFS='|' read -r name relative_csv column start_index naive_period <<< "$spec"
    python Timer-HF/scripts/online_benchmark/eval_graph_refresh.py \
      --ckpt "$CKPT" \
      --csv "$DATA_ROOT/$relative_csv" \
      --column "$column" \
      --start-index "$start_index" \
      --steps "$STEPS" \
      --context-length "$context_length" \
      --horizon 96 \
      --refresh-lengths "$REFRESH_LENGTHS" \
      --adaptive-thresholds "$ADAPTIVE_THRESHOLDS" \
      --adaptive-min-refresh "$ADAPTIVE_MIN_REFRESH" \
      --adaptive-max-refresh "$ADAPTIVE_MAX_REFRESH" \
      --naive-period "$naive_period" \
      "${rope_args[@]}" \
      --output "$OUTPUT_ROOT/$MODE/${name}_L${context_length}.json"
  done
done
