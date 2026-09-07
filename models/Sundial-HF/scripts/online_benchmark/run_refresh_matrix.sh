#!/usr/bin/env bash
set -euo pipefail

CKPT=${CKPT:-checkpoints/Sundial-base-128M}
DATA_ROOT=${DATA_ROOT:-${ROLLKV_DATASETS:-datasets}}
OUTPUT_ROOT=${OUTPUT_ROOT:-results/sundial_rolling_0812}
CONTEXT_LENGTHS=${CONTEXT_LENGTHS:-2880}
STEPS=${STEPS:-64}
REFRESH_LENGTHS=${REFRESH_LENGTHS:-1,2,4,8,16,64,0}
ADAPTIVE_THRESHOLDS=${ADAPTIVE_THRESHOLDS:-}
ADAPTIVE_MIN_REFRESH=${ADAPTIVE_MIN_REFRESH:-4}
ADAPTIVE_MAX_REFRESH=${ADAPTIVE_MAX_REFRESH:-32}
NUM_SAMPLES=${NUM_SAMPLES:-1}
SAMPLING_STEPS=${SAMPLING_STEPS:-50}
SEED=${SEED:-7}
NOISE_MODE=${NOISE_MODE:-antithetic}
ROPE_MODE=${ROPE_MODE:-rebase}

case "$ROPE_MODE" in
  rebase) rope_args=(--rope-rebase) ;;
  baseline) rope_args=(--no-rope-rebase) ;;
  *) echo "ROPE_MODE must be rebase or baseline" >&2; exit 2 ;;
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

mkdir -p "$OUTPUT_ROOT/$ROPE_MODE"
for context_length in $CONTEXT_LENGTHS; do
  for spec in "${datasets[@]}"; do
    IFS='|' read -r name relative_csv column start_index naive_period <<< "$spec"
    python Sundial-HF/scripts/online_benchmark/eval_graph_refresh.py \
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
      --num-samples "$NUM_SAMPLES" \
      --sampling-steps "$SAMPLING_STEPS" \
      --seed "$SEED" \
      --noise-mode "$NOISE_MODE" \
      "${rope_args[@]}" \
      --output "$OUTPUT_ROOT/$ROPE_MODE/${name}_L${context_length}.json"
  done
done
