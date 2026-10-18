#!/usr/bin/env bash
# Forecasting: sMSE and CRPS for all datasets × horizons × models.
# Usage: ./run_forecasting.sh [--test] [hydra overrides...]
#   --test   etth1 only, horizon 96 only, fewer origins (quick pipeline check)
# Full run: 5 datasets × 4 horizons × 6 models.
set -euo pipefail
cd "$(dirname "$0")"
export HYDRA_FULL_ERROR=1

DATASETS_FULL='[etth1,etth2,ettm1,ettm2,weather]'
MODELS='[ivdfm,itransformer,timemixer,timexer,ddfm,mlpmultivariate]'
HORIZONS_FULL='[96,192,336,720]'
OUTPUT_DIR=./results/forecasting
# Optional: set to a path to write LaTeX tables. Default: disabled.
# (Disabled by default so runs do not write outside results/.)
TABLES_DIR="${TABLES_DIR:-}"

# DDFM: shrunk for speed (scripts can override)
DDFM_MAX_EPOCH=30
DDFM_LAYERS='[32,16]'
DDFM_WINDOW=96

TEST_MODE=
ARGS=()
for a in "$@"; do
  if [ "$a" = "--test" ]; then
    TEST_MODE=1
  else
    ARGS+=("$a")
  fi
done

if [ -n "$TEST_MODE" ]; then
  echo "=== Test mode: etth1, horizon 96, all models, forecasting_max_origins=2 ==="
  CMD=(uv run python -u run.py experiment=forecasting \
    experiment.datasets='[etth1]' \
    experiment.models="$MODELS" \
    experiment.horizons='[96]' \
    experiment.window=96 \
    experiment.stride=25 \
    experiment.forecasting_max_origins=2 \
    experiment.ddfm_max_epoch=3 \
    experiment.ddfm_encoder_layers="$DDFM_LAYERS" \
    experiment.ddfm_window_size=$DDFM_WINDOW \
    experiment.output_dir="$OUTPUT_DIR" \
    "${ARGS[@]}")
  if [ -n "$TABLES_DIR" ]; then
    CMD+=(experiment.forecasting_tables_dir="$TABLES_DIR")
  fi
  "${CMD[@]}"
  echo "=== Done (test): results in $OUTPUT_DIR ==="
  exit 0
fi

echo "=== Full run: datasets=$DATASETS_FULL horizons=$HORIZONS_FULL models=$MODELS ==="
CMD=(uv run python -u run.py experiment=forecasting \
  experiment.datasets="$DATASETS_FULL" \
  experiment.models="$MODELS" \
  experiment.horizons="$HORIZONS_FULL" \
  experiment.window=96 \
  experiment.stride=25 \
  experiment.ddfm_max_epoch=$DDFM_MAX_EPOCH \
  experiment.ddfm_encoder_layers="$DDFM_LAYERS" \
  experiment.ddfm_window_size=$DDFM_WINDOW \
  experiment.output_dir="$OUTPUT_DIR" \
  "${ARGS[@]}")
if [ -n "$TABLES_DIR" ]; then
  CMD+=(experiment.forecasting_tables_dir="$TABLES_DIR")
fi
"${CMD[@]}"
echo "=== Done: results in $OUTPUT_DIR ==="
