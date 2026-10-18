#!/usr/bin/env bash
# Optuna hyperparameter search for iVDFM forecasting (minimize sMSE), one study per dataset.
# Usage: bash run_hyperparameter.sh [--test]
#   --test  2 trials per dataset on half the training windows (pipeline check)
# Best parameters are written to results/optuna/<dataset>_ivdfm_best.json. The values used in the
# paper are fixed in config/experiment/forecasting.yaml (ivdfm_dataset_params).
set -euo pipefail
cd "$(dirname "$0")"

WINDOW=96
STRIDE=25

if [[ "${1:-}" == "--test" ]]; then
  for ds in etth1 etth2 ettm1 ettm2 illness weather; do
    echo "--- $ds (n_trials=2) ---"
    uv run python -u scripts/tune_forecasting.py --dataset "$ds" --n-trials 2 --window "$WINDOW" --stride "$STRIDE" --train-window-ratio 0.5
  done
  exit 0
fi

# Trial budgets used for the paper's search.
declare -A TRIALS=( [etth1]=65 [etth2]=65 [ettm1]=65 [ettm2]=65 [illness]=110 [weather]=200 )
for ds in etth1 etth2 ettm1 ettm2 illness weather; do
  echo "--- $ds (n_trials=${TRIALS[$ds]}) ---"
  uv run python -u scripts/tune_forecasting.py --dataset "$ds" --n-trials "${TRIALS[$ds]}" --window "$WINDOW" --stride "$STRIDE"
done
echo "Done. Best parameters in results/optuna/<dataset>_ivdfm_best.json"
