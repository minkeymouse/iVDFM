#!/usr/bin/env bash
# Factor recovery (Tables 1-2): 10 simulations, dynamic + static suites, all five models.
# Per-model hyperparameters are fixed in config/experiment/factor_recovery.yaml (suite_model_overrides).
# Usage: bash run_factor_recovery.sh [hydra overrides...]
set -euo pipefail
cd "$(dirname "$0")"
export HYDRA_FULL_ERROR=1

uv run python -u run.py \
  experiment=factor_recovery \
  dataset=synthetic_dynamic \
  model=ivdfm \
  experiment.n_simulations=10 \
  experiment.suites='[dynamic,static]' \
  experiment.models='[dfm,ddfm,vae,ivae,ivdfm]' \
  experiment.output_dir=./results/factor_recovery \
  experiment.T=200 \
  experiment.N=20 \
  experiment.r=5

echo "Factor recovery finished. Results in results/factor_recovery/{dynamic,static}/{model}/"
