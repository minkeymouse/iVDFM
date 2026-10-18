#!/usr/bin/env bash
# Causal intervention on synthetic SCMs (base, regime, chain) with iVDFM.
# Usage: ./run_causal_intervention.sh [--test] [hydra overrides...]
#   --test  base SCM only, 1 simulation, single T and r (smoke test)
# Full run: every SCM at T in {100,200,500,1000} and r_fit in {2,3,4,5}; T=200, r=3 fills the main table.
set -euo pipefail
cd "$(dirname "$0")"
export HYDRA_FULL_ERROR=1

TEST_MODE=
ARGS=()
for a in "$@"; do
  if [ "$a" = "--test" ]; then TEST_MODE=1; else ARGS+=("$a"); fi
done

run_scm() {
  local scm="$1"; shift
  uv run python -u run.py experiment=causal_intervention experiment.scm_type="$scm" \
    experiment.save_synthetic_data=false "$@"
}

if [ -n "$TEST_MODE" ]; then
  echo "=== Test: base SCM, 1 simulation ==="
  run_scm base experiment.n_simulations=1 experiment.optuna.n_trials=0 "${ARGS[@]}"
  echo "=== Done (test): results/causal/base ==="
  exit 0
fi

for scm in base regime chain; do
  echo "=== ${scm} SCM (all T, all r) ==="
  run_scm "$scm" 'experiment.T_list=[100,200,500,1000]' 'experiment.r_fit_list=[2,3,4,5]' "${ARGS[@]}"
done
echo "=== Done: results/causal/{base,regime,chain} ==="
