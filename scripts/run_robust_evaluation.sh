#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="./medical/bin/python3"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Error: Python interpreter not found at $PYTHON_BIN"
  exit 1
fi

run_exp() {
  local config="$1"
  local tag="$2"
  echo "=========================================="
  echo "Run config: $config"
  echo "Tag: $tag"
  echo "=========================================="
  PYTHONPATH=. "$PYTHON_BIN" experiments/run.py --config "$config" --tag "$tag"
  echo
}

TAG="robust_eval_$(date +%Y%m%d_%H%M%S)"

run_exp "config/exp_rna_cv.yaml" "$TAG"
run_exp "config/exp_concat_cv.yaml" "$TAG"
run_exp "config/exp_concat_rf_cv.yaml" "$TAG"
run_exp "config/exp_stacking.yaml" "$TAG"

LATEST_RNA=$(ls -t outputs/logs/exp_rna_cv_${TAG}_*.json 2>/dev/null | head -n1 || true)
LATEST_CONCAT=$(ls -t outputs/logs/exp_concat_cv_${TAG}_*.json 2>/dev/null | head -n1 || true)
LATEST_RF=$(ls -t outputs/logs/exp_concat_rf_cv_${TAG}_*.json 2>/dev/null | head -n1 || true)
LATEST_STACK=$(ls -t outputs/logs/exp_stacking_${TAG}_*.json 2>/dev/null | head -n1 || true)

FILES=()
for f in "$LATEST_RNA" "$LATEST_CONCAT" "$LATEST_RF" "$LATEST_STACK"; do
  if [[ -n "$f" ]]; then
    FILES+=("$f")
  fi
done

if [[ ${#FILES[@]} -ge 2 ]]; then
  echo "Running statistical comparison on latest CV results..."
  PYTHONPATH=. "$PYTHON_BIN" scripts/statistical_evaluation.py \
    --files "${FILES[@]}" \
    --metric accuracy \
    --out-md outputs/logs/statistical_evaluation.md
  echo "Saved: outputs/logs/statistical_evaluation.md"
else
  echo "Skip statistical comparison: not enough CV result files found."
fi

echo "Robust evaluation finished. Tag: $TAG"
