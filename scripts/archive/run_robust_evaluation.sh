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
run_exp "config/exp_mofa.yaml" "$TAG"
run_exp "config/exp_stacking.yaml" "$TAG"

LATEST_RNA=$(ls -t outputs/logs/exp_rna_cv_${TAG}_*.json 2>/dev/null | head -n1 || true)
LATEST_CONCAT=$(ls -t outputs/logs/exp_concat_cv_${TAG}_*.json 2>/dev/null | head -n1 || true)
LATEST_MOFA=$(ls -t outputs/logs/exp_mofa_${TAG}_*.json 2>/dev/null | head -n1 || true)
LATEST_STACK=$(ls -t outputs/logs/exp_stacking_${TAG}_*.json 2>/dev/null | head -n1 || true)

FILES=()
for f in "$LATEST_RNA" "$LATEST_CONCAT" "$LATEST_MOFA" "$LATEST_STACK"; do
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
  PYTHONPATH=. "$PYTHON_BIN" scripts/statistical_evaluation.py \
    --files "${FILES[@]}" \
    --metric balanced_accuracy \
    --out-md outputs/logs/statistical_evaluation_balanced_accuracy.md
  PYTHONPATH=. "$PYTHON_BIN" scripts/statistical_evaluation.py \
    --files "${FILES[@]}" \
    --metric macro_f1 \
    --out-md outputs/logs/statistical_evaluation_macro_f1.md

  echo "Generating publication plots..."
  PYTHONPATH=. "$PYTHON_BIN" scripts/generate_statistical_plots.py \
    --files "${FILES[@]}" \
    --metric accuracy \
    --out-dir outputs/figures
  PYTHONPATH=. "$PYTHON_BIN" scripts/generate_statistical_plots.py \
    --files "${FILES[@]}" \
    --metric balanced_accuracy \
    --out-dir outputs/figures
  PYTHONPATH=. "$PYTHON_BIN" scripts/generate_statistical_plots.py \
    --files "${FILES[@]}" \
    --metric macro_f1 \
    --out-dir outputs/figures

  echo "Generating comparison tables and class-level error analysis..."
  PYTHONPATH=. "$PYTHON_BIN" scripts/generate_comparison_tables.py \
    --files "${FILES[@]}" \
    --metrics accuracy balanced_accuracy macro_f1 \
    --stats-md outputs/logs/statistical_evaluation.md \
    --baseline-method rna \
    --out-latex outputs/logs/method_comparison_table.tex \
    --out-md outputs/logs/method_comparison_table.md
  PYTHONPATH=. "$PYTHON_BIN" scripts/generate_class_error_analysis.py \
    --files "${FILES[@]}" \
    --out-csv outputs/logs/class_error_analysis.csv \
    --out-md outputs/logs/class_error_analysis.md

  echo "Saved: outputs/logs/statistical_evaluation.md"
  echo "Saved: outputs/logs/statistical_evaluation_balanced_accuracy.md"
  echo "Saved: outputs/logs/statistical_evaluation_macro_f1.md"
  echo "Saved: outputs/logs/method_comparison_table.tex"
  echo "Saved: outputs/logs/method_comparison_table.md"
  echo "Saved: outputs/logs/class_error_analysis.csv"
  echo "Saved: outputs/logs/class_error_analysis.md"
else
  echo "Skip statistical comparison: not enough CV result files found."
fi

echo "Robust evaluation finished. Tag: $TAG"
