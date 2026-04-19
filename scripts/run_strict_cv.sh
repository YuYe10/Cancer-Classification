#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="./medical/bin/python3"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Error: Python interpreter not found at $PYTHON_BIN"
  exit 1
fi

echo "=========================================="
echo "Strict CV 1/3: RNA-only"
echo "Config: config/exp_rna_cv.yaml"
echo "=========================================="
PYTHONPATH=. "$PYTHON_BIN" experiments/run.py --config config/exp_rna_cv.yaml

echo

echo "=========================================="
echo "Strict CV 2/3: Concat"
echo "Config: config/exp_concat_cv.yaml"
echo "=========================================="
PYTHONPATH=. "$PYTHON_BIN" experiments/run.py --config config/exp_concat_cv.yaml

echo

echo "=========================================="
echo "Strict CV 3/4: Stacking"
echo "Config: config/exp_stacking.yaml"
echo "=========================================="
PYTHONPATH=. "$PYTHON_BIN" experiments/run.py --config config/exp_stacking.yaml

echo
echo "=========================================="
echo "Strict CV 4/4: Concat + Random Forest"
echo "Config: config/exp_concat_rf_cv.yaml"
echo "=========================================="
PYTHONPATH=. "$PYTHON_BIN" experiments/run.py --config config/exp_concat_rf_cv.yaml

echo
echo "Strict CV experiments finished."
