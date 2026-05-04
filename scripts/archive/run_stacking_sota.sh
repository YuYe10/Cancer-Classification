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
echo "Stacking SOTA 1/3: XGBoost + XGBoost"
echo "Config: config/exp_stacking_sota_cv.yaml"
echo "=========================================="
PYTHONPATH=. "$PYTHON_BIN" experiments/run.py --config config/exp_stacking_sota_cv.yaml --tag stacking_sota

echo

echo "=========================================="
echo "Stacking SOTA 2/3: RF + XGBoost"
echo "Config: config/exp_stacking_hybrid_cv.yaml"
echo "=========================================="
PYTHONPATH=. "$PYTHON_BIN" experiments/run.py --config config/exp_stacking_hybrid_cv.yaml --tag stacking_sota

echo

echo "=========================================="
echo "Stacking SOTA 3/3: Stable XGBoost + XGBoost"
echo "Config: config/exp_stacking_sota_stable_cv.yaml"
echo "=========================================="
PYTHONPATH=. "$PYTHON_BIN" experiments/run.py --config config/exp_stacking_sota_stable_cv.yaml --tag stacking_sota

echo

echo "Stacking SOTA experiments finished."
