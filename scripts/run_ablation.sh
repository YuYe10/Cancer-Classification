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
echo "Ablation 1/3: Remove Methylation (RNA-only)"
echo "Config: config/ablation_no_meth.yaml"
echo "=========================================="
PYTHONPATH=. "$PYTHON_BIN" experiments/run.py --config config/ablation_no_meth.yaml

echo
echo "=========================================="
echo "Ablation 2/3: Remove RNA (Meth-only)"
echo "Config: config/ablation_no_rna.yaml"
echo "=========================================="
PYTHONPATH=. "$PYTHON_BIN" experiments/run.py --config config/ablation_no_rna.yaml

echo
echo "=========================================="
echo "Ablation 3/3: No Feature Selection"
echo "Config: config/ablation_no_fs.yaml"
echo "=========================================="
PYTHONPATH=. "$PYTHON_BIN" experiments/run.py --config config/ablation_no_fs.yaml

echo
echo "All ablation experiments finished."
