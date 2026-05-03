#!/bin/bash
#==============================================================================
# Batch experiment runner — traverses config subdirectories recursively.
#
# Usage:
#   ./run_all.sh                  #  run all experiments
#   ./run_all.sh rna              #  run only rna/ experiments
#   ./run_all.sh stacking/cv      #  run only stacking/cv/ experiments
#   ./run_all.sh ablation         #  run only ablation/ experiments
#
# Config path convention:
#   config/<category>/<name>.yaml   →  PYTHONPATH=. experiments/run.py --config config/<category>/<name>.yaml
#==============================================================================
set -euo pipefail

GROUP="${1:-}"

if [ -n "$GROUP" ]; then
    echo "=== Running group: config/${GROUP}/ ==="
    CONFIGS=$(find "config/${GROUP}" -name "*.yaml" -not -path '*/shared/*' | sort)
else
    echo "=== Running ALL experiments ==="
    CONFIGS=$(find config -name "*.yaml" -not -path '*/shared/*' -not -path 'config/*.yaml' | sort)
fi

if [ -z "$CONFIGS" ]; then
    echo "No config files found."
    exit 0
fi

TOTAL=$(echo "$CONFIGS" | wc -l)
COUNT=0

for cfg in $CONFIGS; do
    COUNT=$((COUNT + 1))
    echo ""
    echo "=== [${COUNT}/${TOTAL}] Running $cfg ==="
    PYTHONPATH=. ./medical/bin/python3 experiments/run.py --config "$cfg"
done

echo ""
echo "=== All ${COUNT} experiment(s) completed ==="
