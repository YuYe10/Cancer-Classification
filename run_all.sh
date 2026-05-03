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
#   config/<category>/<name>.yaml   →  PYTHONPATH=. python3 experiments/run.py --config config/<category>/<name>.yaml
#==============================================================================
set -euo pipefail

# Determine script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
CONFIG_DIR="${PROJECT_ROOT}/config"

GROUP="${1:-}"

#==============================================================================
# Helper: Print error message and exit
#==============================================================================
error_exit() {
    echo "[ERROR] $1" >&2
    exit 1
}

#==============================================================================
# Step 1: Validate config directory exists
#==============================================================================
if [ ! -d "${CONFIG_DIR}" ]; then
    error_exit "Config directory not found: ${CONFIG_DIR}\nPlease ensure the config/ directory exists at the project root."
fi

#==============================================================================
# Step 2: Validate Python interpreter
#==============================================================================
PYTHON_CMD=""
if [ -x "${PROJECT_ROOT}/medical/bin/python3" ]; then
    PYTHON_CMD="${PROJECT_ROOT}/medical/bin/python3"
elif command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
else
    error_exit "Python3 interpreter not found.\nPlease install Python3 or ensure medical/bin/python3 exists."
fi

# Verify run.py exists
RUN_PY="${PROJECT_ROOT}/experiments/run.py"
if [ ! -f "${RUN_PY}" ]; then
    error_exit "Experiment runner not found: ${RUN_PY}\nPlease ensure experiments/run.py exists."
fi

#==============================================================================
# Step 3: Collect config files
#==============================================================================
if [ -n "$GROUP" ]; then
    GROUP_DIR="${CONFIG_DIR}/${GROUP}"
    if [ ! -d "${GROUP_DIR}" ]; then
        error_exit "Group directory not found: ${GROUP_DIR}\nAvailable groups: $(ls -1 "${CONFIG_DIR}" | grep -v '^shared$' | tr '\n' ' ')"
    fi
    echo "=== Running group: config/${GROUP}/ ==="
    CONFIGS=$(find "${GROUP_DIR}" -name "*.yaml" -not -path '*/shared/*' | sort)
else
    echo "=== Running ALL experiments ==="
    # Include both top-level configs (base.yaml) and nested configs
    CONFIGS=$(find "${CONFIG_DIR}" -name "*.yaml" -not -path '*/shared/*' | sort)
fi

#==============================================================================
# Step 4: Validate config files found
#==============================================================================
if [ -z "${CONFIGS}" ]; then
    echo "[WARNING] No config files found."
    if [ -n "$GROUP" ]; then
        echo "  Group directory: config/${GROUP}/"
        echo "  Please ensure .yaml files exist in this directory."
    else
        echo "  Config directory: ${CONFIG_DIR}"
        echo "  Please ensure .yaml files exist under config/ (excluding shared/)."
    fi
    echo ""
    echo "Directory structure:"
    find "${CONFIG_DIR}" -name "*.yaml" | head -20 || true
    exit 1
fi

#==============================================================================
# Step 5: Display summary and confirm
#==============================================================================
TOTAL=$(echo "${CONFIGS}" | wc -l)
echo "Found ${TOTAL} configuration file(s):"
echo "${CONFIGS}" | while read -r cfg; do
    echo "  - ${cfg}"
done
echo ""

#==============================================================================
# Step 6: Run experiments
#==============================================================================
COUNT=0
SUCCESS=0
FAILED=0
FAILED_CONFIGS=""

for cfg in ${CONFIGS}; do
    COUNT=$((COUNT + 1))
    echo ""
    echo "=== [${COUNT}/${TOTAL}] Running ${cfg} ==="
    
    if [ ! -f "${cfg}" ]; then
        echo "[WARNING] Config file not accessible: ${cfg}"
        FAILED=$((FAILED + 1))
        FAILED_CONFIGS="${FAILED_CONFIGS}\n  - ${cfg} (not accessible)"
        continue
    fi
    
    # Run experiment with error handling
    if PYTHONPATH="${PROJECT_ROOT}" "${PYTHON_CMD}" "${RUN_PY}" --config "${cfg}"; then
        SUCCESS=$((SUCCESS + 1))
        echo "[SUCCESS] ${cfg}"
    else
        FAILED=$((FAILED + 1))
        FAILED_CONFIGS="${FAILED_CONFIGS}\n  - ${cfg}"
        echo "[FAILED] ${cfg} (exit code: $?)"
        # Continue with next experiment instead of stopping
        set +e
    fi
done

#==============================================================================
# Step 7: Print summary
#==============================================================================
echo ""
echo "========================================"
echo "=== Experiment Summary ==="
echo "========================================"
echo "Total:   ${COUNT}"
echo "Success: ${SUCCESS}"
echo "Failed:  ${FAILED}"
echo "========================================"

if [ ${FAILED} -gt 0 ]; then
    echo ""
    echo "Failed configurations:${FAILED_CONFIGS}"
    exit 1
else
    echo ""
    echo "=== All ${COUNT} experiment(s) completed successfully ==="
    exit 0
fi
