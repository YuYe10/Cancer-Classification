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
echo "Leaderboard 1/5: All CV runs"
echo "=========================================="
PYTHONPATH=. "$PYTHON_BIN" scripts/summarize_results.py --mode cv --metric accuracy_mean --top-k 30 --group-by exp --out-md outputs/logs/leaderboard_cv.md --out-csv outputs/logs/leaderboard_cv.csv --out-grouped-md outputs/logs/leaderboard_cv_grouped_by_exp.md

echo

echo "=========================================="
echo "Leaderboard 2/5: Stacking SOTA tag"
echo "=========================================="
PYTHONPATH=. "$PYTHON_BIN" scripts/summarize_results.py --mode cv --tag stacking_sota --metric accuracy_mean --top-k 30 --group-by config --out-md outputs/logs/leaderboard_stacking_sota.md --out-csv outputs/logs/leaderboard_stacking_sota.csv --out-grouped-md outputs/logs/leaderboard_stacking_sota_grouped.md

echo

echo "=========================================="
echo "Leaderboard 3/5: Overall holdout+CV"
echo "=========================================="
PYTHONPATH=. "$PYTHON_BIN" scripts/summarize_results.py --top-k 30 --group-by mode --out-md outputs/logs/leaderboard_all.md --out-csv outputs/logs/leaderboard_all.csv --out-grouped-md outputs/logs/leaderboard_all_grouped_by_mode.md

echo

echo "=========================================="
echo "Leaderboard 4/5: Best configs by experiment"
echo "=========================================="
PYTHONPATH=. "$PYTHON_BIN" scripts/export_best_configs.py --group-by exp --metric accuracy_mean --out-md outputs/logs/best_configs_by_exp.md --out-csv outputs/logs/best_configs_by_exp.csv

echo

echo "=========================================="
echo "Leaderboard 5/5: Target monitor (ACC >= 0.95)"
echo "=========================================="
PYTHONPATH=. "$PYTHON_BIN" scripts/monitor_target.py --threshold 0.95 --out-md outputs/logs/target_hits_095.md --out-csv outputs/logs/target_hits_095.csv

echo

echo "Leaderboard generation finished."
