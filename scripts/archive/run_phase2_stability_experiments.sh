#!/bin/bash
# 启动特征维度与 repeats 收敛实验（并行）

export PYTHONPATH=.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "========== Starting Phase 2: Stability & Sensitivity Experiments =========="
echo "Experiment timestamp: $(date '+%Y%m%d_%H%M%S')"
echo ""

# 创建日志目录
mkdir -p outputs/logs/phase2_experiments

# 启动特征维度实验（5 个并行）
echo "[$(date '+%H:%M:%S')] Starting feature dimension search experiments..."
for dim in 100 300 500 1000 1500; do
  echo "  Launching: dim=$dim" &
  ./medical/bin/python3 experiments/run.py \
    --config config/exp_concat_cv_dim${dim}.yaml \
    --tag hp_dim_search_${dim} \
    > outputs/logs/phase2_experiments/dim${dim}.log 2>&1 &
  sleep 2
done

# 启动 repeats 收敛实验（2 个并行，repeats=10,15）
echo "[$(date '+%H:%M:%S')] Starting repeats convergence experiments..."
for repeat in 10 15; do
  echo "  Launching: repeats=$repeat" &
  ./medical/bin/python3 experiments/run.py \
    --config config/exp_concat_cv_repeat${repeat}.yaml \
    --tag hp_repeat_search_${repeat} \
    > outputs/logs/phase2_experiments/repeat${repeat}.log 2>&1 &
  sleep 2
done

echo ""
echo "[$(date '+%H:%M:%S')] All 7 experiments launched in background."
echo "Monitor progress with:"
echo "  tail -f outputs/logs/phase2_experiments/*.log"
echo ""
echo "Results will be appended to: outputs/logs/summary_v2.csv"
echo "========== Background Tasks Running =========="

wait
echo "[$(date '+%H:%M:%S')] All experiments completed!"
echo ""
echo "========== Phase 2 Summary =========="
./medical/bin/python3 scripts/summarize_results.py \
  --summary-csv outputs/logs/summary_v2.csv \
  --mode cv \
  --metric accuracy_mean \
  --group-by exp \
  --top-k 20 \
  --out-md outputs/logs/phase2_summary.md

echo "Summary saved to: outputs/logs/phase2_summary.md"
