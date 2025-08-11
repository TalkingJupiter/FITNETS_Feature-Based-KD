#!/bin/bash
#SBATCH --job-name=fitnets_distill
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --partition=h100
#SBATCH --signal=B:TERM@120

set -euo pipefail

# ----------------- SLURM VARS -----------------
GPU_INDEX=${1:-0}
RUN_LABEL="Run${GPU_INDEX}"
JOB_ID="${SLURM_JOB_ID:-manual}"
JOB_TS=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="logs/${JOB_ID}_${RUN_LABEL}"
mkdir -p "${OUTPUT_DIR}"

# Rename default SLURM logs to this run's directory
mv "logs/${JOB_ID}.out" "${OUTPUT_DIR}/distill_${JOB_ID}_${RUN_LABEL}.out" 2>/dev/null || true
mv "logs/${JOB_ID}.err" "${OUTPUT_DIR}/distill_${JOB_ID}_${RUN_LABEL}.err" 2>/dev/null || true

# ---------------- ENV SETUP -------------------
echo "[$(date)] Starting FitNets distillation on $(hostname), GPU ${GPU_INDEX}"
echo "[$(date)] Run label: ${RUN_LABEL}"

source ~/.bashrc
conda activate fitnets-env

export CUDA_VISIBLE_DEVICES=${GPU_INDEX}

# ---------------- MONITOR SETUP ----------------
MONITOR_JSON="${OUTPUT_DIR}/telemetry_${JOB_TS}_${RUN_LABEL}.jsonl"
MONITOR_LOG="${OUTPUT_DIR}/monitor_stdout_${JOB_TS}_${RUN_LABEL}.log"

echo "[$(date)] Starting telemetry monitor..."
python monitor/monitor.py \
  --gpu ${GPU_INDEX} \
  --interval 5 \
  --log_path "${MONITOR_JSON}" > "${MONITOR_LOG}" 2>&1 &

MONITOR_PID=$!
echo "${MONITOR_PID}" > "${OUTPUT_DIR}/monitor_${JOB_TS}_${RUN_LABEL}.pid"

sleep 3
if ! ps -p ${MONITOR_PID} > /dev/null; then
  echo "[$(date)] ERROR: monitor.py failed to start." >&2
  exit 1
fi

# ---------------- CLEANUP HANDLER ----------------
cleanup() {
  echo "[$(date)] Terminating monitor PID ${MONITOR_PID}..."
  kill ${MONITOR_PID} 2>/dev/null || true
  wait ${MONITOR_PID} 2>/dev/null || true
  echo "[$(date)] Monitor stopped."
}
trap cleanup INT TERM EXIT

# ---------------- TRAINING ----------------
TRAIN_LOG="${OUTPUT_DIR}/train_${JOB_TS}_${RUN_LABEL}.log"
echo "[$(date)] Launching distillation training..."

START_TS=$(date +%s)

srun -u python run_distill.py "${OUTPUT_DIR}" | tee "${TRAIN_LOG}"
STATUS=${PIPESTATUS[0]}

END_TS=$(date +%s)
DURATION=$((END_TS - START_TS))

echo "[$(date)] Training completed with status ${STATUS}"
echo "[$(date)] Duration: ${DURATION} seconds (~$((DURATION/60)) minutes)"
echo "[$(date)] Output dir: ${OUTPUT_DIR}"
echo "[$(date)] Telemetry: ${MONITOR_JSON}"
echo "[$(date)] Training log: ${TRAIN_LOG}"

exit ${STATUS}
