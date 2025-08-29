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
#SBATCH --array=0-3
#SBATCH --signal=B:TERM@120

set -euo pipefail

# ------------- BASICS -------------
mkdir -p logs

JOB_ID="${SLURM_JOB_ID:-manual}"
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"      # 0..3 from --array
RUN_LABEL="Run${TASK_ID}"
JOB_TS=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="logs/${JOB_ID}_${RUN_LABEL}"
mkdir -p "${OUTPUT_DIR}"

# Try to move Slurm's default logs into our run dir (may be a no-op early on)
mv "logs/${JOB_ID}.out" "${OUTPUT_DIR}/slurm_${JOB_ID}_${RUN_LABEL}.out" 2>/dev/null || true
mv "logs/${JOB_ID}.err" "${OUTPUT_DIR}/slurm_${JOB_ID}_${RUN_LABEL}.err" 2>/dev/null || true

echo "[$(date)] Host: $(hostname)"
echo "[$(date)] Job: ${JOB_ID}  Task: ${TASK_ID}  Output: ${OUTPUT_DIR}"

# ------------- ENV -------------
source ~/.bashrc
conda activate fitnets

# Respect Slurm's GPU assignment; avoid overriding if Slurm already set it
if [[ -n "${SLURM_JOB_GPUS:-}" ]]; then
  # pick the first GPU from the list Slurm granted
  export CUDA_VISIBLE_DEVICES="$(echo "${SLURM_JOB_GPUS}" | sed 's/,.*//')"
fi
echo "[$(date)] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# ------------- DATA CHECK -------------
DATA_DIR="data/amazon_polarity_tok"
if [[ ! -d "${DATA_DIR}" ]]; then
  echo "[$(date)] ERROR: '${DATA_DIR}' not found. Export the dataset first with the loader." >&2
  exit 2
fi

# ------------- EXTERNAL MONITOR -------------
MONITOR_JSON="${OUTPUT_DIR}/telemetry_${JOB_TS}_${RUN_LABEL}.jsonl"
MONITOR_LOG="${OUTPUT_DIR}/monitor_stdout_${JOB_TS}_${RUN_LABEL}.log"

echo "[$(date)] Starting telemetry monitor..."
python monitor/stream_monitor.py \
  --gpu 0 \
  --interval 1 \
  --outfile "${MONITOR_JSON}" > "${MONITOR_LOG}" 2>&1 &

MONITOR_PID=$!
echo "${MONITOR_PID}" > "${OUTPUT_DIR}/monitor_${JOB_TS}_${RUN_LABEL}.pid"
# confirm it's alive
sleep 1
kill -0 "${MONITOR_PID}" 2>/dev/null || { echo "[$(date)] ERROR: monitor failed. See ${MONITOR_LOG}"; exit 3; }


cleanup() {
  echo "[$(date)] Stopping monitor PID ${MONITOR_PID}..."
  kill "${MONITOR_PID}" 2>/dev/null || true
  wait "${MONITOR_PID}" 2>/dev/null || true
  echo "[$(date)] Monitor stopped."
}
trap cleanup INT TERM EXIT

# ------------- TRAINING -------------
TRAIN_LOG="${OUTPUT_DIR}/train_${JOB_TS}_${RUN_LABEL}.log"
echo "[$(date)] Launching feature-based KD training..."
START_TS=$(date +%s)

# Use srun so Slurm binds resources correctly
# (unbuffered -u prevents log buffering)
set +e
srun -u python run_distill.py \
  --data-dir "${DATA_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --teacher-name bert-base-uncased \
  --student-name distilbert-base-uncased \
  --batch-size 32 \
  --num-workers 4 \
  --epochs 3 \
  --kd-T 2.0 \
  --kd-alpha 0.5 \
  --feat-beta 1.0 \
  --t-layer -2 \
  --s-layer -2 \
  --feat-pool cls \
  --feat-normalize |& tee "${TRAIN_LOG}"
STATUS=${PIPESTATUS[0]}
set -e

END_TS=$(date +%s)
DURATION=$((END_TS - START_TS))

echo "[$(date)] Training exit status: ${STATUS}"
echo "[$(date)] Duration: ${DURATION}s (~$((DURATION/60)) min)"
echo "[$(date)] Output dir: ${OUTPUT_DIR}"
echo "[$(date)] Telemetry:  ${MONITOR_JSON}"
echo "[$(date)] Train log:  ${TRAIN_LOG}"

exit "${STATUS}"
