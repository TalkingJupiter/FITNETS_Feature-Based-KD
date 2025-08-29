# FITNETS Feature-Based Knowledge Distillation on Amazon Polarity  

---

## 1. PURPOSE
This document describes the design, configuration, operation, and monitoring procedures for a feature-based knowledge distillation (KD) workflow—augmented with response-based KD—targeting the Amazon Polarity sentiment classification task. It provides implementers and operators with standardized instructions for environment setup, dataset handling, model training, telemetry collection, and troubleshooting in a High-Performance Computing (HPC) environment.

---

## 2. SCOPE
- **Systems:** REPACSS HPC cluster with Slurm scheduler and NVIDIA GPUs (e.g., H100 NVL).
- **Models:** Teacher—`bert-base-uncased`; Student—`distilbert-base-uncased`.
- **Data:** Tokenized Amazon Polarity (two-class sentiment).

---

## 3. BACKGROUND
Knowledge Distillation transfers capability from a large **Teacher** model to a smaller **Student** model. This project employs:
- **Feature-Based KD (FitNets-style):** Aligns internal hidden representations between teacher and student via a connector module.
- **Response-Based KD:** Minimizes divergence between teacher and student logits under temperature scaling while retaining cross-entropy with ground-truth labels.

---

## 4. SYSTEM OVERVIEW

### 4.1 Repository Structure (reference)
```
FITNETS_Feature-Based-KD/
├─ dataset/
│  └─ amazon_polarity_loader.py
├─ monitor/
│  ├─ monitor.py            # internal CSV monitor (context manager; .log_once())
│  └─ stream_monitor.py     # external JSONL streamer (standalone process)
├─ scripts/
│  ├─ preprocess.sh         # optional CPU pre-processing job
│  └─ fitnets_distill.sh    # GPU training job (starts external monitor)
├─ run_amazon_polarity_feature_kd.py  # training entrypoint
├─ configs/                 # optional configuration files
├─ logs/                    # Slurm and runtime outputs
└─ serialization_dir/       # model checkpoints (if enabled)
```

### 4.2 Key Components
- **Training Driver:** `run_distill.py`
- **Telemetry:** 
  - **External monitor:** `monitor/stream_monitor.py` → JSONL (continuous, fsync each record).
  <!-- - **Internal monitor (optional):** `monitor/monitor.py` → CSV (logged at defined steps). -->

---

<!-- ## 5. ROLES & RESPONSIBILITIES
- **Operator:** Submits/monitors jobs, verifies telemetry, handles errors.
- **Research Lead:** Sets hyperparameters, validates accuracy, curates results.
- **System Administrator:** Ensures GPU/driver availability, storage permissions, and module stacks.
- **Compliance Officer:** Verifies adherence to data handling and retention requirements. -->

---

## 5. PREREQUISITES

### 5.1 Software
- Conda/Miniforge (Python 3.10)
- PyTorch w/ CUDA (cluster-appropriate wheel)
- `transformers`, `datasets`, `pynvml`, `psutil`
- Slurm (`sbatch`, `srun`)

### 5.2 Hardware
- NVIDIA GPU allocated via Slurm (`--gres=gpu:1` or more).

---

## 6. INSTALLATION & ENVIRONMENT

```bash
# Load or locate conda
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# Create env
conda create -y -n fitnets python=3.10
conda activate fitnets

# Core deps
python -m pip install --upgrade pip
pip install "torch==2.*" --index-url https://download.pytorch.org/whl/cu121  # or CPU URL
pip install transformers datasets pynvml psutil

# (Optional) Repo-specific requirements
pip install -r requirements.txt
```

---

## 7. DATA MANAGEMENT

### 7.1 Dataset Export (one-time)
```bash
python dataset/amazon_polarity_loader.py \
  --model-name bert-base-uncased \   
  --dataset amazon_polarity \
  --batch-size 32 \
  --max-length 256 \ 
  --out-dir data/amazon_polarity_tok --save-format hf
```

### 7.2 On-Load Normalization
The training script normalizes the label column to `labels` (renames from `label` if needed) and ensures Torch tensors for `input_ids`, `attention_mask`, and `labels`.

---

## 8. TRAINING LOGIC

### 8.1 Distillation Objectives
- **Response KD:** `CE(student, y) * alpha + KL(student_T, teacher_T) * (1 - alpha)`.
- **Feature KD:** `MSE(Connector(student_features), teacher_features)`, optional L2 normalization.

### 8.2 Key Switches
- `--feat-beta`: weight of feature loss (set `0` to disable feature KD).
- `--kd-alpha`: weight on CE (set `1.0` to disable KL; set `0.0` to rely on KL only).
- `--t-layer`, `--s-layer`: hidden state indices (e.g., `-2` for penultimate).
- `--feat-pool`: `cls` or `mean` pooling.
- `--feat-normalize`: L2 normalize features before MSE.

---

## 9. MONITORING THE PROCESS (PRIMARY SECTION)

### 9.1 External Telemetry (Recommended for Operations)
**File:** `monitor/stream_monitor.py`  
**Output:** JSON Lines (`.jsonl`) with one record per interval; fsync ensures durability.

**Start (Slurm job snippet):**
```bash
MONITOR_JSON="${OUTPUT_DIR}/telemetry_${JOB_TS}_${RUN_LABEL}.jsonl"
python monitor/stream_monitor.py   --gpu 0   --interval 1   --outfile "${MONITOR_JSON}" > "${OUTPUT_DIR}/monitor_stdout.log" 2>&1 &
MONITOR_PID=$!
# Verify
sleep 1 && kill -0 "${MONITOR_PID}" || { echo "Monitor failed"; exit 3; }
```

**Graceful Stop (ensures flush):**
```bash
cleanup() {
  echo "Stopping monitor PID ${MONITOR_PID}..."
  kill -INT "${MONITOR_PID}" 2>/dev/null || true  # SIGINT → graceful
  sleep 1
  kill "${MONITOR_PID}" 2>/dev/null || true       # fallback TERM
  wait "${MONITOR_PID}" 2>/dev/null || true
}
trap cleanup INT TERM EXIT
```

**Quick Inspection:**
```bash
tail -n 3 "${MONITOR_JSON}"
jq . "${MONITOR_JSON}" | head
```

**Fields Recorded (per sample):**
- `timestamp`, `gpu_index`, `gpu_name`
- `power_watts`, `memory_used_MB`
- `gpu_utilization_percent`, `memory_utilization_percent`, `temperature_C`
- `cpu_utilization_percent`

### 9.2 Internal Telemetry (Research/Per-Step Snapshots)
**File:** `monitor/monitor.py` (context manager)  
**Usage:** Training script calls `mon.log_once()` at start, at step cadence, and after validation, writing CSV to `--output-dir/distill_power.csv`. If the internal monitor isn’t present, a no-op fallback keeps training running without CSV.

### 9.3 Energy Estimation (Post-Run)
Integrators may approximate energy (Wh) by discrete integration over `power_watts` vs. time:
- For JSONL: parse timestamps → seconds, then trapezoidal rule.
- For CSV: use `elapsed_sec` if provided.

---

## 10. SLURM EXECUTION

### 10.1 GPU Job Template (`scripts/fitnets_distill.sh`)
```bash
#!/usr/bin/env bash
#SBATCH --job-name=fitnets_distill
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --partition=h100
#SBATCH --array=0-3
#SBATCH --signal=B:TERM@120
set -euo pipefail

mkdir -p logs
JOB_ID="${SLURM_JOB_ID:-manual}"; TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
RUN_LABEL="Run${TASK_ID}"; JOB_TS=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="logs/${JOB_ID}_${RUN_LABEL}"; mkdir -p "${OUTPUT_DIR}"

source ~/.bashrc; conda activate fitnets
if [[ -n "${SLURM_JOB_GPUS:-}" ]]; then export CUDA_VISIBLE_DEVICES="$(echo "${SLURM_JOB_GPUS}" | sed 's/,.*//')"; fi
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# Start external monitor (visible index 0 when one GPU is granted)
MONITOR_JSON="${OUTPUT_DIR}/telemetry_${JOB_TS}_${RUN_LABEL}.jsonl"
python monitor/stream_monitor.py --gpu 0 --interval 1 --outfile "${MONITOR_JSON}"   > "${OUTPUT_DIR}/monitor_stdout_${JOB_TS}.log" 2>&1 &
MONITOR_PID=$!
sleep 1; kill -0 "${MONITOR_PID}" || { echo "Monitor failed"; exit 3; }

cleanup() {
  echo "Stopping monitor PID ${MONITOR_PID}..."
  kill -INT "${MONITOR_PID}" 2>/dev/null || true
  sleep 1; kill "${MONITOR_PID}" 2>/dev/null || true
  wait "${MONITOR_PID}" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

DATA_DIR="data/amazon_polarity_tok"
[[ -d "${DATA_DIR}" ]] || { echo "Data dir missing: ${DATA_DIR}"; exit 2; }

srun -u python run_amazon_polarity_feature_kd.py   --data-dir "${DATA_DIR}"   --output-dir "${OUTPUT_DIR}"   --teacher-name bert-base-uncased   --student-name distilbert-base-uncased   --batch-size 32 --num-workers 4 --epochs 3   --kd-T 2.0 --kd-alpha 0.5   --feat-beta 1.0 --t-layer -2 --s-layer -2   --feat-pool cls --feat-normalize   --telemetry-every 50   |& tee "${OUTPUT_DIR}/train_${JOB_TS}.log"
```

---

## 11. How to Run on HPC (Automatically)
If you submit `submit_both.sh` using sbatch on your HPC it should automatically check if you have the right conda env and submit the jobs according to your system needs.
```bash
sbatch submit_both.sh
```

## 11. SECURITY, PRIVACY, AND COMPLIANCE
- **Data:** Amazon Polarity is public; do not intermingle with Controlled Unclassified Information (CUI) or PII.
- **Retention:** Retain logs/telemetry only as long as necessary for reproducibility and reporting.
- **Software Licenses:** Respect licenses for pretrained models, datasets, and dependencies.

---

## 12. LOGGING & ARTIFACTS
- **Training Logs:** `logs/<JOB>_<RUN>/train_*.log`
- **Telemetry:** `telemetry_*.jsonl` (external), `distill_power.csv` (internal, optional)
- **Checkpoints:** If enabled, store under `serialization_dir/` with run identifiers.
- **Slurm Logs:** `logs/%x_%A_%a.out|.err`

---

## 13. TROUBLESHOOTING
- **`KeyError: 'label'`:** Normalize dataset column to `labels` (the training script already does this).
- **`NVMLError_InvalidArgument`:** Use visible GPU index `0` when only one GPU is granted by Slurm; don’t pass physical IDs.
- **Empty Telemetry File:**
  - Ensure **external** streamer is used (JSONL, fsync each record), or
  - Send **SIGINT** on cleanup to allow graceful CSV write for the **internal** monitor, and/or
  - Reduce `--interval` to capture early samples.
- **CRLF in Shell Scripts:** Run `sed -i 's/\r$//' script.sh`.
- **Conda Not Found:** Load module or set `CONDA_BASE` appropriately.

---

## 14. PERFORMANCE & REPRODUCIBILITY
- Fix seeds in scripts (optional) for deterministic behavior where feasible.
- Log all hyperparameters in the run directory.
- Prefer pinned package versions for repeatability.

---

## 15. CHANGE HISTORY
- **v1.0:** Initial release—feature KD + response KD; dual telemetry pathways; Slurm templates; dataset normalization.

---

## 16. POINTS OF CONTACT
- **Author:** _Batuhan Sencer, Texas Tech, batuhan.sencer@ttu.edu_  

---
