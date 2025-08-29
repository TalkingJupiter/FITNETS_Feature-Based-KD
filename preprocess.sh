#!/bin/bash
#SBATCH --job-name=FITNETS-preprocess
#SBATCH --output=logs/preproc_%j.out
#SBATCH --error=logs/preproc_%j.err
#SBATCH --partition=zen4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

mkdir -p logs data

source "$(conda info --base)/etc/profile.d/conda.sh"
set +u
conda activate fitnets
set -u

export TOKENIZERS_PARALLELISM=true

echo "[$(date)] Step 1/1 Ensure the Amazon Polarity file exits"
if [ ! -f "data/amazon_polarity_tok" ]; then
  echo "[$(date)] Data does not exists. Starting loading process..."
  python dataset/amazon_polarity_loader.py \
    --model-name bert-base-uncased \
    --dataset amazon_polarity \
    --batch-size 32 \
    --max-length 256 \
    --out-dir data/amazon_polarity_tok \
    --save-format hf
else
    echo "Amazon Polarity dataset exists, skipping..."
fi

echo "[$(date)] Preporcessing done"