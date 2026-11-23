#!/bin/bash
#SBATCH --job-name=pp_probe
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --output=/scratch/pgopalan/project_Praveen/logs/%x_%j.log

# Learnability probe: can bul_100m be taught to copy text from its prompt?
# Trains twice on the same copy task (LoRA + stop token, then full fine-tune)
# and scores each on 200 held-out copy rows.

module load gnu12/12.3.0 python/3.11.7-cx
cd /scratch/pgopalan/project_Praveen
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
echo "node: $(hostname)  job: $SLURM_JOB_ID  started: $(date)"

venv/bin/python scripts/make_copy_probe.py \
    --source data/splits/train.json --out-dir data/probe_copy

echo "===== LoRA + stop token ====="
venv/bin/python scripts/train_lora.py --model models/bul_100m \
    --train data/probe_copy/train.json --val data/probe_copy/val.json \
    --train-stop-token --epochs 5 --out adapters/probe_copy_lora
venv/bin/python scripts/eval_test.py --model models/bul_100m \
    --adapter adapters/probe_copy_lora/best \
    --test data/probe_copy/test.json --out adapters/probe_copy_lora/test_eval.json

echo "===== full fine-tune ====="
venv/bin/python scripts/train_lora.py --model models/bul_100m \
    --train data/probe_copy/train.json --val data/probe_copy/val.json \
    --full-finetune --epochs 5 --out adapters/probe_copy_fullft
venv/bin/python scripts/eval_test.py --model models/bul_100m \
    --adapter adapters/probe_copy_fullft/best \
    --test data/probe_copy/test.json --out adapters/probe_copy_fullft/test_eval.json

echo "finished: $(date)  exit: $?"
