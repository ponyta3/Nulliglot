#!/bin/bash
#SBATCH --job-name=pp_full
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/pgopalan/project_Praveen/logs/%x_%j.log

module load gnu12/12.3.0 python/3.11.7-cx
cd /scratch/pgopalan/project_Praveen
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
echo "node: $(hostname)  job: $SLURM_JOB_ID  started: $(date)"
nvidia-smi
venv/bin/python scripts/train_lora.py \
    --model models/bul_100m \
    --train data/splits/train.json --val data/splits/val.json \
    --train-stop-token \
    --epochs 3 \
    --out adapters/bul_instruct
echo "finished: $(date)  exit: $?"
