#!/bin/bash
#SBATCH --job-name=pp_eval
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/pgopalan/project_Praveen/logs/%x_%j.log

module load gnu12/12.3.0 python/3.11.7-cx
cd /scratch/pgopalan/project_Praveen
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
echo "node: $(hostname)  job: $SLURM_JOB_ID  started: $(date)"
venv/bin/python scripts/eval_test.py \
    --model models/bul_100m \
    --adapter adapters/bul_instruct/best \
    --test data/splits/test.json \
    --out adapters/bul_instruct/test_eval.json
echo "finished: $(date)  exit: $?"
