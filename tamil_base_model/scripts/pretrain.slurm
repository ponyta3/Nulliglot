#!/bin/bash
#SBATCH --job-name=ta_pretrain
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-12:00:00
#SBATCH --output=/scratch/pgopalan/tamil_base_model/logs/pretrain_%x_%j.out
# Pre-train the Tamil base model on one GPU.
#     sbatch --job-name=<run> pretrain.slurm <run> [options of pretrain.py]
# The model goes to models/<run>/. Submitting the same line again continues an interrupted run.
#     TOKENS=<dir> sbatch ...   reads the token files from another folder (for tests)
# Python comes from the virtual environment of the Bulgarian project; it is only read.
set -e
module load gnu12/12.3.0 python/3.11.7-cx
R=/scratch/pgopalan/tamil_base_model
PY=/scratch/pgopalan/project_Praveen/venv/bin/python
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
RUN=$1; shift
[ -n "$RUN" ] || { echo "no run name given"; exit 1; }
cd $R/scripts
echo "node: $(hostname)  job: $SLURM_JOB_ID  run: $RUN  options: $@  started: $(date)"
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
$PY pretrain.py ${TOKENS:-$R/data/tokens/bpe_32000} $R/tokenizer/bpe_32000 $R/models/$RUN "$@"
echo "finished: $(date)"
