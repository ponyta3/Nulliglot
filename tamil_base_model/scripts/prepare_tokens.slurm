#!/bin/bash
#SBATCH --job-name=ta_tokens
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/pgopalan/tamil_base_model/logs/tokens_%j.out
# Turn the train, validation and test splits into token-id files with the chosen tokenizer.
#     sbatch prepare_tokens.slurm          full corpus -> data/tokens/bpe_32000/
#     sbatch prepare_tokens.slurm test     40 documents per split -> tmp/pretrain_test/tokens/
# Python comes from the virtual environment of the Bulgarian project; it is only read.
set -e
module load gnu12/12.3.0 python/3.11.7-cx
R=/scratch/pgopalan/tamil_base_model
PY=/scratch/pgopalan/project_Praveen/venv/bin/python
export RAYON_NUM_THREADS=$SLURM_CPUS_PER_TASK TOKENIZERS_PARALLELISM=true PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
cd $R/scripts
if [ "$1" = test ]; then OUT=$R/tmp/pretrain_test/tokens; LIMIT="--limit-docs 40"
else OUT=$R/data/tokens/bpe_32000; LIMIT=""; fi
echo "node: $(hostname)  job: $SLURM_JOB_ID  started: $(date)"
$PY prepare_tokens.py $R/data/final $R/data/splits $R/tokenizer/bpe_32000 $OUT $LIMIT
cat $OUT/meta.json | head -40
echo "finished: $(date)"
