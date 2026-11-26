#!/bin/bash
#SBATCH --job-name=ta_tok
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/pgopalan/tamil_base_model/logs/tokenizer_%j.out
# Split the final corpus by document, train the Tamil tokenizer at several vocabulary sizes on
# the training split, and compare them on the validation split.
#     sbatch tokenizer.slurm          full run   -> data/splits/, tokenizer/
#     sbatch tokenizer.slurm test     small test -> tmp/tok_test/ (60 documents per source)
#     sbatch tokenizer.slurm census   split + count of every character -> data/splits/, tokenizer/chars.tsv
# Python comes from the virtual environment of the Bulgarian project (it has `tokenizers` and
# `transformers`); it is only read, nothing is installed.
set -e
module load gnu12/12.3.0 python/3.11.7-cx
R=/scratch/pgopalan/tamil_base_model
PY=/scratch/pgopalan/project_Praveen/venv/bin/python
export RAYON_NUM_THREADS=$SLURM_CPUS_PER_TASK TOKENIZERS_PARALLELISM=true PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
cd $R/scripts
F=$R/data/final
if [ "$1" = test ]; then
  S=$R/tmp/tok_test/splits; T=$R/tmp/tok_test/tokenizer; LIMIT="--limit-docs 60"; SIZES="2000 4000"
  rm -rf $R/tmp/tok_test
else
  S=$R/data/splits; T=$R/tokenizer; LIMIT=""; SIZES="8000 16000 24000 32000 48000 64000"
fi
mkdir -p $S $T

echo "=== 1. split by document"; date
$PY make_split.py $F $S $LIMIT

if [ "$1" = census ]; then
  echo "=== 2. character census only"; date
  $PY train_tokenizer.py $F $S/train.txt $T --census-only; date; exit 0
fi

echo "=== 2. train"; date
$PY train_tokenizer.py $F $S/train.txt $T --vocab $SIZES

echo "=== 3. compare on the validation split"; date
$PY eval_tokenizer.py $F $S/val.txt $T
echo "=== done"; date
