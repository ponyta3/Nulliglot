#!/bin/bash
#SBATCH --job-name=tdl_5078
#SBATCH --partition=normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=48G --time=2-00:00:00
#SBATCH --output=/scratch/pgopalan/tamil_base_model/logs/tdl_1950_78_%j.out
# Nationalised books printed 1950 to 1978: download and OCR with the tam_v3.1 model.
# Resumable: resubmit to continue.
R=/scratch/pgopalan/tamil_base_model
export PATH=$R/tools/ocr-env/bin:$PATH
export TESSDATA_PREFIX=$R/tools/tessdata
export OCR_LANG=tam_v3.1
cd $R
for i in 0 1 2 3; do
  python scripts/tdl_pilot.py data/tdl_pilot/run_1950_78_books.json data/tdl_1950_78 --workers 8 --shard $i/4 \
    >> logs/tdl_1950_78_shard$i.log 2>&1 &
done
wait
