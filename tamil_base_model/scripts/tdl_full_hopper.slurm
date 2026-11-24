#!/bin/bash
#SBATCH --job-name=tdl_ocr
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/pgopalan/tamil_base_model/logs/tdl_full_%j.out

# Download and OCR the nationalised books from the Tamil Digital Library.
# Resumable: books whose OCR text already exists are skipped, so the job can be resubmitted.
R=/scratch/pgopalan/tamil_base_model
export PATH=$R/tools/ocr-env/bin:$PATH
export TESSDATA_PREFIX=$R/tools/tessdata
cd $R
for i in 0 1 2 3; do
  python scripts/tdl_pilot.py data/tdl_pilot/final_run_books.json data/tdl_full --workers 4 --shard $i/4 \
    >> logs/tdl_full_shard$i.log 2>&1 &
done
wait
echo "finished $(date)  books: $(ls data/tdl_full/ocr | wc -l)"
