#!/bin/bash
#SBATCH --job-name=tdl_v31
#SBATCH --partition=normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=48 --mem=64G --time=1-00:00:00
#SBATCH --output=/scratch/pgopalan/tamil_base_model/logs/tdl_v31_%j.out
# Second OCR pass over the PDFs already downloaded, with the tam_v3.1 Tesseract model
# (github.com/newsbaba24x7/tamil-tess-trainer, Apache-2.0). Resumable: resubmit to continue.
R=/scratch/pgopalan/tamil_base_model
export PATH=$R/tools/ocr-env/bin:$PATH
export TESSDATA_PREFIX=$R/tools/tessdata
export OCR_LANG=tam_v3.1
cd $R
for i in 0 1 2 3; do
  python scripts/tdl_pilot.py data/tdl_pilot/final_run_books.json data/tdl_v31 --pdf-dir data/tdl_full/pdf \
    --workers 12 --shard $i/4 >> logs/tdl_v31_shard$i.log 2>&1 &
done
wait
