#!/bin/bash
#SBATCH --job-name=ocr_old
#SBATCH --partition=normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=24 --mem=32G --time=04:00:00
#SBATCH --output=/scratch/pgopalan/tamil_base_model/logs/ocr_pre1979_%j.out
# Test of the tam_v3.1 model on 40 books printed before 1979: download, OCR, score against
# proofread Wikisource pages.
R=/scratch/pgopalan/tamil_base_model
export PATH=$R/tools/ocr-env/bin:$PATH
export TESSDATA_PREFIX=$R/tools/tessdata
cd $R
B=data/ocr_bench_pre1979; T=data/tdl_pre1979_test
for i in 0 1 2 3; do
  OCR_LANG=tam_v3.1 python scripts/tdl_pilot.py $B/books.json $T --workers 6 --shard $i/4 >> logs/ocr_pre1979_shard$i.log 2>&1 &
done
wait
ls $T/ocr | wc -l
python scripts/ocr_bench_run.py $B/bench.json $T/pdf $T/ocr $B/tess_stock_fast.json tam 2
python scripts/ocr_bench_run.py $B/bench.json $T/pdf $T/ocr $B/tess_v31.json tam_v3.1 2
python scripts/ocr_bench_score.py $B/bench.json $B/tess_stock_fast.json $B/tess_v31.json
