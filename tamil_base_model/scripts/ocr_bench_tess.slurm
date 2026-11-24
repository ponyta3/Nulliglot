#!/bin/bash
#SBATCH --job-name=ocr_bench
#SBATCH --partition=normal
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=24 --mem=32G --time=01:00:00
#SBATCH --output=/scratch/pgopalan/tamil_base_model/logs/ocr_bench_%j.out
R=/scratch/pgopalan/tamil_base_model
export PATH=$R/tools/ocr-env/bin:$PATH
export TESSDATA_PREFIX=$R/tools/tessdata
cd $R
B=data/ocr_bench
for m in tam_v3.1 tam_fast_v3.1; do
  [ -s $TESSDATA_PREFIX/$m.traineddata ] || curl -sSL -o $TESSDATA_PREFIX/$m.traineddata \
    https://github.com/newsbaba24x7/tamil-tess-trainer/releases/download/v3.1.0/$m.traineddata
done
[ -s $TESSDATA_PREFIX/tam_best.traineddata ] || curl -sSL -o $TESSDATA_PREFIX/tam_best.traineddata \
  https://github.com/tesseract-ocr/tessdata_best/raw/main/tam.traineddata
ls -la $TESSDATA_PREFIX/*.traineddata
python scripts/ocr_bench_run.py $B/bench.json data/tdl_full/pdf data/tdl_full/ocr $B/tess_stock_fast.json tam 2
python scripts/ocr_bench_run.py $B/bench.json data/tdl_full/pdf data/tdl_full/ocr $B/tess_stock_best.json tam_best 2
python scripts/ocr_bench_run.py $B/bench.json data/tdl_full/pdf data/tdl_full/ocr $B/tess_v31.json tam_v3.1 2
python scripts/ocr_bench_run.py $B/bench.json data/tdl_full/pdf data/tdl_full/ocr $B/tess_v31_otsu.json tam_v3.1 0
python scripts/ocr_bench_run.py $B/bench.json data/tdl_full/pdf data/tdl_full/ocr $B/tess_v31_fast.json tam_fast_v3.1 2
python scripts/ocr_bench_score.py $B/bench.json $B/tess_stock_fast.json $B/tess_stock_best.json $B/tess_v31.json $B/tess_v31_otsu.json $B/tess_v31_fast.json
