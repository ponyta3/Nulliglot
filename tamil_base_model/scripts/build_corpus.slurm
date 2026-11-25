#!/bin/bash
#SBATCH --job-name=ta_build
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/pgopalan/tamil_base_model/logs/build_corpus_%j.out
# Build the cleaned Tamil corpus from the raw files. Nothing is downloaded and no OCR is run.
#   1. clean the seven typed sources          data/raw/*            -> data/clean/<source>
#   2. clean the OCR books page by page       data/tdl_*/ocr        -> data/clean/ocr_pages_*
#   3. find duplicate text across everything  data/clean/*          -> data/final/<source>, ocr_dup_pages.tsv
#   4. write the OCR books again without the duplicate pages, sentences joined across pages
#   5. count documents and words              data/final/SUMMARY.tsv
set -e
R=/scratch/pgopalan/tamil_base_model
export PATH=$R/tools/ocr-env/bin:$PATH
cd $R/scripts
RAW=$R/data/raw; C=$R/data/clean; F=$R/data/final; X=$R/survey/exclusions; W=$RAW/wikimedia
WS=$R/data/tdl_ocr_dedup/ws_proofread_pages.tsv
rm -rf $C $F; mkdir -p $C $F

echo "=== 1. typed sources"; date
python clean_wikisource.py $W/tawikisource-latest-pages-articles.xml.bz2 $C/wikisource $X/tawikisource_translation_flags.txt > $C/wikisource.log 2>&1 &
python clean_thamizh_mann.py $RAW/thamizh_mann $C/thamizh_mann $X/thamizh_mann_flagged.tsv --joins-out $C/thamizh_mann_joins.tsv > $C/thamizh_mann.log 2>&1 &
python clean_fte.py $RAW/fte $C/fte $X/fte_translation_flagged.tsv > $C/fte.log 2>&1 &
python clean_kaniyam.py $RAW/kaniyam $C/kaniyam $X/kaniyam_flagged_posts.tsv > $C/kaniyam.log 2>&1 &
for w in wikinews wikibooks wikiquote; do
  python clean_wiki_small.py $W/ta$w-latest-pages-articles.xml.bz2 $C/$w $w > $C/$w.log 2>&1 &
done
echo "=== 2. OCR books, page level"
python tdl_clean.py $R/data/tdl_v31 $C/ocr_pages_1979_on --book-conf 80 --skip-pages $WS > $C/ocr_pages_1979_on.log 2>&1 &
python tdl_clean.py $R/data/tdl_1950_78 $C/ocr_pages_1950_78 --book-conf 80 --skip-pages $WS > $C/ocr_pages_1950_78.log 2>&1 &
wait
for f in $C/*.log; do echo "--- $(basename $f)"; cat $f; done

echo "=== 3. duplicates across all sources"; date
python -u corpus_dedup.py $F \
  --typed wikisource=$C/wikisource thamizh_mann=$C/thamizh_mann fte=$C/fte kaniyam=$C/kaniyam \
          wikinews=$C/wikinews wikibooks=$C/wikibooks wikiquote=$C/wikiquote \
  --ocr ocr_1979_on=$C/ocr_pages_1979_on ocr_1950_78=$C/ocr_pages_1950_78

echo "=== 4. OCR books without duplicate pages, pages joined"; date
python tdl_clean.py $R/data/tdl_v31 $F/ocr_1979_on --book-conf 80 --skip-pages $WS --skip-pages $F/ocr_dup_pages.tsv --join-pages &
python tdl_clean.py $R/data/tdl_1950_78 $F/ocr_1950_78 --book-conf 80 --skip-pages $WS --skip-pages $F/ocr_dup_pages.tsv --join-pages &
wait
find $F/ocr_1979_on $F/ocr_1950_78 -name '*.txt' -size 0 -delete

echo "=== 5. summary"; date
python corpus_summary.py $F | tee $F/SUMMARY.tsv
date
