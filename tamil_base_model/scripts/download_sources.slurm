#!/bin/bash
#SBATCH --job-name=ta_dl
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/pgopalan/tamil_base_model/logs/download_sources_%j.out
# Download the raw files of sources 1 to 6 (survey/final_list.md) into data/raw/.
R=/scratch/pgopalan/tamil_base_model
export PATH=$R/tools/ocr-env/bin:$PATH
cd $R
python -u scripts/download_sources.py data/raw $SOURCES
du -sh data/raw/*
