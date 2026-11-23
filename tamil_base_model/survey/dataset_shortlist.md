# Tamil dataset shortlist

Survey of Hugging Face for original Tamil text, 23 November 2026. Rules applied: Tamil script
only, brand names and digits allowed, no machine-translated or generated text.

**How to read the numbers.** Hugging Face rate-limited the sampling, so sizes are estimates:

- **P**: measured with `probe.py` on 300 or more sampled rows.
- **R**: measured on a small range of the files (first part of a file, or a few files).
- **C**: read from the dataset card only.

"Tamil rows" is the share of sampled documents in which at least 95% of the letters are Tamil
script. Word counts are before filtering and before removing duplicates across sources.

## Core shortlist

| # | Dataset | Text | Words (millions) | Document length (words) | Tamil rows | Licence | Work needed | Link |
|---|---|---|---|---|---|---|---|---|
| 1 | `ai4bharat/sangraha`, `verified/tam` | Curated web pages (91%), PDF/OCR (8%), transcripts | about 2,190 (R) | whole pages, median 165 | 90% | CC-BY-4.0 | Repair ":" stored as "ஃ" (about 26% of documents); drop navigation fragments | [Tamil file list](https://huggingface.co/datasets/ai4bharat/sangraha/tree/main/verified/tam); to read the text, open `samples/sangraha_verified_tam_sample.txt` |
| 2 | `HuggingFaceFW/fineweb-2`, `tam_Taml` | Web pages from Common Crawl, 2013–2024 | 1,885 (P) | whole pages, median 191 | 93% | ODC-By | Some machine-translated content-farm pages, no label to remove them | [view](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/viewer/tam_Taml) |
| 3 | `wikimedia/wikipedia`, `20231101.ta` | Encyclopedia | 38 (P) | median 105 | 81% | CC-BY-SA | Drop templated stubs (about a quarter of articles, 8% of words) | [view](https://huggingface.co/datasets/wikimedia/wikipedia/viewer/20231101.ta) |
| 4 | `ramSeraph/indic_wikisource`, Tamil rows | Proofread book pages, mostly 20th-century prose | about 22 (R) | 140 per page | 93% | CC-BY-SA / GFDL | Strip wiki templates; join pages into books | [Tamil file](https://huggingface.co/datasets/ramSeraph/indic_wikisource/blob/main/ta.parquet), [viewer at the Tamil rows](https://huggingface.co/datasets/ramSeraph/indic_wikisource/viewer/default/train?p=4500) |
| 5 | `Joemn/project-madurai-corpus-tamil`, `raw_markdown` | 1,346 literary works, mostly classical verse | about 20 (R) | about 12,500 per work | 76% | per work | Drop English volumes and reserved-rights works | [view](https://huggingface.co/datasets/Joemn/project-madurai-corpus-tamil/viewer/raw_markdown) |
| 6 | `wikimedia/wikisource`, `20231201.ta` | Classics, Bharathi, Kalki | 4.7 (P) | median 780 | 95% | CC-BY-SA | About 12% is Bible and Quran (human translations) | [view](https://huggingface.co/datasets/wikimedia/wikisource/viewer/20231201.ta) |

Only the Tamil part of each dataset is used. From `ai4bharat/sangraha` only the `verified`
subset, `tam` split is taken (53 files, 18.7 GB); its `synthetic` subset is machine-translated
and is not used, and no other language is downloaded. The Hugging Face viewer does not work for
this part of Sangraha (it returns an error and can show another language), and the file-list
link shows file names only. To read the text, open `samples/sangraha_verified_tam_sample.txt`:
the first 200 documents, read straight from the first Tamil file (433,274 Tamil letters,
152 Devanagari letters, 6,349 Latin letters). `ramSeraph/indic_wikisource` holds twelve languages in
one table; the Tamil rows are the file `ta.parquet`.

Sources 1 and 2 together hold about 4 billion words before duplicates are removed. Both crawl
Tamil web sites, so they overlap; the overlap has not been measured. Sources 3 to 6 add about
85 million words of cleaner and longer text.

## Left out for licence reasons

The model will be released on Hugging Face for research, so sources without a usable licence
are not used (decided 23 November 2026).

| Dataset | Text | Words (millions) | Reason | Link |
|---|---|---|---|---|
| `livinNector/tamil_news_dataset`, `train` | Dinamalar and Tamil Murasu news articles | 230–300 (P) | No licence; copyrighted newspaper text scraped from the sites | [view](https://huggingface.co/datasets/livinNector/tamil_news_dataset) |
| `csebuetnlp/xlsum`, `tamil` | BBC Tamil articles | about 12 (R) | Non-commercial and share-alike; it would restrict the model's licence for 0.3% more text. Can be added back if the model is released as non-commercial | [view](https://huggingface.co/datasets/csebuetnlp/xlsum/viewer/tamil) |

## Optional, if more text is needed

| Dataset | Text | Words (millions) | Why not core | Link |
|---|---|---|---|---|
| HPLT 3.0, `tam_Taml` (files hosted on data.hplt-project.org, not Hugging Face) | Web pages, Internet Archive and Common Crawl | about 3,700, unverified (C) | Not on Hugging Face; overlaps FineWeb-2. It labels each page with a machine-translation score, which no other source has | [view](https://huggingface.co/datasets/HPLT/HPLT3.0) |
| `ai4bharat/sangraha`, `unverified/tam` | CulturaX and MADLAD pages | about 690 (R) | Same Common Crawl text as FineWeb-2 | [view](https://huggingface.co/datasets/ai4bharat/sangraha) |
| `ai4bharat/IndicCorpV2`, `tam_Taml` | News crawl | about 470 (R) | Shuffled single paragraphs, so no long context | [view](https://huggingface.co/datasets/ai4bharat/IndicCorpV2) |
| `ai4bharat/IndicHeadlineGeneration`, `ta` | News articles | about 12 (R) | Non-commercial | [view](https://huggingface.co/datasets/ai4bharat/IndicHeadlineGeneration) |
| `LMA-Project-Resources-Vijay/tamil-corpus`, crawl, books and OCR parts | Mixed compilation | unknown | Mixed licences; contains YouTube auto-captions that must be left out | [view](https://huggingface.co/datasets/LMA-Project-Resources-Vijay/tamil-corpus) |
| `Joemn/TamilNadu-State-board-books-2025-Tamil-and-english-versions`, Tamil-medium | 160 school textbooks | unknown (46.6M characters, C) | Government copyright, no open licence | [view](https://huggingface.co/datasets/Joemn/TamilNadu-State-board-books-2025-Tamil-and-english-versions) |
| `aiwithvarun7/theekkathir-text-dataset` | News articles | unknown, small | Could not be sampled | [view](https://huggingface.co/datasets/aiwithvarun7/theekkathir-text-dataset) |

## Excluded

- **Machine-translated:** `ai4bharat/sangraha` `synthetic` (both scripts), `ai4bharat/Pralekha`,
  and all Tamil Alpaca, Dolly, Orca and similar instruction sets.
- **Same Common Crawl text as FineWeb-2:** `uonlp/CulturaX` (gated), `oscar-corpus/OSCAR-2301`
  (gated), `allenai/MADLAD-400`, `cis-lmu/GlotCC-V1`, `statmt/cc100`, `HPLT/HPLT2.0_cleaned`
  (replaced by 3.0).
- **Corrupted text:** `HuggingFaceFW/finepdfs` (about half of sampled PDFs are garbled).
- **Re-packaged copies** of Wikipedia, IndicCorp or Project Madurai:
  `wickkiey/tamil-wikipedia-markdown`, `muthuramkumar/tamil_corpus_2.3m`,
  `Hemanth-thunder/Tamil_Articles_Corpus`, `mozhi-ai/tamil-corpus`,
  `minimalist-ai/TamilDataset`, `Hemanth-thunder/ta-dinamalar-dataset`,
  `velkadamban/Project_Madurai`, `mastergokul/project-madurai-books`,
  `nirmalpratheep/TamilTextCorpus`.
- **Not original Tamil prose:** subtitles (translated films), speech transcripts, code-mixed
  social media, isolated-sentence collections of unknown source.

## For later stages (not for pre-training)

- **Instruction data:** almost every Tamil instruction set is machine-translated. The
  human-written options found are the Tamil part of `CohereLabs/aya_dataset`,
  `aitamilnadu/thirukkural_instruct` (3,990 rows) and `aitamilnadu/tamil_stories` (1,202 rows).
- **Benchmarks with original Tamil:** `ai4bharat/IndicQA` (Tamil),
  `nirantk/chaii-hindi-and-tamil-question-answering`, `CohereLabs/include-base-44` (Tamil),
  `ai4bharat/MILU` (gated). The benchmark notes come partly from memory of the papers and were
  not all checked.

## Known limits of this survey

- Sizes are from samples; exact counts need the full download.
- The overlap between sources is unmeasured, so the unique total is unknown.
- Machine-translated web pages cannot be fully removed. Only HPLT 3.0 labels them.
- Licences: the licence on a web-crawl dataset (sources 1 and 2) covers the collection; the
  pages themselves still belong to their sites. This is the usual basis for released research
  models, but it is not the same as permission from every site.
