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

| # | Dataset | Text | Words (millions) | Document length (words) | Tamil rows | Licence | Work needed |
|---|---|---|---|---|---|---|---|
| 1 | `ai4bharat/sangraha`, `verified/tam` | Curated web pages (91%), PDF/OCR (8%), transcripts | about 2,190 (R) | whole pages, median 165 | 90% | CC-BY-4.0 | Repair ":" stored as "ஃ" (about 26% of documents); drop navigation fragments |
| 2 | `HuggingFaceFW/fineweb-2`, `tam_Taml` | Web pages from Common Crawl, 2013–2024 | 1,885 (P) | whole pages, median 191 | 93% | ODC-By | Some machine-translated content-farm pages, no label to remove them |
| 3 | `livinNector/tamil_news_dataset`, `train` | Dinamalar and Tamil Murasu news articles | 230–300 (P) | whole articles, median 80–90 | 97% | none stated | Copyrighted newspaper text |
| 4 | `wikimedia/wikipedia`, `20231101.ta` | Encyclopedia | 38 (P) | median 105 | 81% | CC-BY-SA | Drop templated stubs (about a quarter of articles, 8% of words) |
| 5 | `ramSeraph/indic_wikisource`, Tamil rows | Proofread book pages, mostly 20th-century prose | about 22 (R) | 140 per page | 93% | CC-BY-SA / GFDL | Strip wiki templates; join pages into books |
| 6 | `Joemn/project-madurai-corpus-tamil`, `raw_markdown` | 1,346 literary works, mostly classical verse | about 20 (R) | about 12,500 per work | 76% | per work | Drop English volumes and reserved-rights works |
| 7 | `csebuetnlp/xlsum`, `tamil` | BBC Tamil articles | about 12 (R) | median 350 | 99% | CC-BY-NC-SA | Non-commercial |
| 8 | `wikimedia/wikisource`, `20231201.ta` | Classics, Bharathi, Kalki | 4.7 (P) | median 780 | 95% | CC-BY-SA | About 12% is Bible and Quran (human translations) |

Sources 1 and 2 together hold about 4 billion words before duplicates are removed. Both crawl
Tamil web sites, so they overlap; the overlap has not been measured. Sources 3 to 8 add about
0.3 billion words of cleaner and longer text.

## Optional, if more text is needed

| Dataset | Text | Words (millions) | Why not core |
|---|---|---|---|
| HPLT 3.0, `tam_Taml` (hosted on data.hplt-project.org, not Hugging Face) | Web pages, Internet Archive and Common Crawl | about 3,700, unverified (C) | Not on Hugging Face; overlaps FineWeb-2. It labels each page with a machine-translation score, which no other source has |
| `ai4bharat/sangraha`, `unverified/tam` | CulturaX and MADLAD pages | about 690 (R) | Same Common Crawl text as FineWeb-2 |
| `ai4bharat/IndicCorpV2`, `tam_Taml` | News crawl | about 470 (R) | Shuffled single paragraphs, so no long context |
| `ai4bharat/IndicHeadlineGeneration`, `ta` | News articles | about 12 (R) | Shares BBC Tamil with XL-Sum; non-commercial |
| `LMA-Project-Resources-Vijay/tamil-corpus`, crawl, books and OCR parts | Mixed compilation | unknown | Mixed licences; contains YouTube auto-captions that must be left out |
| `Joemn/TamilNadu-State-board-books-2025-Tamil-and-english-versions`, Tamil-medium | 160 school textbooks | unknown (46.6M characters, C) | Government copyright, no open licence |
| `aiwithvarun7/theekkathir-text-dataset` | News articles | unknown, small | Could not be sampled |

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
- Licences: the newspaper sets have none, and several sources are non-commercial. This matters
  if the model or the data is ever released.
