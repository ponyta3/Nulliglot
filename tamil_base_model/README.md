# Tamil base model

A strictly monolingual Tamil base language model, pre-trained from scratch. This is a separate
project from the Bulgarian (`bul_100m`) and German work in the rest of this repository: it shares
no model, tokenizer or data with them.

## Where this project lives

| Place | Path |
|---|---|
| Laptop | `~/Desktop/Research/CS 795/tamil_base_model/` |
| Hopper | `/scratch/pgopalan/tamil_base_model/` |
| GitHub | `tamil_base_model/` in `ponyta3/Nulliglot` |

## Rules decided so far (23 November 2026)

- The model is trained only on Tamil text. No multilingual base model, tokenizer, adapter or data.
- Latin-script brand names and digits inside Tamil text are allowed.
- No machine-translated or synthetic text.
- No corrupted text, no social media or Tanglish, no poorly documented datasets.
- The model will be released on Hugging Face for research, so sources with no licence or a
  non-commercial licence are left out.
- Data comes from Hugging Face datasets: first select good Tamil datasets, then filter and drop
  unwanted rows (English sentences, Tamil written in Latin letters, other Indian scripts).

## Plan

1. **Survey** Tamil datasets, on and off Hugging Face, and shortlist the good ones. Done: see
   `survey/strict_shortlist.md`.
2. **Download and filter** the shortlisted datasets; remove duplicates across sources. Done:
   95.3 million Tamil words in `data/final/` on Hopper (see Status, 25 Nov 2026).
3. **Train a Tamil-only tokenizer** on the filtered text.
4. **Pre-train** the base model on Hopper.
5. **Test** it: validation loss, context use, copying a text from the prompt.

## Status

- 23 Nov 2026: project started. A strict survey on and off Hugging Face found about
  120 million words that meet every rule with a licence from the authors or publisher
  (Wikipedia, proofread Wikisource, Thamizh Mann nationalised books, small news and ebook sets).
  The large sources (Sangraha verified, about 2.2 billion words; HPLT 3.0) carry a licence from
  the collector only. A third search found nothing substantial to add, and an audit found that
  about a third of Tamil Wikipedia is machine-translated or bot-generated; with that removed,
  the fully compliant group is about 95 to 100 million words. Which groups to use is not
  decided. Nothing has been downloaded yet.
- 23 Nov 2026: OCR pilot on 19 nationalised books from the Tamil Digital Library
  (`survey/tdl_ocr_pilot.md`). Books printed from about 1979 onward come out clean (96.4% of
  words exact on a proofread benchmark book); books printed before 1970 come out noisy. Whether
  to scale this up is not decided. Pilot PDFs and text are on the laptop only (`data/tdl_pilot/`).
- 24 Nov 2026: full OCR run on Hopper finished: 1,766 nationalised books read. After the
  confidence gate, cleaning and removal of pages already proofread on Wikisource, 27.6 million
  words remain at gate 84 and 20.1 million at gate 87. About one word in 20 to 30 is still
  wrong. Whether this meets the "no corrupted text" rule, and which gate to use, is not decided.
- 24 Nov 2026: second OCR run with a Tesseract model retrained on scanned Tamil books
  (`tam_v3.1`). Real letter mistakes fall from about 1 word in 15 to 1 in 44 on a 28-book
  benchmark. 40.5 million words after cleaning and duplicate removal; 33.1 million with a
  line-level confidence filter. Which version to use is not decided.
- 24 Nov 2026: decided to use the second run without the line filter (40.5 million words). A
  test on older print showed books of 1950 to 1978 read almost as well (about 1 wrong word in
  30), books before 1950 do not (about 1 in 14). The 1950 to 1978 books were then run: 1,036
  books read, 19.6 million words after cleaning and duplicate removal. OCR source total: about
  60 million words, before a text-level check against Wikisource and Thamizh Mann.
- 25 Nov 2026: PDFs deleted; raw OCR text and word confidences kept. The text was cleaned again
  from the saved OCR output: sentences now run on across page breaks, and pages that appear in
  more than one book are removed. Final OCR text: 54.2 million words in 2,651 books
  (`data/tdl_v31/c_final`, `data/tdl_1950_78/c_final`).
- 25 Nov 2026: raw files of sources 1 to 6 of `survey/final_list.md` downloaded to
  `data/raw/` on Hopper with `scripts/download_sources.py`: the Wikisource, Wikinews, Wikibooks
  and Wikiquote dumps (dated 1 to 2 Nov 2026, checksums verified), the Thamizh Mann folder
  (640 files), all 2,375 Kaniyam posts, and 484 FreeTamilEbooks EPUBs. The 414 ebooks with a
  non-commercial or no-derivatives licence were not downloaded; 136 of the 484 state no exact
  licence on their page and need a check inside the book. Nothing is cleaned or filtered yet.
- 25 Nov 2026: all sources cleaned and duplicates removed across sources
  (`scripts/build_corpus.slurm`, Hopper job 3910224, 1 h 42 min). Final corpus:
  **95.3 million Tamil words** in 10,906 documents (2.35 GB) in `data/final/` on Hopper, one
  folder per source, with `SUMMARY.tsv`, `dedup_stats.tsv` and `ocr_dup_pages.tsv`.

  | Source | Documents | Tamil words | Words in other scripts |
  |---|---|---|---|
  | OCR books, 1979 onward | 1,614 | 29,754,580 | 0.14% |
  | Wikisource | 1,888 | 26,968,120 | 0.72% |
  | Thamizh Mann | 549 | 18,900,192 | 0.29% |
  | OCR books, 1950 to 1978 | 956 | 14,939,626 | 0.13% |
  | FreeTamilEbooks | 296 | 3,704,381 | 0.66% |
  | Wikinews | 3,342 | 486,509 | 0.25% |
  | Kaniyam | 534 | 180,015 | 4.24% |
  | Wikibooks | 692 | 172,261 | 0.99% |
  | Wikiquote | 1,035 | 153,713 | 0.63% |
  | Total | 10,906 | 95,259,397 | 0.36% |

  Before duplicate removal the sources held 120.1 million words; 24.8 million (21%) were copies
  of text in another document. The largest overlaps: 5.2 million words of the 1979-onward OCR
  books are already on Wikisource (the Wikisource copy is kept), and FreeTamilEbooks loses
  almost half its words (7.1 to 3.7 million), nearly all to Wikisource. The two OCR sets fall
  from 60.0 to 44.7 million words. Of the 484 FreeTamilEbooks, 338 were kept by the cleaner
  before duplicate removal: 67 were dropped for a restricted licence found inside the book, 36
  for no licence stated, 24 as translations, 18 for damaged text and 1 for almost no Tamil text.
  Kaniyam has the most words in other scripts (4.24%) and should be looked at when the row
  filter is fixed.

## Not decided yet

- The goal: a research model, or one that should answer questions well. This sets how much
  data is needed.
- Model size, context length and vocabulary size.
- The exact row filter. Proposed: keep a row if at least 95% of its letters are Tamil script.
- Advisor agreement and Hopper compute for the pre-training run.

## Folders

| Folder | Content |
|---|---|
| `survey/` | `final_list.md` is the final list of sources; `exclusions/` holds the items to drop; `strict_shortlist.md` is the full survey result (`dataset_shortlist.md` is the earlier, looser list); `probe.py` samples a Hugging Face dataset and measures how much of it is Tamil script |
| `scripts/` | download, OCR, cleaning and duplicate-removal scripts (`build_corpus.slurm` rebuilds `data/final/`); tokenizer and training scripts (none yet) |
| `data/`, `models/` | downloaded text and model weights; kept on the laptop and Hopper only, not in git |
| `tokenizer/`, `logs/` | the Tamil tokenizer and training logs (none yet) |
