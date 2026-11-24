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
2. **Download and filter** the shortlisted datasets; remove duplicates across sources.
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
| `scripts/` | download, filter, tokenizer and training scripts (none yet) |
| `data/`, `models/` | downloaded text and model weights; kept on the laptop and Hopper only, not in git |
| `tokenizer/`, `logs/` | the Tamil tokenizer and training logs (none yet) |
