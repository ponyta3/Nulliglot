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
  the collector only. Which groups to use is not decided. Nothing has been downloaded yet.

## Not decided yet

- The goal: a research model, or one that should answer questions well. This sets how much
  data is needed.
- Model size, context length and vocabulary size.
- The exact row filter. Proposed: keep a row if at least 95% of its letters are Tamil script.
- Advisor agreement and Hopper compute for the pre-training run.

## Folders

| Folder | Content |
|---|---|
| `survey/` | `strict_shortlist.md` is the survey result (`dataset_shortlist.md` is the earlier, looser list); `probe.py` samples a Hugging Face dataset and measures how much of it is Tamil script |
| `scripts/` | download, filter, tokenizer and training scripts (none yet) |
| `data/`, `models/` | downloaded text and model weights; kept on the laptop and Hopper only, not in git |
| `tokenizer/`, `logs/` | the Tamil tokenizer and training logs (none yet) |
