# Strict Tamil shortlist

Second, stricter survey, 23 November 2026, on and off Hugging Face. It replaces
`dataset_shortlist.md`, which used looser rules.

## Rules applied

A source is kept only if it meets all of these:

1. Original Tamil text in Tamil script. English brand names and digits are allowed.
2. No machine-translated or synthetic text.
3. A clear licence that allows a research release. No "no licence", no non-commercial.
4. No corrupted text (bad OCR, broken font encodings).
5. No social media, no Tanglish.
6. Well documented: the source says where the text came from.

No source is 100% Tamil as delivered. Every one needs a row filter that drops lines which are
not mostly Tamil script. "Text kept" below is the share of characters that survives a filter
keeping lines with at least 90% Tamil letters.

## Group A: meets every rule (licence granted by the authors, publisher or government)

| # | Source | Text | Words (millions) | Purity | Licence | Filter needed | Link |
|---|---|---|---|---|---|---|---|
| A1 | Tamil Wikipedia, Wikimedia dump | Encyclopedia, 192,314 articles | 57 (stated by the site) | 81% of articles are 95%+ Tamil; 94% of text kept (measured on the 2023 copy) | CC BY-SA 4.0 | Line filter; drop templated stubs. Share of machine-translated articles not measured | [site](https://ta.wikipedia.org/) · [dump](https://dumps.wikimedia.org/tawiki/latest/) |
| A2 | Tamil Wikisource, proofread and validated pages only | Books, mostly 20th-century prose, some verse | about 33 (estimate) | 98.6% of text kept on the main-namespace copy | CC BY-SA 4.0; many books nationalised by the Tamil Nadu government (public domain) | Strip wiki markup; skip the 390,901 unproofread OCR pages; Quran pages contain Arabic | [site](https://ta.wikisource.org/) · [dump](https://dumps.wikimedia.org/tawikisource/latest/) |
| A3 | Thamizh Mann nationalised books (University of Toronto Scarborough, Digital Tamil Studies) | 639 volumes of 20th-century scholarship and prose, whole books, publisher's digital text | about 30 (estimate) | 98.9–100% Tamil letters in 6 files; one checked file has 491,547 Tamil, 277 Latin, 0 Devanagari letters | CC BY-SA 4.0, stated per book in `metadata.csv` | Drop 5 English volumes; optionally 41 dictionaries and 16 human-translated works | [files](https://github.com/Digital-Tamil-Studies/open_tamil_texts/tree/master/collections/thamizh_mann) |
| A4 | Tamil Wikinews | Volunteer-written news, mostly 2009–2014 | 0.5–0.8 | not measured | CC BY 4.0 | Strip wiki markup | [site](https://ta.wikinews.org/) |
| A5 | Tamil Wikibooks and Wikiquote | Textbooks, quotations | 0.2 and 0.3 | not measured | CC BY-SA 4.0 | Strip wiki markup | [Wikibooks](https://ta.wikibooks.org/) · [Wikiquote](https://ta.wikiquote.org/) |
| A6 | Kaniyam magazine | Technology magazine, 2012 to now, 2,375 posts | about 1.3 (estimate) | about 13% of words are Latin-script technical terms | Creative Commons with attribution and share-alike (site footer) | Drop English-heavy posts and code | [site](https://www.kaniyam.com/) |
| A7 | FreeTamilEbooks, openly licensed books only | Modern prose, fiction, essays; about 394 of 905 books | unknown | one clean book 99.7% Tamil; one OCR book had broken text | Per book: CC0, public domain, CC BY or CC BY-SA | Keep only those licences; screen out OCR-damaged books; contains unmarked human translations | [site](https://freetamilebooks.com/) |

Group A total: about 120 million words, before removing overlap. A2 and A3 both contain
nationalised books, so part of that is counted twice.

## Group B: large, but the licence comes from the collector, not the original publishers

These have a stated open licence and good documentation, so they pass the rules as written.
The licence was applied by the group that collected the text; the web sites and newspapers the
text came from did not grant it. This is the normal basis for research language models.

| # | Source | Text | Words (millions) | Purity | Licence | Filter needed | Link |
|---|---|---|---|---|---|---|---|
| B1 | `ai4bharat/sangraha`, `verified` subset, `tam` split | Curated web pages (about 91%, mostly news sites), PDF/OCR (8%), lecture transcripts (0.4%) | about 2,200 (estimate) | 90% of documents are 95%+ Tamil; 96.4% of text kept; 1 of 43 documents read looked machine-translated | CC BY 4.0 (AI4Bharat) | Drop `type=speech`; repair ":" stored as "ஃ" (about 27% of documents); line filter | [file list](https://huggingface.co/datasets/ai4bharat/sangraha/tree/main/verified/tam); readable sample in `samples/sangraha_verified_tam_sample.txt` |
| B2 | HPLT 3.0, `tam_Taml`, quality bins 7–10 | Web pages 2012–2024 with URL, quality score and a machine-translation label per page | several thousand (not verified) | 97–99.6% Tamil letters in bins 7–10 | CC0 on the packaging only | Drop pages with the machine-translation label (14–18% of pages sampled in the top two bins); block forum sites; remove Wikipedia copies | [card](https://huggingface.co/datasets/HPLT/HPLT3.0) |
| B3 | WMT News Crawl, Tamil | Sentences from commercial news sites, 2008–2025 | about 69 million tokens (stated) | 99.9% Tamil letters in one file | CC0 on the packaging only | Shuffled sentences, not whole articles | [files](https://data.statmt.org/news-crawl/ta/) |

## Group C: informal or restricted permission, needs a decision

| Source | Text | Words (millions) | Issue | Link |
|---|---|---|---|---|
| Project Madurai, redistribution category | Classical and devotional verse, some prose; 1,318 works | 16.5 Tamil-script | No formal licence. Notice: "You are welcome to freely distribute this file, provided this header page is kept intact". 41 English translations to drop | [site](https://www.projectmadurai.org/pmworks.html) |
| Maatram | Sri Lankan citizen journalism, 1,742 posts | about 1.5 (estimate) | CC BY, no derivatives | [site](https://maatram.org/) |

## Dropped from the earlier shortlist

- `HuggingFaceFW/fineweb-2` (Tamil): about 7% of documents read were machine-translated content
  farms, and it has no label to remove them. Its "high quality" subset was worse (12 of 45).
- `ramSeraph/indic_wikisource`: poorly documented; the same text is rebuilt from the Wikisource
  dump (A2).
- `wikimedia/wikisource` on Hugging Face: passes, but it is contained in A2.

## Fails, do not reconsider

- **Copyrighted news:** Dinamalar, Dinamani, Dinakaran, Vikatan, The Hindu Tamil, Tamil Murasu,
  BBC Tamil, OneIndia and the Kaggle or Hugging Face scrapes of them, whatever label the
  uploader chose.
- **Non-commercial or restricted:** XL-Sum, Vinavu, EMILLE/CIIL, LDC-IL, LDC, IndicCorp v1.
- **No usable licence:** Tamil Virtual Academy library, Tamil Digital Library, Noolaham,
  Tamil Nadu textbooks, Assembly debates, Vikaspedia, Tamil Wiki, CC-100, All India Radio.
- **Translated or synthetic:** Sangraha `synthetic`, Pralekha, PIB Tamil, Supreme Court
  judgments in Tamil, every Tamil corpus in OPUS, all Tamil Alpaca, Dolly and Orca sets.
- **Social media or code-mixed:** all Tamil shared-task sets (sentiment, offensive language).
- **Isolated sentences or copies:** IndicCorp v2, Leipzig corpora, re-uploads of Wikipedia.

## Limits of this survey

- Several sizes are estimates from samples or are stated by the source.
- The licence page of the Thamizh Mann collection at the University of Toronto blocked automated
  access; the licence wording comes from the repository's metadata and should be checked by hand.
- Human translations into Tamil exist inside A2, A3 and A7 and are not all marked.
- Machine-translated pages cannot be fully removed from B1; only B2 labels them.
