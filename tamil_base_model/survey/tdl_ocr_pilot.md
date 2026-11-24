# OCR pilot: nationalised books from the Tamil Digital Library

23 November 2026, run on the laptop. Question: can scanned nationalised books from the Tamil
Digital Library (tamildigitallibrary.in, Tamil Virtual Academy) be turned into text clean enough
for the strictly monolingual Tamil model?

## Result

- Books printed from about 1979 onward come out clean. All 9 such books in the pilot scored a mean
  Tesseract confidence of 84 or more, except one at 83.4.
- Books printed before 1970 come out noisy. None of the 9 reached 84, and the two from 1934 are
  heavily corrupted.
- On one book that Wikisource volunteers have fully proofread (printed 1964, confidence 88),
  96.4% of words came out exactly right. Real letter mistakes were 0.74% (0.49% wrong letter,
  0.25% missing letter). A further 2.7% of letters were extra text: page headers, page numbers
  and stray marks.
- Accuracy below confidence 84 was not measured against proofread text. It was judged by reading
  samples (`tdl_pilot/ocr_samples.txt`): roughly one word in eight is wrong in the 1958 and 1969
  books.

So the idea works for the later books only, with a confidence gate. Older print fails the
"no corrupted text" rule with this OCR engine.

## What the library holds

- The nationalised-books category (நாட்டுடைமை நூல்) lists 5,079 books; 5,073 were collected
  into `tdl_pilot/tdl_nationalised_list.tsv` (id, title, author, year, publisher, page URL).
  474 authors.
- 3,364 titles match a Tamil Wikisource index file exactly. Wikisource holds these as scans; only
  the proofread pages are in our source 1.
- 228 titles match the Thamizh Mann collection exactly, and about 241 more look like volumes of
  its collected-works series (source 2).
- About 1,324 titles match neither. Of these, 536 were printed in 1979 or later, 58 in 1970–78,
  508 before 1970, and 222 have no year.
- Title matching is exact, so the true overlap is higher. Three of the pilot's "new" books turned
  out to be Thamizh Mann series volumes.
- The category also contains translated works (16 of the unmatched titles have a Latin-script
  author or title) and textbook-corporation books; these need the same removal as in the other
  sources.

## Pilot method

- 20 books chosen at random from the unmatched titles, four per period (before 1950, 1950s,
  1960–79, 1980–99, 2000 on), one book per author, fixed random seed. One had no PDF on the site,
  leaving 19.
- Each PDF was downloaded, every page rendered at 300 dpi and read with Tesseract 5.5.4, Tamil
  model, with Sauvola binarisation (`thresholding_method=2`). The default binarisation returned
  nothing on grey or brown scans (three books came out empty in the first run).
- 3,556 pages, 531,035 Tamil words. OCR took 1 to 48 seconds per book on the laptop.
- Scripts: `scripts/tdl_pilot.py` (download and OCR), `scripts/tdl_pilot_report.py` (per-book
  table), `scripts/tdl_bench.py` (accuracy against Wikisource).

## Per-book results

`conf` is the mean Tesseract word confidence over pages with text. `low` is the share of pages
under 80. Full table: `tdl_pilot/pilot_book_stats.tsv`.

| Printed | Book id | Pages | Tamil words | conf | low pages | Verdict from reading |
|---|---|---|---|---|---|---|
| 1934 | 3214 | 53 | 7,527 | 71.4 | 98% | corrupted |
| 1934 | 61924 | 82 | 13,321 | 63.4 | 100% | corrupted |
| 1944 | 37922 | 132 | 16,508 | 82.6 | 11% | verse, some errors |
| 1955 | 2180 | 33 | 1,142 | 78.2 | 53% | noisy |
| 1956 | 26567 | 47 | 6,882 | 81.4 | 23% | some errors |
| 1958 | 17548 | 353 | 58,862 | 76.7 | 83% | noisy |
| 1958 | 5228 | 105 | 11,710 | 79.0 | 47% | noisy |
| 1961 | 10244 | 97 | 13,661 | 76.8 | 60% | some errors |
| 1969 | 61847 | 477 | 76,608 | 75.1 | 91% | noisy |
| 1979 | 33960 | 141 | 14,613 | 85.3 | 11% | clean |
| 1979 | 40288 | 202 | 24,936 | 85.0 | 3% | clean |
| 1983 | 3423 | 74 | 7,594 | 89.6 | 3% | clean |
| 1983 | 42756 | 122 | 18,535 | 84.2 | 13% | clean |
| 1991 | 1862 | 135 | 15,975 | 83.4 | 20% | mostly clean |
| 1997 | 27960 | 36 | 4,328 | 87.4 | 6% | clean |
| 2001 | 8160 | 305 | 50,849 | 89.5 | 2% | clean |
| 2002 | 27034 | 232 | 24,253 | 89.4 | 8% | clean |
| 2009 | 48943 | 521 | 102,763 | 89.8 | 1% | clean |
| 2018 | 49554 | 409 | 60,968 | 88.9 | 1% | clean |

By words: 58% of the pilot text is from books at confidence 84 or more, 38% from 75 to 84, and
4% below 75.

## Defects that remain in "clean" books

- A zero-width character after every final consonant mark; removed by a simple rule.
- Words broken across lines without a hyphen. They need re-joining; a wrong join or a missed join
  leaves a broken word.
- Running headers, page numbers and footnote marks mixed into the text.
- About 1 word in 28 still wrong (from the benchmark book).
- Tesseract's Tamil model outputs Tamil script only, so any English passage becomes garbage
  Tamil at low confidence. A page-level confidence filter removes most of it.

## Size estimate (rough)

- New titles printed 1979 or later: 536 books. The clean pilot books that are not Thamizh Mann
  volumes average about 16,000 words, which gives roughly 8 to 9 million words.
- Wikisource-indexed titles printed 1979 or later: 1,900 books. The share of their pages that is
  not yet proofread on Wikisource was not measured, so no word estimate is given. This is likely
  the larger gain.
- These are estimates from 19 books and one benchmark book.

## Licence

- A Tamil Virtual Academy letter filed on Wikimedia Commons declares the nationalised books
  released under CC0. The letter itself was not read in this pilot; it was found through search.
- The site footer says "All rights reserved by Tamil Virtual Academy". The CC0 statement covers
  the nationalised books, not the rest of the library.
- "Published before 1966" is not a valid public-domain test in India (the term is the author's
  life plus 60 years). Selection must be by the nationalised list, not by year.

## Open points

- Only one benchmark book; a second one (id 36457) turned out to be a different edition from the
  Wikisource scan.
- A better OCR engine could rescue the older books. Not tested.
- Unproofread Wikisource pages already carry OCR text from another engine. Its quality was not
  compared with this pilot.
- The pilot PDFs (885 MB) and OCR text are in `data/tdl_pilot/` on the laptop only.

## Follow-up, same day

**Unproofread pages on Wikisource** (`tdl_pilot/ws_proofread_status.tsv`, from the Wikisource
`page` and `page_props` dumps). Tamil Wikisource has 393,627 pages not yet proofread and
216,683 proofread or validated. Of the library's nationalised books, 2,449 match a Wikisource
index that has pages:

| Printed | Books | Pages not proofread | Pages proofread | Fully proofread books |
|---|---|---|---|---|
| 1979 or later | 1,431 | 156,836 | 131,669 | 503 |
| before 1979 | 902 | 82,923 | 37,918 | 261 |
| year unknown | 116 | 9,466 | 5,295 | 31 |

At about 130 to 150 words a page, the 156,836 unproofread pages of the later books are roughly
20 to 23 million words.

**Wikisource's own OCR text is worse than ours.** The unproofread pages hold text a bot
uploaded from Google OCR. On 40 proofread pages of the benchmark book, that first-revision text
had 92.4% of words intact (1.13% wrong letters); Tesseract in this pilot had 96.4% (0.44%).
One book, 40 pages.

**Cleaning script** (`scripts/tdl_clean.py`): book gate at confidence 84, page gate at 80,
header and page-number removal, line filter, re-joining of words split across lines using the
run's own vocabulary. On the benchmark book, 86.6% of the cleaned words match a whole word of
the proofread text and 90.0% of the proofread words are found whole. The re-joining is weak on
a small vocabulary and should improve on the full run.

**A caution on the gate.** One pilot book at confidence 85 (id 40288) shows a systematic
letter confusion (க read as ச) when read closely. A stricter gate of 87 keeps 6 of the 19 pilot
books. The gate value should be set after reading samples from the full run.

**Full run** started 23 Nov 2026 on the laptop: 1,775 books into `data/tdl_full/`
(list: `data/tdl_pilot/final_run_books.json`), four parallel workers.

| Group | Books |
|---|---|
| Run: new titles, printed 1979 or later | 453 |
| Run: new titles, year unknown | 207 |
| Run: on Wikisource, not fully proofread, 1979 or later | 998 |
| Run: on Wikisource, not fully proofread, year unknown | 117 |
| Skipped: printed before 1979 | 1,811 |
| Skipped: in the Thamizh Mann collection | 573 |
| Skipped: fully proofread on Wikisource | 492 |
| Skipped: no direct PDF link found on the book page | 363 |
| Skipped: Latin-script title or author | 55 |
| Skipped: dictionaries | 4 |

Pages already proofread on Wikisource must be removed from the OCR text afterwards, so that
source 1 keeps the proofread version and nothing is counted twice. Not done yet.

**Moved to Hopper, 23 Nov 2026, about 23:55.** The laptop run was stopped at 156 books and its
OCR text copied to `/scratch/pgopalan/tamil_base_model/data/tdl_full/`. The rest runs as Slurm
job 3884302 (`scripts/tdl_full_hopper.slurm`: `normal` partition, 16 cores, 2 days, four
parallel workers; resumable by resubmitting). Tesseract 5.5.4 is installed with micromamba in
`tools/ocr-env`, with the laptop's `tam.traineddata` in `tools/tessdata`. Two pilot books
re-read on Hopper gave the same confidence as on the laptop (89.6 and 87.3 against 89.6 and
87.4). The laptop still holds 156 books of OCR text and about 25 GB of PDFs in `data/tdl_full/`.

## Full run result, 24 Nov 2026

The Hopper job finished in 10 h 20 min. 1,766 of the 1,775 books were read (345,675 pages,
49.6 million Tamil words before cleaning). Nine were not: seven have no PDF link, one PDF is
damaged (id 41741) and one download timed out (id 41607).

| Step | Gate 84 | Gate 87 |
|---|---|---|
| Books passing the confidence gate | 1,088 | 742 |
| Tamil words before cleaning | 32.2 million | 22.9 million |
| After cleaning (`tdl_clean.py`) | 29.1 million | 21.2 million |
| After removing pages proofread on Wikisource | 27.6 million | 20.1 million |

- Proofread pages are listed by `scripts/ws_proofread_pages.py` (129,168 pages) and left out
  with `tdl_clean.py --skip-pages`. This assumes the Wikisource page number is the page number
  in the library's PDF. Page counts agree within 2 pages for 609 of 796 matched books, so the
  removal is approximate. 382 volumes of multi-volume series could not be matched to the right
  Wikisource volume and are not deduplicated. A text-level duplicate check against source 1 is
  still needed when source 1 is downloaded.
- Twelve random passages were read, three per confidence band (84–85, 85–87, 87–89, 89+). Every
  band shows the same kinds of error at roughly one word in 20 to 30: க/ச confusion, இ for தி
  or சி at word start, a Tamil numeral for ந, words split by a space. Twelve passages are too
  few to show a difference between bands.
- Text by band at gate 84, after cleaning: 84–85 2.5 million words, 85–87 4.8 million,
  87–89 5.9 million, 89 and above 14.3 million.
- Cleaned text is on the laptop only, in `data/tdl_full_hopper/dedup84` and `dedup87`. Raw OCR
  text and 104 GB of PDFs are on Hopper in `data/tdl_full/`.

## Better OCR model, 24 Nov 2026

The stock Tesseract Tamil model was the main cause of the errors. A benchmark was built from
proofread Wikisource pages (`scripts/ocr_bench_build.py`: 28 books from the full run, spread
over all confidence levels, 12 proofread pages each; 289 pages could be matched to a PDF page)
and run on Hopper (`scripts/ocr_bench_run.py`, `ocr_bench_score.py`, job 3889219).

| Model | Words exactly as proofread | Real letter mistakes (share of words) |
|---|---|---|
| Stock `tam`, fast (used in the full run) | 90.9% | 6.5% |
| Stock `tam`, best | 90.8% | not split |
| `tam_v3.1` | 95.5% | 2.3% |

- `tam_v3.1` is a Tesseract model fine-tuned on 174,624 real scanned lines from proofread
  Tamil Wikisource books (github.com/newsbaba24x7/tamil-tess-trainer, Apache-2.0). One of the
  28 benchmark books appears in its training lists.
- The remaining 2.2% of words have the right letters but different spacing from the proofread
  text. Figures exclude book 47991, whose Wikisource scan does not match the library PDF.
- Books that scored below confidence 82 with the stock model reach 94.3% of words exact with
  `tam_v3.1`, so the old confidence gate will be set again on the new run.
- Line filter with `tam_v3.1`: keeping lines whose lowest word confidence is 80 or more keeps
  89% of lines, and 95.9% of those have every word right (93.5% without the filter).
- Not tested: Surya 2.5 (a generative model; its published Indic scores are low), Google
  Document AI (paid), further fine-tuning on the library's own scans, books printed before 1979.

The full run is being repeated on Hopper with `tam_v3.1` over the PDFs already downloaded
(job 3889280, `scripts/tdl_v31_hopper.slurm`, output `data/tdl_v31/`, with per-word confidence
kept in `conf/`).

## Result of the second run (`tam_v3.1`), 24 Nov 2026

Job 3889280 finished in 4 h 10 min: 1,767 books, 345,793 pages, 48.6 million Tamil words before
cleaning. One book failed (41741, damaged PDF).

On the 28 benchmark books the new model's book confidence does not predict the error rate (a
book at 52 had 1.8% wrong words, one at 92 had 6.5%). The book gate is therefore kept loose
(80, which drops 49 unreadable books) and a line filter is added to `tdl_clean.py`
(`--line-conf`): a line holding a word under the given confidence is dropped and the paragraph
is ended there.

| Cleaning step | Tamil words |
|---|---|
| Book gate 80, cleaned | 42.7 million |
| and pages proofread on Wikisource removed | 40.5 million |
| and line filter at 80 | 33.1 million |
| and line filter at 90 | 24.8 million |

- Without the line filter about 1 word in 44 has a real letter mistake (benchmark). With the
  filter at 80, 95.9% of kept lines have every word right, against 93.5% without it.
- The line filter cuts paragraphs where a line is dropped, so dialogue and verse become
  fragmented. This is the cost of the cleaner text.
- The first run gave 27.6 million words at about 1 wrong word in 15 (gate 84).
- Cleaned text is on the laptop in `data/tdl_v31/c_gate80_dedup`, `c_gate80_dedup_line80` and
  `c_gate80_dedup_line90`. Raw text and word confidences are on Hopper in `data/tdl_v31/`.
