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
