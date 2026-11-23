# Final list of Tamil sources

23 November 2026. Result of three search rounds on and off Hugging Face. Every source here
meets all the rules after the listed removals: original Tamil in Tamil script (English brand
names and digits allowed), no machine-translated or synthetic text, an open licence from the
authors, publisher or government, no corrupted text, no social media or Tanglish, documented.

Details and everything that was rejected are in `strict_shortlist.md`. The items to drop are
listed in `exclusions/`.

| # | Source | Text | Words after cleaning (millions) | Removed before use | Machine-translation risk left | Licence | Link |
|---|---|---|---|---|---|---|---|
| 1 | Tamil Wikisource, proofread and validated pages | Printed books typed by volunteers: 20th-century prose, some verse, encyclopedias | about 33 | Unproofread OCR pages; 28 human-translated books, Bible, Quran; wiki markup | None found; print books | CC BY-SA 4.0; many books nationalised (public domain) | [site](https://ta.wikisource.org/) · [dump](https://dumps.wikimedia.org/tawikisource/latest/) |
| 2 | Thamizh Mann nationalised books (University of Toronto Scarborough) | 575 whole books of 20th-century scholarship and prose | about 27 (estimate) | 18 human translations, 41 dictionaries, 5 English volumes | None found; print books | CC BY-SA 4.0 | [files](https://github.com/Digital-Tamil-Studies/open_tamil_texts/tree/master/collections/thamizh_mann) |
| 3 | Tamil Wikinews | Volunteer-written news, mostly 2009–2014 | 0.5–0.8 | Wiki markup | No translation-tool articles found | CC BY 4.0 | [site](https://ta.wikinews.org/) |
| 4 | Tamil Wikibooks and Wikiquote | Textbooks, quotations | about 0.5 | Wiki markup | No translation-tool articles found; some quotations are hand-translated | CC BY-SA 4.0 | [Wikibooks](https://ta.wikibooks.org/) · [Wikiquote](https://ta.wikiquote.org/) |
| 5 | Kaniyam magazine | Technology articles, 2012 to now | about 0.6 | All posts by one author suspected of machine-assisted translation; translated posts; English-heavy posts | Low: suspected posts removed | Creative Commons, attribution and share-alike | [site](https://www.kaniyam.com/) |
| 6 | FreeTamilEbooks, open-licensed books | About 378 books: fiction, essays, non-fiction | unknown | Non-commercial and no-derivatives books; 18 human translations; OCR-damaged books | None found in descriptions; book bodies not yet read | CC0, public domain, CC BY or CC BY-SA per book | [site](https://freetamilebooks.com/) |
| 7 | Tamil Wikipedia, cleaned | About 122,000 encyclopedia articles | about 34 | 65,051 listed articles made with translation tools or by bots, plus a first-revision tag check; templated stubs; non-Tamil lines | Small: unlabelled pasted translations cannot be detected | CC BY-SA 4.0 | [site](https://ta.wikipedia.org/) · [dump](https://dumps.wikimedia.org/tawiki/latest/) |

## Totals

- Sources 1 to 6: about 62 million words plus the ebooks. Almost all of it is printed books, with
  no machine translation found.
- With source 7 (cleaned Wikipedia): about 96 million words plus the ebooks.

Sources 1 and 2 both contain nationalised books, so some text is counted twice until duplicates
are removed. Sizes are estimates; exact counts come after download and filtering.

## Source 7 is a choice

Tamil Wikipedia is included only after removing everything Wikipedia itself labels as
tool-translated or bot-made. Text that an editor pasted from a translator without using the
tool has no label. To have no machine-translation risk at all, leave source 7 out.

## Not in this list

- Large web collections (Sangraha verified, HPLT 3.0, WMT News Crawl): their licence comes from
  the collector, and they contain some machine-translated pages. Kept separate.
- Project Madurai: no formal licence.
- Everything else examined failed at least one rule; see `strict_shortlist.md`.
