"""Clean source 1, Tamil Wikisource: proofread and validated pages to plain text, one file per book.

    python clean_wikisource.py <tawikisource-...-pages-articles.xml.bz2> <out_dir>
                               <tawikisource_translation_flags.txt> [--min-tamil 0.8]

Only pages of the Page namespace with quality level 3 (proofread) or 4 (validated) are used;
pages that are still raw OCR (levels 0 to 2) and text typed directly into the main namespace
are left out. Pages are grouped by their index file (the scanned book) and put in page order.
  - The page header and footer (running title, page number) are removed.
  - Wiki markup is removed (wikitext.py); footnotes, tables, contents lists and images go.
  - As on Wikisource itself, text runs on from one page to the next unless the page ends
    with a paragraph break or the next page is not proofread.
  - Paragraphs that are not mostly Tamil are dropped.
Dropped books: index files in the translation list, index files transcluded by the main-
namespace pages in that list (Bible, Quran), and books left with under 100 Tamil words.
Writes <out_dir>/ws_NNNNN.txt, <out_dir>/clean_stats.tsv and <out_dir>/dropped_templates.tsv.
"""
import argparse, collections, os, re
import wikitext
from textutil import is_tamil, nwords, tidy

A, Z = "\x02", "\x03"
HYPH = re.compile(r"[\u0B80-\u0BFF]-$")
TAMIL1 = re.compile(r"[\u0B80-\u0BFF]")
NOINC = re.compile(r"<noinclude\b[^>]*>.*?</noinclude\s*>", re.S | re.I)
QUAL = re.compile(r'<pagequality level="(\d)"')
# an image at the very top or bottom of a page does not end the paragraph that runs across the page break
_IMG = r"(?:\{\{\s*(?:css[ _]image[ _]crop|raw[ _]image)[^{}]*\}\}|\[\[\s*(?:படிமம்|கோப்பு|file|image)\s*:[^\[\]]*\]\])"
EDGE_S = re.compile(r"^(?:\s*" + _IMG + r")+\s*", re.I)
EDGE_E = re.compile(r"(?:\s*" + _IMG + r")+\s*$", re.I)
PAGES_TAG = re.compile(r"<pages\b[^>]*?\bindex\s*=\s*(\"[^\"]*\"|'[^']*'|[^\s/>]+)", re.I)

def norm(t):
    return re.sub(r"\s+", " ", t.replace("_", " ")).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump"); ap.add_argument("out"); ap.add_argument("flags")
    ap.add_argument("--min-tamil", type=float, default=0.8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    flag_index, flag_ns0, sec = set(), set(), 0
    for l in open(a.flags, encoding="utf-8"):
        l = l.strip()
        if l.startswith("#"): sec += 1; continue
        if l: (flag_ns0 if sec == 2 else flag_index).add(norm(l.split(":", 1)[1] if re.match(r"^(அட்டவணை|Index)\s*:", l) else l))
    books = collections.defaultdict(dict); levels = collections.Counter(); lost = collections.Counter()
    for pid, title, ns, text, red in wikitext.dump_pages(a.dump):
        if ns == 0:
            if norm(title) in flag_ns0 or norm(title.split("/")[0]) in flag_ns0:
                for m in PAGES_TAG.finditer(text): flag_index.add(norm(m.group(1).strip("\"'")))
            continue
        if ns != 250: continue
        m = QUAL.search(text); lv = m.group(1) if m else "none"; levels[lv] += 1
        if lv not in ("3", "4"): continue
        name = title.split(":", 1)[1]
        idx, _, num = name.rpartition("/")
        if not num.isdigit(): idx, num = name, "0"
        body = EDGE_E.sub("", EDGE_S.sub("", NOINC.sub("", text).replace("\ufeff", "").strip()))
        if not body: continue
        books[norm(idx)][int(num)] = (lv, wikitext.strip(A + body + Z, lost))
    tot = collections.Counter(); n = 0
    with open(f"{a.out}/clean_stats.tsv", "w", encoding="utf-8") as rep:
        rep.write("file\tstatus\tpages\tvalidated_pages\tparas\twords\tindex\n")
        for idx in sorted(books):
            pages = books[idx]; n += 1; fid = f"ws_{n:05d}"
            paras, prev, open_end = [], None, False
            for num in sorted(pages):
                ps = [p for p in pages[num][1]]
                cont = open_end and prev == num - 1 and ps and ps[0].startswith(A) and ps[0] != A
                open_end = bool(ps) and ps[-1].endswith(Z) and ps[-1] != Z
                ps = [tidy(p.replace(A, "").replace(Z, "")) if "\n" not in p else
                      "\n".join(x for x in (tidy(l.replace(A, "").replace(Z, "")) for l in p.split("\n")) if x) for p in ps]
                ps = [p for p in ps if p]
                if cont and paras and ps and "\n" not in paras[-1] and "\n" not in ps[0]:
                    if HYPH.search(paras[-1]) and TAMIL1.match(ps[0]):      # word split by the page break
                        paras[-1] = paras[-1][:-1] + ps[0]
                    else:
                        paras[-1] += " " + ps[0]
                    ps = ps[1:]
                paras += ps; prev = num
            kept = [p for p in paras if is_tamil(p, a.min_tamil)]
            words = nwords("\n".join(kept))
            if idx in flag_index or "மொழிபெயர்ப்பு" in idx or "மொழிப்பெயர்ப்பு" in idx: status = "dropped: translation"
            elif words < 100: status = "dropped: under 100 Tamil words"
            else:
                status = "kept"
                open(f"{a.out}/{fid}.txt", "w", encoding="utf-8").write("\n\n".join(kept) + "\n")
            tot[status] += 1; tot["words " + status] += words; tot["pages " + status] += len(pages)
            rep.write(f"{fid}\t{status}\t{len(pages)}\t{sum(1 for v in pages.values() if v[0] == '4')}\t"
                      f"{len(kept)}\t{words}\t{idx}\n")
    with open(f"{a.out}/dropped_templates.tsv", "w", encoding="utf-8") as o:
        for k, v in lost.most_common(): o.write(f"{v}\t{k}\n")
    print("page quality levels:", dict(levels))
    for k, v in sorted(tot.items()): print(k, v)
    print("Tamil words inside dropped templates:", sum(lost.values()), lost.most_common(25))

if __name__ == "__main__":
    main()
