"""Remove text that appears more than once across all cleaned Tamil sources.

    python corpus_dedup.py <out_dir> --typed name=dir [name=dir ...] --ocr name=dir [name=dir ...]
                           [--min-share 0.5]

--typed  cleaned sources that were typed or published as text (Wikisource, Thamizh Mann, ebooks,
         Kaniyam, the small wikis), in order of priority: when two sources hold the same text,
         the copy in the source named first is kept.
--ocr    page-level folders written by tdl_clean.py without --join-pages (one paragraph per
         page, with clean_stats.tsv and clean_pages.tsv). OCR copies rank below every typed
         source, and among themselves by OCR confidence.

Method (the same rule as for the OCR books alone)
  1. Each document is cut into units: a page for OCR books; for typed text, runs of paragraphs
     of at least 60 words. A unit is described by its set of 5-word sequences.
  2. A sampled index finds the pairs of documents that share sequences.
  3. A unit is dropped when at least --min-share of its sequences occur in higher-ranked
     documents that share text with its document, or earlier in the same document. Two copies
     of a text never match exactly (OCR mistakes, different editions), hence a share.
Output
  <out_dir>/<name>/<doc>.txt   the typed sources with duplicate units removed
  <out_dir>/ocr_dup_pages.tsv  (book, page) of duplicate OCR pages, for tdl_clean.py --skip-pages
  <out_dir>/dedup_stats.tsv    one row per document
"""
import argparse, collections, glob, os, re, zlib
import numpy as np

TAW = re.compile(r"[஀-௿]+")
K, MOD, MIN_SHARED, MIN_SEQ, UNIT_WORDS = 5, 8, 5, 20, 60
EMPTY = np.zeros(0, np.uint32)

def hashes(words):
    n = len(words) - K + 1
    if n <= 0: return EMPTY
    return np.fromiter((zlib.crc32(" ".join(words[i:i + K]).encode()) for i in range(n)), dtype=np.uint32, count=n)

def share(h, ref):
    if len(h) < MIN_SEQ or not len(ref): return 0.0
    i = np.searchsorted(ref, h); i[i == len(ref)] = 0
    return float((ref[i] == h).mean())

class Doc:
    __slots__ = ("src", "name", "rank", "paras", "units", "uh", "allh", "words", "pages", "ocr")

def load_typed(src, d, docs, order):
    for f in sorted(glob.glob(f"{d}/*.txt")):
        x = Doc(); x.src, x.name, x.ocr, x.pages = src, os.path.basename(f)[:-4], False, None
        x.paras = [p for p in open(f, encoding="utf-8").read().split("\n\n") if p.strip()]
        x.units, cur, n = [], [], 0                 # units: lists of paragraph indexes
        for i, p in enumerate(x.paras):
            cur.append(i); n += len(TAW.findall(p))
            if n >= UNIT_WORDS: x.units.append(cur); cur, n = [], 0
        if cur:
            if x.units: x.units[-1] += cur
            else: x.units.append(cur)
        x.rank = (order, 0.0, x.name)
        docs.append(x)

def load_ocr(src, d, docs, order):
    conf = {}
    for l in list(open(f"{d}/clean_stats.tsv", encoding="utf-8"))[1:]:
        r = l.rstrip("\n").split("\t")
        if r[5] == "kept": conf[r[0]] = float(r[1])
    pages = {}
    for l in open(f"{d}/clean_pages.tsv", encoding="utf-8"):
        b, m = l.rstrip("\n").split("\t"); pages[b] = m.split(",") if m else []
    for f in sorted(glob.glob(f"{d}/*.txt")):
        x = Doc(); x.src, x.name, x.ocr = src, os.path.basename(f)[:-4], True
        x.paras = [p for p in open(f, encoding="utf-8").read().split("\n\n") if p.strip()]
        x.pages = pages[x.name]
        assert len(x.pages) == len(x.paras), f"{src} {x.name}: paragraphs and page list differ"
        x.units = [[i] for i in range(len(x.paras))]
        x.rank = (order, -conf.get(x.name, 0.0), x.name)
        docs.append(x)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--typed", nargs="*", default=[]); ap.add_argument("--ocr", nargs="*", default=[])
    ap.add_argument("--min-share", type=float, default=0.5)
    a = ap.parse_args()
    docs = []
    for o, s in enumerate(a.typed): load_typed(*s.split("=", 1), docs, o)
    for s in a.ocr: load_ocr(*s.split("=", 1), docs, len(a.typed))        # all OCR sets share one rank level
    for x in docs:
        x.uh = [hashes([w for i in u for w in TAW.findall(x.paras[i])]) for u in x.units]
        x.allh = np.unique(np.concatenate(x.uh)) if x.uh else EMPTY
        x.words = [sum(len(TAW.findall(x.paras[i])) for i in u) for u in x.units]
    docs.sort(key=lambda x: x.rank)
    print("documents", len(docs), "words", sum(sum(x.words) for x in docs), flush=True)

    inv = collections.defaultdict(list)             # sampled index of shared sequences
    for n, x in enumerate(docs):
        for h in x.allh[x.allh % MOD == 0].tolist(): inv[h].append(n)
    pair = collections.Counter()
    for bl in inv.values():
        if 1 < len(bl) <= 40:
            for i in range(len(bl)):
                for j in range(i + 1, len(bl)): pair[(bl[i], bl[j])] += 1    # bl is in rank order
    del inv
    senior = collections.defaultdict(list)
    for (lo, hi), c in pair.items():
        if c >= MIN_SHARED: senior[hi].append(lo)
    print("document pairs that share text:", sum(len(v) for v in senior.values()), flush=True)

    os.makedirs(a.out, exist_ok=True)
    tot = collections.defaultdict(collections.Counter); cross = collections.Counter()
    with open(f"{a.out}/dedup_stats.tsv", "w", encoding="utf-8") as rep, \
         open(f"{a.out}/ocr_dup_pages.tsv", "w", encoding="utf-8") as dup:
        rep.write("source\tdoc\tunits\tunits_dropped_other\tunits_dropped_same\twords\twords_kept\tmost_shared_with\n")
        for n, x in enumerate(docs):
            sen = senior.get(n, [])
            ref = np.unique(np.concatenate([docs[s].allh for s in sen])) if sen else EMPTY
            seen = EMPTY; keep = []; d_other = d_same = wkept = 0; lost_h = []
            for u, h, w in zip(x.units, x.uh, x.words):
                so, ss = share(h, ref), share(h, seen)
                if so >= a.min_share:
                    d_other += 1; lost_h.append(h)
                    if x.ocr: dup.write(f"{x.name}\t{x.pages[u[0]]}\n")
                elif ss >= a.min_share:
                    d_same += 1
                    if x.ocr: dup.write(f"{x.name}\t{x.pages[u[0]]}\n")
                else:
                    keep += u; wkept += w
                seen = np.union1d(seen, h)
            top = ""
            if lost_h:                                # which document holds most of the dropped text
                lh = np.unique(np.concatenate(lost_h))
                best = max(sen, key=lambda s: int(np.isin(lh, docs[s].allh, assume_unique=True).sum()))
                top = f"{docs[best].src}/{docs[best].name}"
                cross[(x.src, docs[best].src)] += sum(x.words) - wkept
            if not x.ocr and keep:
                os.makedirs(f"{a.out}/{x.src}", exist_ok=True)
                with open(f"{a.out}/{x.src}/{x.name}.txt", "w", encoding="utf-8") as o:
                    o.write("\n\n".join(x.paras[i] for i in keep) + "\n")
            t = tot[x.src]; t["docs"] += 1; t["docs_left"] += bool(keep); t["words"] += sum(x.words); t["words_kept"] += wkept
            rep.write(f"{x.src}\t{x.name}\t{len(x.units)}\t{d_other}\t{d_same}\t{sum(x.words)}\t{wkept}\t{top}\n")
    print("source\tdocs\tdocs_left\twords\twords_kept")
    for s, t in tot.items(): print(f"{s}\t{t['docs']}\t{t['docs_left']}\t{t['words']}\t{t['words_kept']}")
    print("words dropped, by (source of the dropped copy, source of the kept copy):")
    for (s1, s2), w in cross.most_common(40): print(f"  {s1}\t<- {s2}\t{w}")

if __name__ == "__main__":
    main()
