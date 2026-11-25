"""Remove text that appears more than once in the cleaned OCR books.

    python tdl_dedup.py <out_dir> <clean_dir> [<clean_dir> ...] [--min-share 0.5] [--hist]

Input: folders written by tdl_clean.py without --join-pages (one file per book, one paragraph
per page, plus clean_stats.tsv and clean_pages.tsv). This script removes nothing: it lists the
duplicate pages, and tdl_clean.py leaves them out when run again with --skip-pages. The library lists some books twice, holds several editions of one work, and
has collected volumes that contain smaller books, so the same pages are read more than once.

Method
  1. Every paragraph is turned into its set of 5-word sequences (Tamil words only).
  2. A sampled index finds pairs of books that share sequences.
  3. Books are ranked: higher OCR confidence first, then more words. For each book, a paragraph
     is dropped when at least --min-share of its sequences occur in a higher-ranked book that
     shares text with it, or earlier in the same book. Two scans of one page differ by OCR
     mistakes, so the match is by share, not exact.
Writes <out_dir>/dup_pages.tsv (book, page) and <out_dir>/dedup_stats.tsv. --hist only prints
how the share is distributed and writes nothing.
"""
import argparse, collections, glob, os, re, zlib
import numpy as np

TAW = re.compile(r"[஀-௿]+")
K, MOD, MIN_SHARED, MIN_SEQ = 5, 8, 5, 20

def para_hashes(p):
    w = TAW.findall(p)
    return np.fromiter((zlib.crc32(" ".join(w[i:i + K]).encode()) for i in range(len(w) - K + 1)),
                       dtype=np.uint32, count=max(0, len(w) - K + 1))

def share(h, ref):
    """Share of the hashes h found in the sorted array ref."""
    if len(h) < MIN_SEQ or not len(ref): return 0.0
    i = np.searchsorted(ref, h); i[i == len(ref)] = 0
    return float((ref[i] == h).mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("dirs", nargs="+")
    ap.add_argument("--min-share", type=float, default=0.5)
    ap.add_argument("--hist", action="store_true")
    a = ap.parse_args()
    conf, paras, hashes, src, pnum = {}, {}, {}, {}, {}
    for d in a.dirs:
        for l in list(open(f"{d}/clean_stats.tsv"))[1:]:
            r = l.split("\t")
            if r[5].strip() == "kept": conf[r[0]] = float(r[1])
        for l in open(f"{d}/clean_pages.tsv"):
            b, m = l.rstrip("\n").split("\t")
            pnum[b] = m.split(",") if m else []
        for f in glob.glob(f"{d}/*.txt"):
            b = os.path.basename(f)[:-4]
            assert b not in paras, f"book {b} is in two folders"
            paras[b] = [p for p in open(f, encoding="utf-8").read().split("\n\n") if p.strip()]
            assert len(paras[b]) == len(pnum[b]), f"book {b}: paragraphs and page list differ"
            hashes[b] = [para_hashes(p) for p in paras[b]]
            src[b] = d
    words = {b: sum(len(TAW.findall(p)) for p in ps) for b, ps in paras.items()}
    rank = {b: i for i, b in enumerate(sorted(paras, key=lambda b: (-conf.get(b, 0), -words[b], b)))}
    allh = {b: np.unique(np.concatenate(h)) if h else np.zeros(0, np.uint32) for b, h in hashes.items()}

    inv = collections.defaultdict(list)                       # sampled index of shared sequences
    for b, h in allh.items():
        for x in h[h % MOD == 0].tolist(): inv[x].append(b)
    pair = collections.Counter()
    for bl in inv.values():
        if 1 < len(bl) <= 30:
            bl.sort()
            for i in range(len(bl)):
                for j in range(i + 1, len(bl)): pair[(bl[i], bl[j])] += 1
    senior = collections.defaultdict(list)
    for (x, y), c in pair.items():
        if c >= MIN_SHARED:
            lo, hi = (x, y) if rank[x] < rank[y] else (y, x)
            senior[hi].append(lo)

    hist = collections.Counter(); rows = []; dup = []
    if not a.hist: os.makedirs(a.out, exist_ok=True)
    for b in sorted(paras, key=rank.get):
        ref = np.unique(np.concatenate([allh[s] for s in senior[b]])) if senior[b] else np.zeros(0, np.uint32)
        seen = np.zeros(0, np.uint32); keep = []; d_other = d_self = w_drop = 0
        for p, h, pn in zip(paras[b], hashes[b], pnum[b]):
            so, ss = share(h, ref), share(h, seen)
            if senior[b] and len(h) >= MIN_SEQ: hist[min(int(so * 10), 9)] += 1
            if so >= a.min_share: d_other += 1; w_drop += len(TAW.findall(p)); dup.append((b, pn))
            elif ss >= a.min_share: d_self += 1; w_drop += len(TAW.findall(p)); dup.append((b, pn))
            else: keep.append(p)
            seen = np.union1d(seen, h)
        kw = words[b] - w_drop
        rows.append((b, os.path.basename(os.path.dirname(src[b] + "/")), f"{conf.get(b, 0):.1f}", len(paras[b]),
                     len(keep), d_other, d_self, words[b], kw))
    if a.hist:
        tot = sum(hist.values())
        print("paragraphs in books that share text with a higher-ranked book:", tot)
        for i in range(10): print(f"  share {i*10:3d}-{i*10+10:3d}%  {hist[i]:7d}")
    else:
        with open(f"{a.out}/dup_pages.tsv", "w") as o:
            for b, pn in dup: o.write(f"{b}\t{pn}\n")
        with open(f"{a.out}/dedup_stats.tsv", "w") as o:
            o.write("book\tsource\tconf\tparas\tparas_kept\tdropped_other_book\tdropped_same_book\twords\twords_kept\n")
            for r in rows: o.write("\t".join(map(str, r)) + "\n")
    print("books", len(rows), "with text left", sum(1 for r in rows if r[4]),
          "words before", sum(r[7] for r in rows), "after", sum(r[8] for r in rows))
    print("paragraphs dropped: in another book", sum(r[5] for r in rows), "same book", sum(r[6] for r in rows))

if __name__ == "__main__":
    main()
