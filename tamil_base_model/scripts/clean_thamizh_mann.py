"""Clean source 2, Thamizh Mann nationalised books: publisher's text files to clean Tamil text.

    python clean_thamizh_mann.py <raw/thamizh_mann> <out_dir> <thamizh_mann_flagged.tsv>
                                 [--min-tamil 0.8] [--no-rejoin] [--joins-out f.tsv]

Dropped works: those in the flagged list (translations, dictionaries, English volumes).
Inside a kept work
  - contents rows and table rows (lines whose cells are separated by tabs) are dropped, but a
    tab used as indentation or after a list number is only removed; lines that are mostly
    digits and lines that are not mostly Tamil are dropped. The last rule also removes the
    few works stored in an old font encoding, which read as Latin gibberish;
  - the title-page lines before the first real paragraph are dropped;
  - paragraphs found in three or more works (the publisher's preface, notices) are dropped;
  - footnote numbers stuck to the end of a word are removed;
  - words that the typesetting file split in two ("தொடங் கினார்"), inside a line or at a line
    end, are re-joined. Two neighbouring words are joined only when the joined form is a word
    of the collection and (a) it is common and one half is much rarer than it, or (b) one half
    is a fragment: a form that almost always stands next to a word it completes, or (c) in a
    prose paragraph, the second half begins with a letter no Tamil word begins with (ங ண ழ ள ற
    ன, or a bare consonant). Real word pairs and the metre of verse stay as they are.
Writes <out_dir>/<work id>.txt and <out_dir>/clean_stats.tsv.
"""
import argparse, collections, csv, glob, hashlib, os, re
from textutil import is_tamil, nwords, tidy, TAW

FOOT = re.compile(r"(?<=[஀-௿])\d{1,3}(?=[\s.,;:!?)\"'’”]|$)")
NO_START = set("ஙணழளறன")
PAIR = re.compile(r"([஀-௿]+) (?=([஀-௿]+))")

def key(p):
    return hashlib.md5("".join(TAW.findall(p)).encode()).digest() if nwords(p) >= 8 else None

NUMCELL = re.compile(r"^[\d\s.,:;()\[\]/*+\-–—௦-௲]*$|^\(?[\u0B80-\u0BFF]{1,2}[.)]$")   # 12, (3), அ., ௧௨

def untab(l, tabs):
    """Handle a line that contains tabs. Returns the text to keep, or "" to drop the line.
    Leading tabs are indentation. A leading number cell is a list number. A line that ends in
    a page-number cell or has dot leaders is a contents row. What is left is kept when it is
    one cell of text, or a short label followed by a sentence (a speaker and a speech);
    rows of several short cells (tables, word lists, indexes) are dropped."""
    cells = [c.strip() for c in l.split("\t") if c.strip()]
    if cells and NUMCELL.match(cells[0]): cells = cells[1:]
    if not cells: return ""
    if len(cells) > 1 and (NUMCELL.match(cells[-1]) or re.search(r"\.{3,}|…", l)):
        tabs["contents rows"] += 1; return ""
    if len(cells) == 1:
        tabs["one text cell"] += 1; return cells[0]
    if len(cells) <= 3 and nwords(cells[-1]) >= 6 and all(nwords(c) <= 3 for c in cells[:-1]):
        tabs["label and sentence"] += 1; return " ".join(cells)
    tabs["table rows"] += 1; tabs["table row words"] += nwords(l)
    return ""

def lines_of(path, tabs):
    out = []
    for l in open(path, encoding="utf-8", errors="replace"):
        if "\t" in l.strip(): l = untab(l, tabs)
        l = tidy(FOOT.sub("", l))
        if not l: out.append(""); continue
        digits = sum(c.isdigit() for c in l)
        ok = is_tamil(l, MIN) and digits <= 0.2 * len(l.replace(" ", ""))
        if not ok and nwords(l) == 0 and len(l) > 20: tabs["lines with no Tamil letters"] += 1
        out.append(l if ok else "")
    return out

def main():
    global MIN
    ap = argparse.ArgumentParser()
    ap.add_argument("raw"); ap.add_argument("out"); ap.add_argument("flagged")
    ap.add_argument("--min-tamil", type=float, default=0.8)
    ap.add_argument("--no-rejoin", action="store_true")
    ap.add_argument("--joins-out")
    a = ap.parse_args(); MIN = a.min_tamil
    os.makedirs(a.out, exist_ok=True)
    flagged = {r["identifier_local"]: r["flags"] for r in csv.DictReader(open(a.flagged, encoding="utf-8"), delimiter="\t")}
    meta = {r["identifier_local"]: r for r in csv.DictReader(open(f"{a.raw}/metadata.csv", encoding="utf-8"))}
    docs, seen, vocab = {}, collections.Counter(), collections.Counter()
    tabs = collections.Counter()
    for f in sorted(glob.glob(f"{a.raw}/data/*.txt")):
        wid = os.path.basename(f)[:-4]
        if wid in flagged: continue
        docs[wid] = L = lines_of(f, tabs)
        for k in {key(l) for l in L} - {None}: seen[k] += 1
    for wid, L in docs.items():
        L = [l for l in L if not l or seen.get(key(l), 0) < 3]
        first = next((i for i, l in enumerate(L[:400]) if nwords(l) >= 25), 0)   # none: verse, keep all
        docs[wid] = L = L[first:]
        for l in L: vocab.update(TAW.findall(l))
    # Fragments learned from the collection itself. A word form is a "tail fragment" when, in
    # at least 8 of 10 places where it occurs, gluing it to the word before it gives a word that
    # exists elsewhere ("கினார்", "யில்", "களைத்"); a "head fragment" likewise with the word
    # after it ("தொடங்", "செல்வங்"). Real words do not behave like that.
    t_all, t_hit, h_all, h_hit = (collections.Counter() for _ in range(4))
    for L in docs.values():
        prev = None
        for l in L:
            w = TAW.findall(l) if l else []
            seq = ([prev] if prev and w else []) + w
            for x, y in zip(seq, seq[1:]):
                ok = (x + y) in vocab
                t_all[y] += 1; h_all[x] += 1
                if ok: t_hit[y] += 1; h_hit[x] += 1
            prev = w[-1] if w else None
    tails = {y for y, n in t_all.items() if n >= 20 and t_hit[y] >= 0.8 * n}
    heads = {x for x, n in h_all.items() if n >= 20 and h_hit[x] >= 0.8 * n}
    print("tail fragments", len(tails), "head fragments", len(heads))
    joins = collections.Counter()
    def fragment(x, y, prose):
        """True when x and y are two halves of one word."""
        j = vocab.get(x + y, 0)
        if j < 2 or len(x) < 2 or len(y) < 2: return False
        if j >= 5 and j >= 3 * min(vocab[x], vocab[y]): return True
        if y in tails or x in heads: return True
        # a Tamil word cannot begin with ங ண ழ ள ற ன or with a bare consonant. Verse is cut
        # by metre and may do so on purpose, so this test is used in prose paragraphs only.
        return prose and (y[0] in NO_START or y[1] == "\u0BCD")
    def rejoin(line):
        prose = nwords(line) >= 12
        def f(m):
            x, y = m.group(1), m.group(2)
            if fragment(x, y, prose): joins[(x, y)] += 1; return x
            return m.group(0)
        return PAIR.sub(f, line)
    def rejoin_lines(L):
        """Join words inside each line, and a word cut in two by a line end (the two lines merge)."""
        out = []
        for l in L:
            l = rejoin(l) if l else l
            if l and out and out[-1]:
                ma, mb = re.search(r"([\u0B80-\u0BFF]+)(-?)$", out[-1]), re.match(r"[\u0B80-\u0BFF]+", l)
                if ma and mb and (fragment(ma.group(1), mb.group(0), False) or
                                  (ma.group(2) and vocab.get(ma.group(1) + mb.group(0), 0) >= 2)):
                    joins[(ma.group(1), mb.group(0))] += 1
                    out[-1] = out[-1].rstrip("-") + l; continue
            out.append(l)
        return out
    tot = collections.Counter()
    with open(f"{a.out}/clean_stats.tsv", "w", encoding="utf-8") as rep:
        rep.write("work\tstatus\tlines\twords\ttitle\tauthor\n")
        for wid in sorted(set(docs) | set(flagged)):
            m = meta.get(wid, {}); words = n = 0
            if wid in flagged: status = "dropped: " + flagged[wid]
            else:
                L = docs[wid] if a.no_rejoin else rejoin_lines(docs[wid])
                text = re.sub(r"\n{3,}", "\n\n", "\n".join(L)).strip()
                words = nwords(text); n = sum(1 for l in L if l)
                status = "kept" if words >= 200 else "dropped: almost no Tamil text"
                if status == "kept": open(f"{a.out}/{wid}.txt", "w", encoding="utf-8").write(text + "\n")
            tot[status] += 1; tot["words kept"] += words if status == "kept" else 0
            one = lambda v: re.sub(r"\s+", " ", v or "")
            rep.write(f"{wid}\t{status}\t{n}\t{words}\t{one(m.get('title_tam'))}\t{one(m.get('personal_role_names_tam'))}\n")
    for k, v in sorted(tot.items()): print(k, v)
    print("lines with tabs:", dict(tabs))
    print("word pairs re-joined:", sum(joins.values()), "distinct", len(joins))
    if a.joins_out:
        with open(a.joins_out + ".fragments", "w", encoding="utf-8") as o:
            for y in sorted(tails, key=lambda y: -t_all[y]): o.write(f"tail\t{y}\t{t_all[y]}\t{t_hit[y]}\n")
            for x in sorted(heads, key=lambda x: -h_all[x]): o.write(f"head\t{x}\t{h_all[x]}\t{h_hit[x]}\n")
        with open(a.joins_out, "w", encoding="utf-8") as o:
            for (x, y), c in joins.most_common():
                o.write(f"{c}\t{x} {y}\t{vocab[x]}\t{vocab[y]}\t{vocab[x + y]}\n")

if __name__ == "__main__":
    main()
