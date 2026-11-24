"""Turn the OCR output of tdl_pilot.py into clean text, one file per kept book.

    python tdl_clean.py <ocr_run_dir> <out_dir> [--book-conf 84] [--page-conf 80] [--skip-pages f.tsv]

Steps
  1. Quality gate: keep a book only if its mean Tesseract confidence is at least --book-conf;
     inside a kept book drop pages under --page-conf or with fewer than 200 Tamil letters.
  2. Remove the zero-width character Tesseract writes after a final consonant mark.
  3. Drop running headers and page numbers (short first/last lines containing digits, or a
     first line that repeats on many pages of the book).
  4. Drop lines that are not mostly Tamil letters.
  5. Re-join words split across line ends, using the vocabulary of the whole run: the two
     halves are joined when the joined form is a known word and at least as common as the
     rarer half.
Writes <out_dir>/<book>.txt and <out_dir>/clean_stats.tsv.
"""
import argparse, collections, glob, os, re

TA = re.compile(r"[஀-௿]")
TAW = re.compile(r"[஀-௿]+")
PAGE = re.compile(r"\n=== page (\d+) ===\n(.*?)(?=\n=== page \d+ ===\n|\Z)", re.S)

def load_stats(d):
    s = collections.defaultdict(dict)
    for f in glob.glob(f"{d}/page_stats*.tsv"):
        for l in open(f):
            r = l.rstrip("\n").split("\t")
            s[r[0]][int(r[1])] = (int(r[2]), float(r[5]))
    return s

def page_lines(text, repeated, confs=None, line_conf=0):
    """Clean lines of one page. A line dropped by the confidence filter leaves a None, so that
    the text on either side is not joined into one sentence."""
    lines = [l.strip() for l in text.replace("‌", "").replace("‍", "").split("\n")]
    low = set()
    if confs is not None and line_conf:
        cl = confs.split("\n")
        if len(cl) == len(lines):
            low = {i for i, c in enumerate(cl) if c.strip() and min(map(int, c.split())) < line_conf}
    lines = [(l, i in low) for i, l in enumerate(lines) if l]
    isl = [x[1] for x in lines]; lines = [x[0] for x in lines]
    for end in (0, -1):                              # header / footer
        if lines:
            l = lines[end]; key = re.sub(r"[^஀-௿]", "", l)
            if (len(l) < 60 and re.search(r"\d", l)) or (key and key in repeated) or len(key) < 3:
                lines.pop(end); isl.pop(end)
    out = []
    for l, bad in zip(lines, isl):
        ta = len(TA.findall(l)); other = len(re.findall(r"[^\W\d_]", l)) - ta
        if bad:
            if out and out[-1] is not None: out.append(None)
        elif ta >= 3 and ta >= 0.9 * (ta + other) and ta >= 0.5 * len(l.replace(" ", "")):
            out.append(l)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("out")
    ap.add_argument("--book-conf", type=float, default=84)
    ap.add_argument("--page-conf", type=float, default=80)
    ap.add_argument("--line-conf", type=int, default=0, help="drop lines holding a word under this "
                    "confidence (needs <run>/conf/<book>.txt)")
    ap.add_argument("--skip-pages", help="TSV of book id and page number to leave out "
                    "(pages already proofread on Wikisource)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    stats = load_stats(a.run)
    skip = set()
    if a.skip_pages:
        skip = {tuple(l.split("\t")[:2]) for l in open(a.skip_pages)}
    kept, report = {}, []
    for f in sorted(glob.glob(f"{a.run}/ocr/*.txt")):
        bid = os.path.basename(f)[:-4]; st = stats.get(bid, {})
        body = [c for t, c in st.values() if t >= 200]
        conf = sum(body) / len(body) if body else 0
        pages = {int(m.group(1)): m.group(2) for m in PAGE.finditer(open(f).read())}
        if conf < a.book_conf:
            report.append((bid, round(conf, 1), len(pages), 0, 0, "dropped: low confidence")); continue
        heads = collections.Counter(re.sub(r"[^஀-௿]", "", (p.strip().split("\n") or [""])[0].replace("‌", ""))
                                    for p in pages.values())
        repeated = {k for k, v in heads.items() if k and v >= max(4, len(pages) // 10)}
        cf = {}
        if a.line_conf and os.path.exists(f"{a.run}/conf/{bid}.txt"):
            cf = {int(m.group(1)): m.group(2) for m in PAGE.finditer(open(f"{a.run}/conf/{bid}.txt").read())}
        good = [(n, page_lines(p, repeated, cf.get(n), a.line_conf)) for n, p in sorted(pages.items())
                if st.get(n, (0, 0))[0] >= 200 and st[n][1] >= a.page_conf
                and (bid, str(n)) not in skip]
        kept[bid] = (conf, len(pages), good)
    vocab = collections.Counter()                    # words not at a line edge
    for conf, n, good in kept.values():
        for _, lines in good:
            for l in lines:
                if l: vocab.update(TAW.findall(l)[1:-1])
    with open(f"{a.out}/clean_stats.tsv", "w") as rep:
        rep.write("book\tconf\tpages\tpages_kept\twords\tnote\n")
        for bid, (conf, n, good) in kept.items():
            words = 0
            with open(f"{a.out}/{bid}.txt", "w") as o:
                for _, lines in good:
                    buf = ""
                    for l in lines:
                        if l is None:                # a dropped line: end the paragraph here
                            if buf:
                                words += len(TAW.findall(buf)); o.write(buf + "\n\n"); buf = ""
                            continue
                        if buf:
                            ma, mb = re.search(r"([஀-௿]+)-?$", buf), re.match(r"[஀-௿]+", l)
                            if ma and mb:
                                x, y = ma.group(1), mb.group(0); j = x + y
                                if buf.endswith("-") or (vocab[j] and vocab[j] >= min(vocab[x], vocab[y])):
                                    buf = buf.rstrip("-") + l; continue
                            buf += " " + l
                        else:
                            buf = l
                    if buf:
                        words += len(TAW.findall(buf)); o.write(buf + "\n\n")
            rep.write(f"{bid}\t{conf:.1f}\t{n}\t{len(good)}\t{words}\tkept\n")
        for r in report:
            rep.write("\t".join(map(str, r)) + "\n")
    tot = sum(int(l.split("\t")[4]) for l in list(open(f"{a.out}/clean_stats.tsv"))[1:])
    print("books kept", len(kept), "dropped", len(report), "Tamil words kept", tot)

if __name__ == "__main__":
    main()
