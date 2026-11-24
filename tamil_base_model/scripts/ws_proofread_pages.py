"""List the pages of each library book that are already proofread or validated on Tamil
Wikisource, so they can be left out of the OCR text (source 1 keeps the proofread version).

    python ws_proofread_pages.py <dir with the two .sql.gz files> <ws_proofread_status.tsv> <out.tsv>

Output rows: book id, page number, quality (3 proofread, 4 validated). The page number is the
Wikisource Page: number, which is the page number inside the scan PDF.
"""
import collections, re, sys, urllib.parse
sys.path.insert(0, __file__.rsplit("/", 1)[0])
import gzip

def rows(path):
    pat = re.compile(rb"\((\d+),(\d+|'(?:[^'\\]|\\.)*'),('(?:[^'\\]|\\.)*'|\d+)")
    with gzip.open(path, "rb") as f:
        for line in f:
            if line[:1] == b"(" or line.startswith(b"INSERT INTO"):
                yield from pat.finditer(line)

d, status, out = sys.argv[1:4]
index2book = collections.defaultdict(list); mismatch = 0
for l in list(open(status))[1:]:
    r = l.rstrip("\n").split("\t"); u = urllib.parse.unquote(r[0])
    m = re.fullmatch(r"0*(\d+)", u) or re.search(r"-(\d+)-", u) or re.search(r"/Articles/0*(\d+)_", u)
    dig = lambda x: re.findall(r"\d+", x)
    td, wd = dig(re.sub(r"-\s*0?\s*$", "", r[2])), dig(r[3].rsplit(".", 1)[0])
    if m and (td == wd or (not td and wd == ["1"])):     # same volume number in both titles
        index2book[r[3]].append(m.group(1))
    elif m:
        mismatch += 1
qual = {}
for m in rows(f"{d}/tawikisource-latest-page_props.sql.gz"):
    if m.group(2) == b"'proofread_page_quality_level'":
        qual[int(m.group(1))] = int(m.group(3).strip(b"'"))
n = 0
with open(out, "w") as f:
    for m in rows(f"{d}/tawikisource-latest-page.sql.gz"):
        pid = int(m.group(1))
        if m.group(2) != b"250" or qual.get(pid, 0) < 3:
            continue
        t = m.group(3)[1:-1].replace(b"\\'", b"'").replace(b'\\"', b'"').decode("utf-8", "replace")
        if "/" not in t:
            continue
        idx, pg = t.rsplit("/", 1)
        if not pg.isdigit():
            continue
        for b in index2book.get(idx.replace("_", " "), ()):
            f.write(f"{b}\t{pg}\t{qual[pid]}\n"); n += 1
print("proofread pages listed", n, "for", len(index2book), "index files;",
      mismatch, "books left out because the volume number differs from the index name")
