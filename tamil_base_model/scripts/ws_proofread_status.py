"""Per-book proofreading status on Tamil Wikisource, joined to the Tamil Digital Library list.

Reads the `page` and `page_props` SQL dumps (no API calls) and writes, for every index file,
the number of Page: pages at each quality level (0 empty, 1 not proofread, 2 problematic,
3 proofread, 4 validated).

    python ws_proofread_status.py <dir with the two .sql.gz files> <tdl_list.tsv> <out.tsv>
"""
import collections, gzip, re, sys

def rows(path):
    pat = re.compile(rb"\((\d+),(\d+|'(?:[^'\\]|\\.)*'),('(?:[^'\\]|\\.)*'|\d+)")
    with gzip.open(path, "rb") as f:
        for line in f:
            if line[:1] == b"(" or line.startswith(b"INSERT INTO"):
                yield from pat.finditer(line)

d, tdl, out = sys.argv[1:4]
qual = {}
for m in rows(f"{d}/tawikisource-latest-page_props.sql.gz"):
    if m.group(2) == b"'proofread_page_quality_level'":
        qual[int(m.group(1))] = int(m.group(3).strip(b"'"))
book = collections.defaultdict(lambda: [0] * 5)
for m in rows(f"{d}/tawikisource-latest-page.sql.gz"):
    if m.group(2) != b"250":
        continue
    pid = int(m.group(1))
    title = m.group(3)[1:-1].replace(b"\\'", b"'").replace(b'\\"', b'"').decode("utf-8", "replace")
    if "/" not in title or pid not in qual:
        continue
    book[title.rsplit("/", 1)[0].replace("_", " ")][qual[pid]] += 1
norm = lambda s: re.sub(r"[^஀-௿]", "", s)
by_norm = {}
for b, c in book.items():
    by_norm.setdefault(norm(b.rsplit(".", 1)[0]), (b, c))
tot = collections.Counter(); n = 0
with open(out, "w") as f:
    f.write("tdl_id\tyear\ttitle\tws_index\tq0\tq1\tq2\tq3\tq4\n")
    for l in open(tdl):
        r = l.rstrip("\n").split("\t"); k = norm(r[1])
        if not k or k not in by_norm:
            continue
        b, c = by_norm[k]; m = re.search(r"(1[89]\d\d|20\d\d)", r[3]); y = int(m.group(1)) if m else 0
        f.write("\t".join(map(str, [r[0], y, r[1], b, *c])) + "\n"); n += 1
        per = "1979+" if y >= 1979 else "unknown" if y == 0 else "before 1979"
        tot[per, "books"] += 1; tot[per, "not_proofread"] += c[1] + c[2]; tot[per, "proofread"] += c[3] + c[4]
        tot[per, "fully"] += (c[1] + c[2] == 0); tot[per, "untouched"] += (c[3] + c[4] == 0)
allq = collections.Counter()
for c in book.values():
    for i, v in enumerate(c): allq[i] += v
print("all Wikisource Page: pages by quality", dict(allq), "index files with pages", len(book))
print("library books matched to an index with pages:", n)
for per in ("1979+", "before 1979", "unknown"):
    print(per, {k[1]: v for k, v in tot.items() if k[0] == per})
