"""Build an OCR benchmark from proofread Tamil Wikisource pages.

    python ocr_bench_build.py <ws_proofread_status.tsv> <ws_proofread_pages.tsv> <clean_stats.tsv> <out.json>
    python ocr_bench_build.py <ws_proofread_status.tsv> <ws_proofread_pages.tsv> years <out.json> <library_list.tsv> <books_out.json>

The second form picks books printed before 1979, eight per period, for the old-print test; the
"conf" field then holds the year, and <books_out.json> is the download list for tdl_pilot.py.

Picks books from the full run spread over Tesseract confidence bands, 12 proofread pages each,
and fetches the proofread text from the Wikisource API. Output: {book: {"index": name,
"conf": c, "pages": {page: wikitext}}}.
"""
import collections, json, random, re, sys, time
import httpx
status, pages_f, stats_f, out = sys.argv[1:5]
idx = {}; year = {}
for l in list(open(status))[1:]:
    r = l.rstrip("\n").split("\t")
    if r[0].isdigit(): idx[str(int(r[0]))] = r[3]; year[str(int(r[0]))] = int(r[1])
pp = collections.defaultdict(list)
for l in open(pages_f):
    b, p, q = l.split("\t"); pp[b].append(int(p))
conf = {}; per_band = 4
random.seed(11)
bands = [(0, 78), (78, 82), (82, 84), (84, 86), (86, 88), (88, 90), (90, 101)]
if stats_f == "years":
    conf = {b: float(y) for b, y in year.items() if y}; per_band = 8
    bands = [(1800, 1940), (1940, 1950), (1950, 1960), (1960, 1970), (1970, 1979)]
else:
    for l in list(open(stats_f))[1:]:
        r = l.split("\t"); conf[r[0]] = float(r[1])
pick = []
for lo, hi in bands:
    c = sorted(b for b in pp if b in conf and b in idx and lo <= conf[b] < hi and len(pp[b]) >= 30)
    pick += random.sample(c, min(per_band, len(c)))
res = {}
cl = httpx.Client(headers={"User-Agent": "ocr-benchmark-script/0.1 (research)"}, timeout=60)
for b in pick:
    ps = sorted(pp[b]); ps = [p for p in ps if p > ps[0] + 5][: len(ps)]      # skip front matter
    ps = sorted(random.sample(ps, min(12, len(ps))))
    titles = "|".join(f"Page:{idx[b]}/{p}" for p in ps); got = {}
    for attempt in range(6):
        try:
            j = cl.get("https://ta.wikisource.org/w/api.php", params=dict(action="query", prop="revisions",
                       rvprop="content", rvslots="main", titles=titles, format="json", formatversion=2, maxlag=5)).json()
            for pg in j["query"]["pages"]:
                if "revisions" in pg:
                    got[pg["title"].rsplit("/", 1)[1]] = pg["revisions"][0]["slots"]["main"]["content"]
            break
        except Exception as e:
            time.sleep(5 * (attempt + 1))
    res[b] = {"index": idx[b], "conf": conf[b], "pages": got}
    print(b, conf[b], len(got), flush=True); time.sleep(1)
json.dump(res, open(out, "w"), ensure_ascii=False)
print("books", len(res), "pages", sum(len(v["pages"]) for v in res.values()))
if stats_f == "years":
    rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[5])]
    json.dump([r for r in rows if r[0].isdigit() and str(int(r[0])) in res], open(sys.argv[6], "w"), ensure_ascii=False)
