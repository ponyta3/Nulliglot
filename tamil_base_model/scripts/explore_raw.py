"""One-off look at the raw source files (data/raw) to design the cleaning. Prints summaries."""
import bz2, collections, csv, glob, json, re, sys, random
import xml.etree.ElementTree as ET
R = sys.argv[1]; what = sys.argv[2]
random.seed(1)
def dump_pages(path):
    ns = "{http://www.mediawiki.org/xml/export-0.11/}"
    for ev, el in ET.iterparse(bz2.open(path, "rb")):
        if el.tag.endswith("}page"):
            t = el.find("./{*}title").text; n = int(el.find("./{*}ns").text)
            tx = el.find("./{*}revision/{*}text"); red = el.find("./{*}redirect") is not None
            yield t, n, (tx.text or "") if tx is not None else "", red
            el.clear()
if what == "tm":
    rows = list(csv.reader(open(f"{R}/thamizh_mann/metadata.csv", encoding="utf-8")))
    print(rows[0]); print(rows[1]); print(len(rows))
    fs = sorted(glob.glob(f"{R}/thamizh_mann/data/*.txt"))
    for f in random.sample(fs, 2):
        t = open(f, encoding="utf-8").read(); L = t.split("\n")
        print("=====", f, len(t), len(L)); print("\n".join(L[:70])); print("..."); m = len(L)//2; print("\n".join(L[m:m+40])); print("..."); print("\n".join(L[-25:]))
    ll = collections.Counter()
    for f in random.sample(fs, 60):
        for l in open(f, encoding="utf-8"):
            ll[min(len(l.strip())//20, 15)] += 1
    print("line length /20:", sorted(ll.items()))
elif what == "kaniyam":
    posts = [p for f in sorted(glob.glob(f"{R}/kaniyam/posts_*.json")) for p in json.load(open(f))]
    print(len(posts), list(posts[0].keys()))
    tags = collections.Counter(t for p in posts for t in re.findall(r"<(\w+)", p["content"]["rendered"]))
    print(tags.most_common(50))
    cls = collections.Counter(c for p in posts for c in re.findall(r'class="([^"]+)"', p["content"]["rendered"]))
    print(cls.most_common(40))
    for p in random.sample(posts, 3): print("=====", p["id"], p["date"], p["author"], p["title"]["rendered"]); print(p["content"]["rendered"][:2500])
    print([(u["id"], u["name"]) for f in glob.glob(f"{R}/kaniyam/users_*.json") for u in json.load(open(f))][:60])
elif what in ("ws", "wn", "wb", "wq"):
    w = {"ws": "tawikisource", "wn": "tawikinews", "wb": "tawikibooks", "wq": "tawikiquote"}[what]
    nsc = collections.Counter(); q = collections.Counter(); tpl = collections.Counter(); tag = collections.Counter()
    samples = []; words = collections.Counter(); TA = re.compile(r"[஀-௿]+"); redir = 0; idx = set()
    for t, n, x, red in dump_pages(f"{R}/wikimedia/{w}-latest-pages-articles.xml.bz2"):
        nsc[n] += 1; redir += red
        if what == "ws" and n == 250:
            m = re.search(r'<pagequality level="(\d)"', x); lv = m.group(1) if m else "none"; q[lv] += 1
            if lv not in ("3", "4"): continue
            words[lv] += len(TA.findall(x)); idx.add(t.split(":", 1)[1].rsplit("/", 1)[0])
        elif what == "ws": 
            if n == 0: words["ns0"] += len(TA.findall(x)); q["ns0 with <pages"] += "<pages" in x
            continue
        elif n != 0 or red: continue
        else: words["ns0"] += len(TA.findall(x))
        tpl.update(m.strip().lower()[:40] for m in re.findall(r"\{\{\s*([^|{}\n<]+)", x))
        tag.update(re.findall(r"<(\w+)", x))
        if random.random() < (0.00004 if what == "ws" else 0.002) and len(samples) < 5: samples.append((t, x))
    print("namespaces", nsc.most_common(20), "redirects", redir); print("quality", q); print("words", words, "index files", len(idx))
    print("templates", len(tpl), tpl.most_common(150)); print("tags", tag.most_common(50))
    for t, x in samples: print("=====", t); print(x[:3000])
