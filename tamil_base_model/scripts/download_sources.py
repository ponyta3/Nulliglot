"""Download the raw files of sources 1 to 6 of survey/final_list.md. Nothing is cleaned here.

    python download_sources.py <out_dir> [source ...]

Sources: wikimedia (1, 3, 4: Wikisource, Wikinews, Wikibooks, Wikiquote dumps), thamizh_mann (2),
kaniyam (5), fte (6: FreeTamilEbooks). With no source named, all are fetched. Files that are
already present are skipped, so the script can be run again after a failure.

Layout under <out_dir>:
  wikimedia/<wiki>-latest-pages-articles.xml.bz2 (+ sha1 list; page tables for Wikisource)
  thamizh_mann/...            the collections/thamizh_mann folder of the GitHub repository
  kaniyam/posts_NN.json       all posts from the site's public API, 100 per file
  fte/catalogue_NN.json       all book records from the site's public API
  fte/epub/<slug>.epub        one EPUB per book whose stated licence is not clearly restricted
  fte/books.tsv               every book with its licence text and what was done
  manifest.tsv                every file fetched: source, path, URL, bytes, time
"""
import hashlib, html, io, json, os, re, sys, tarfile, time, urllib.error, urllib.parse, urllib.request

UA = "research-downloader/1.0"
OUT = sys.argv[1]
WANT = sys.argv[2:] or ["wikimedia", "thamizh_mann", "kaniyam", "fte"]

def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)

def manifest(source, path, url):
    with open(f"{OUT}/manifest.tsv", "a") as m:
        m.write(f"{source}\t{os.path.relpath(path, OUT)}\t{url}\t{os.path.getsize(path)}\t"
                f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n")

def fetch(url, path, source, tries=4, pause=1.0):
    """Download url to path (skipped if present). Returns True when the file is there."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return True
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as r, open(path + ".part", "wb") as o:
                while True:
                    b = r.read(1 << 20)
                    if not b: break
                    o.write(b)
            os.replace(path + ".part", path)
            manifest(source, path, url)
            time.sleep(pause)
            return True
        except urllib.error.HTTPError as e:
            log("HTTP", e.code, url)
            if e.code in (400, 401, 403, 404, 410): break      # refused or missing: do not retry
        except Exception as e:
            log("error", repr(e)[:120], url)
        time.sleep(10 * (t + 1))
    if os.path.exists(path + ".part"): os.remove(path + ".part")
    return False

def wikimedia():
    d = f"{OUT}/wikimedia"
    for w in ("tawikisource", "tawikinews", "tawikibooks", "tawikiquote"):
        base = f"https://dumps.wikimedia.org/{w}/latest/{w}-latest-"
        names = ["pages-articles.xml.bz2", "sha1sums.txt"]
        if w == "tawikisource": names += ["page.sql.gz", "page_props.sql.gz"]
        for n in names:
            ok = fetch(base + n, f"{d}/{w}-latest-{n}", "wikimedia")
            log(w, n, "ok" if ok else "FAILED")
        f, s = f"{d}/{w}-latest-pages-articles.xml.bz2", f"{d}/{w}-latest-sha1sums.txt"
        if os.path.exists(f) and os.path.exists(s):
            h = hashlib.sha1()
            with open(f, "rb") as i:
                for b in iter(lambda: i.read(1 << 20), b""): h.update(b)
            want = [l.split()[0] for l in open(s) if l.strip().endswith("pages-articles.xml.bz2")]
            log(w, "sha1", "matches" if want and want[0] == h.hexdigest() else f"NOT VERIFIED {want[:1]}")

def thamizh_mann():
    d = f"{OUT}/thamizh_mann"
    if os.path.exists(f"{d}/.done"):
        log("thamizh_mann already present"); return
    tgz = f"{OUT}/thamizh_mann_repo.tar.gz"
    url = "https://codeload.github.com/Digital-Tamil-Studies/open_tamil_texts/tar.gz/refs/heads/master"
    if not fetch(url, tgz, "thamizh_mann"):
        log("thamizh_mann FAILED"); return
    n = 0
    with tarfile.open(tgz) as t:
        for m in t:
            parts = m.name.split("/", 1)
            if len(parts) == 2 and parts[1].startswith("collections/thamizh_mann/") and m.isfile():
                rel = os.path.normpath(parts[1][len("collections/thamizh_mann/"):])
                if rel.startswith(".."): continue
                p = f"{d}/{rel}"; os.makedirs(os.path.dirname(p), exist_ok=True)
                with t.extractfile(m) as i, open(p, "wb") as o: o.write(i.read())
                n += 1
    open(f"{d}/.done", "w").write(str(n))
    os.remove(tgz)
    log("thamizh_mann files", n)

def wp_all(api, d, name, source, fields=None):
    """Fetch every record of a WordPress collection, 100 per file."""
    page, rows = 1, []
    while True:
        u = f"{api}?per_page=100&page={page}" + (f"&_fields={fields}" if fields else "")
        p = f"{d}/{name}_{page:02d}.json"
        if not fetch(u, p, source): break
        r = json.load(open(p))
        rows += r
        if len(r) < 100: break
        page += 1
    return rows

def kaniyam():
    d = f"{OUT}/kaniyam"
    rows = wp_all("https://kaniyam.com/wp-json/wp/v2/posts", d, "posts", "kaniyam")
    log("kaniyam posts", len(rows))
    for what in ("categories", "users"):
        r = wp_all(f"https://kaniyam.com/wp-json/wp/v2/{what}", d, what, "kaniyam", "id,name,slug")
        log("kaniyam", what, len(r))

RESTRICTED = re.compile(r"non[\s-]*com|\bNC\b|no[\s-]*deriv|\bND\b|royalty", re.I)

def ia_epub(content):
    """The site's download links redirect to the Internet Archive, and the site itself drops
    many connections. When the book page names its Archive item, return the EPUB URL there."""
    m = re.search(r'archive\.org/details/([^"/?#\s]+)', content)
    if not m: return None
    try:
        req = urllib.request.Request("https://archive.org/metadata/" + m.group(1), headers={"User-Agent": UA})
        files = json.load(urllib.request.urlopen(req, timeout=60)).get("files", [])
    except Exception as e:
        log("archive metadata error", repr(e)[:80], m.group(1)); return None
    time.sleep(0.5)
    ep = [f["name"] for f in files if f["name"].lower().endswith(".epub") and f.get("source") == "original"] \
         or [f["name"] for f in files if f["name"].lower().endswith(".epub")]
    return f"https://archive.org/download/{m.group(1)}/{urllib.parse.quote(ep[0])}" if ep else None

def fte():
    d = f"{OUT}/fte"
    rows = wp_all("https://freetamilebooks.com/wp-json/wp/v2/ebooks", d, "catalogue", "fte")
    log("fte books in catalogue", len(rows))
    got = skipped = failed = nolink = 0
    with open(f"{d}/books.tsv", "w") as rep:
        rep.write("id\tslug\tdate\tlicence_text\tstatus\ttitle\n")
        for r in rows:
            c = r["content"]["rendered"]
            text = html.unescape(re.sub(r"<[^>]+>", " ", c))
            lic = " | ".join(re.sub(r"\s+", " ", x).strip()[:70]
                             for x in re.findall(r"உரிமை\s*[:–-]\s*([^\n]{0,80})", text))
            links = [u for u, t in re.findall(r'href="([^"]+)"[^>]*>(.*?)</a>', c, re.S)
                     if "epub" in (urllib.parse.unquote(u) + t).lower()]
            if RESTRICTED.search(lic):
                status = "skipped: restricted licence"; skipped += 1
            elif not links:
                status = "no epub link"; nolink += 1
            else:
                p = f"{d}/epub/{r['slug']}.epub"
                ok = os.path.exists(p) or fetch(ia_epub(c) or html.unescape(links[0]), p, "fte", tries=2)
                if not ok: ok = fetch(html.unescape(links[0]), p, "fte", tries=2)
                if ok and open(p, "rb").read(2) != b"PK":
                    os.remove(p); ok = False
                status = "downloaded" if ok else "FAILED"
                got += ok; failed += not ok
            title = html.unescape(r["title"]["rendered"]).replace("\t", " ")
            rep.write(f"{r['id']}\t{r['slug']}\t{r['date'][:10]}\t{lic}\t{status}\t{title}\n")
    log("fte downloaded", got, "skipped (restricted licence)", skipped, "no epub link", nolink, "failed", failed)

os.makedirs(OUT, exist_ok=True)
for s in WANT:
    log("=== ", s)
    {"wikimedia": wikimedia, "thamizh_mann": thamizh_mann, "kaniyam": kaniyam, "fte": fte}[s]()
log("finished")
