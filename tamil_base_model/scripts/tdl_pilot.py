"""Pilot: OCR a few nationalised books from the Tamil Digital Library.

For each book in a JSON list (rows of the listing: id, title, author, year, ..., page URL)
it finds the PDF link on the book page, downloads the PDF, renders every page at 300 dpi,
runs Tesseract (Tamil) and writes one text file per book plus a per-page statistics file.

    python tdl_pilot.py <books.json> <out_dir> [--max-pages N] [--workers 8]
"""
import argparse, csv, io, json, os, re, subprocess, sys, time, urllib.parse
from multiprocessing import Pool
import httpx, pymupdf

UA = {"User-Agent": "Mozilla/5.0 (research pilot; low rate)"}
TA = re.compile(r"[஀-௿]")
LAT = re.compile(r"[A-Za-z]")


def pdf_url(page_url, client):
    h = client.get(page_url).text
    m = re.search(r'data="(https://tamildigitallibrary\.in/assets/docs/uploads/primary_files/[^"]+\.pdf)"', h)
    return m.group(1) if m else None


def ocr_page(args):
    path, i = args
    doc = pymupdf.open(path)
    page = doc[i]
    layer = page.get_text()
    # 300 dpi for book-sized pages; oversized pages are capped at 3,600 px on the long side,
    # otherwise one page can take Tesseract many minutes and over 1 GB of memory
    dpi = int(min(300, 3600 * 72 / max(page.rect.width, page.rect.height, 1)))
    png = page.get_pixmap(dpi=max(dpi, 50), colorspace=pymupdf.csGRAY).tobytes("png")
    env = dict(os.environ, OMP_THREAD_LIMIT="1")
    # thresholding_method=2 (Sauvola): the default (Otsu) returns nothing on grey or brown scans
    try:
        r = subprocess.run(["tesseract", "stdin", "stdout", "-l", "tam", "--psm", "3",
                        "-c", "thresholding_method=2", "tsv"],
                           input=png, capture_output=True, env=env, timeout=180)
    except subprocess.TimeoutExpired:
        return i, "", 0.0, layer
    lines, confs, key, cur = [], [], None, []
    for row in csv.reader(io.StringIO(r.stdout.decode("utf-8", "replace")), delimiter="\t", quoting=csv.QUOTE_NONE):
        if len(row) < 12 or row[0] != "5":
            continue
        k = tuple(row[1:5])
        if k != key:
            if cur:
                lines.append(" ".join(cur))
            cur, key = [], k
        if row[11].strip():
            cur.append(row[11]); confs.append(float(row[10]))
    if cur:
        lines.append(" ".join(cur))
    return i, "\n".join(lines), (sum(confs) / len(confs) if confs else 0.0), layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("books"); ap.add_argument("out")
    ap.add_argument("--max-pages", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--shard", type=lambda s: tuple(map(int, s.split("/"))), default=None,
                    help="i/n: process only books whose position in the list is i modulo n")
    a = ap.parse_args()
    for d in ("pdf", "ocr", "layer"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    client = httpx.Client(headers=UA, follow_redirects=True, timeout=600)
    stats = open(os.path.join(a.out, f"page_stats{'_%d' % a.shard[0] if a.shard else ''}.tsv"), "a")
    for n, row in enumerate(json.load(open(a.books))):
        url = row[-1]
        bid = (re.search(r"-(\d+)-", urllib.parse.unquote(url)) or re.search(r"/Articles/0*(\d+)_", url))
        bid = bid.group(1) if bid else f"book{n}"
        if a.shard and n % a.shard[1] != a.shard[0]:
            continue
        pdf = os.path.join(a.out, "pdf", f"{bid}.pdf")
        out = os.path.join(a.out, "ocr", f"{bid}.txt")
        if os.path.exists(out):
            continue
        try:
            if not os.path.exists(pdf):
                u = pdf_url(url, client)
                if not u:
                    print(bid, "no pdf link", flush=True); continue
                # curl resumes a stalled transfer; the server sometimes stops mid-file
                for attempt in range(8):
                    rc = subprocess.run(["curl", "-sSL", "-C", "-", "--max-time", "900", "--speed-time", "30",
                                         "--speed-limit", "1000", "-A", UA["User-Agent"], "-o", pdf + ".part",
                                         str(httpx.URL(u))]).returncode
                    if rc == 0:
                        break
                    time.sleep(3)
                if rc != 0:
                    print(bid, "download failed", rc, flush=True); continue
                os.rename(pdf + ".part", pdf)
                time.sleep(1)
            doc = pymupdf.open(pdf)
            npages = len(doc) if not a.max_pages else min(len(doc), a.max_pages)
            creator = doc.metadata.get("creator", "")
            doc.close()
            t0 = time.time()
            with Pool(a.workers) as pool:
                res = sorted(pool.map(ocr_page, [(pdf, i) for i in range(npages)], chunksize=4))
            with open(out, "w") as f, open(os.path.join(a.out, "layer", f"{bid}.txt"), "w") as g:
                for i, text, conf, layer in res:
                    f.write(f"\n=== page {i + 1} ===\n{text}\n")
                    g.write(f"\n=== page {i + 1} ===\n{layer}\n")
                    stats.write("\t".join(map(str, [bid, i + 1, len(TA.findall(text)), len(LAT.findall(text)),
                                                    len(text.split()), round(conf, 1), len(TA.findall(layer))])) + "\n")
            stats.flush()
            print(bid, row[1], "pages", npages, "MB", round(os.path.getsize(pdf) / 1e6, 1),
                  "sec", round(time.time() - t0), "creator", creator, flush=True)
        except Exception as e:
            print(bid, "ERROR", repr(e), flush=True)


if __name__ == "__main__":
    main()
