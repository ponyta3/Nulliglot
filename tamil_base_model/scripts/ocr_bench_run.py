"""Run one Tesseract model over the benchmark pages.

    python ocr_bench_run.py <bench.json> <pdf_dir> <existing_ocr_dir> <out.json> <lang> [thresholding_method] [psm]

The Wikisource page number is matched to a PDF page with the existing OCR text (best match
within 12 pages); the match is cached in align.json next to bench.json so every engine reads
the same pages. Needs TESSDATA_PREFIX to point at the model directory.
"""
import csv, io, json, os, re, subprocess, sys
from multiprocessing import Pool
import pymupdf
from rapidfuzz.distance import Levenshtein as L
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ocr_bench_score import clean_wiki, letters

bench_f, pdf_dir, ocr_dir, out, lang = sys.argv[1:6]
th = sys.argv[6] if len(sys.argv) > 6 else "2"
psm = sys.argv[7] if len(sys.argv) > 7 else "3"

def pages(path):
    return {int(m.group(1)): m.group(2) for m in
            re.finditer(r"\n=== page (\d+) ===\n(.*?)(?=\n=== page \d+ ===\n|\Z)", open(path).read(), re.S)}

def align(bench):
    f = os.path.join(os.path.dirname(bench_f), "align.json")
    if os.path.exists(f):
        return json.load(open(f))
    a = {}
    for b, v in bench.items():
        oc = pages(f"{ocr_dir}/{b}.txt"); a[b] = {}
        for n, t in v["pages"].items():
            ref = letters(clean_wiki(t))
            if len(ref) < 300: continue
            best = min(((L.distance(ref, letters(oc[k])), k) for k in range(int(n) - 12, int(n) + 13) if k in oc), default=None)
            if best and best[0] / len(ref) <= 0.5: a[b][n] = best[1]
    json.dump(a, open(f, "w")); return a

def ocr(job):
    b, n, k = job
    page = pymupdf.open(f"{pdf_dir}/{b}.pdf")[k - 1]
    dpi = int(min(300, 3600 * 72 / max(page.rect.width, page.rect.height, 1)))
    png = page.get_pixmap(dpi=max(dpi, 50), colorspace=pymupdf.csGRAY).tobytes("png")
    r = subprocess.run(["tesseract", "stdin", "stdout", "-l", lang, "--psm", psm, "-c", f"thresholding_method={th}", "tsv"],
                       input=png, capture_output=True, env=dict(os.environ, OMP_THREAD_LIMIT="1"), timeout=600)
    lines, key = [], None
    for row in csv.reader(io.StringIO(r.stdout.decode("utf-8", "replace")), delimiter="\t", quoting=csv.QUOTE_NONE):
        if len(row) < 12 or row[0] != "5" or not row[11].strip(): continue
        if tuple(row[1:5]) != key:
            lines.append([]); key = tuple(row[1:5])
        lines[-1].append([row[11], float(row[10])])
    return b, n, lines

if __name__ == "__main__":
    bench = json.load(open(bench_f)); a = align(bench)
    jobs = [(b, n, k) for b in a for n, k in a[b].items()]
    res = {}
    with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", 8))) as p:
        for b, n, lines in p.imap_unordered(ocr, jobs):
            res.setdefault(b, {})[n] = lines
    json.dump(res, open(out, "w"), ensure_ascii=False)
    print(out, "pages", len(jobs))
