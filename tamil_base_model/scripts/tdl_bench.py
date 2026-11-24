"""Score pilot OCR against proofread Tamil Wikisource pages of the same book.

    python tdl_bench.py <bench_dir> <book_id> ...

Reports character error rate on Tamil letters only and word error rate, per book,
using Wikisource pages at proofread level 3 or 4. Pages are aligned by best match
within a small window, since the two scans can differ by a few pages.
"""
import json, re, sys, statistics
from rapidfuzz.distance import Levenshtein as L

def clean_wiki(t):
    t = re.sub(r"<noinclude>.*?</noinclude>", " ", t, flags=re.S)
    t = re.sub(r"<ref[^>]*>.*?</ref>|<[^>]+>", " ", t, flags=re.S)
    for _ in range(4):
        t = re.sub(r"\{\{(?:[^{}|]*\|)*([^{}|]*)\}\}", r"\1", t)   # keep last template argument
    t = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", t)
    t = re.sub(r"'{2,}|^[:;*#=]+|=+$|\{\||\|\}|^\|.*$", " ", t, flags=re.M)
    return t

def norm(t):
    t = t.replace("‌", "").replace("‍", "")
    t = re.sub(r"-\s*\n\s*", "", t)
    return t

letters = lambda t: re.sub(r"[^஀-௿]", "", norm(t))
def words(t):
    # join words broken across lines cannot be recovered reliably; compare word sequences as-is
    return re.findall(r"[஀-௿]+", norm(t))

def pages(path):
    out = {}
    for m in re.finditer(r"\n=== page (\d+) ===\n(.*?)(?=\n=== page \d+ ===\n|\Z)", open(path).read(), re.S):
        out[int(m.group(1))] = m.group(2)
    return out

for layer in ("ocr", "layer"):
    for bid in sys.argv[2:]:
        ws = json.load(open(f"{sys.argv[1]}/ws/{bid}.json"))
        oc = pages(f"{sys.argv[1]}/{layer}/{bid}.txt")
        cers, wers, nl, used = [], [], 0, 0
        ce = ct = we = wt = 0
        for n, p in sorted(ws.items(), key=lambda x: int(x[0])):
            if p["q"] < 3: continue
            ref = letters(clean_wiki(p["text"]))
            if len(ref) < 300: continue
            best = None
            for k in range(int(n) - 12, int(n) + 13):
                if k in oc:
                    hyp = letters(oc[k])
                    d = L.distance(ref, hyp)
                    if best is None or d < best[0]: best = (d, k)
            if best is None or best[0] / len(ref) > 0.5:   # no matching page: different edition
                continue
            used += 1
            ce += best[0]; ct += len(ref); cers.append(best[0] / len(ref))
            rw, hw = words(clean_wiki(p["text"])), words(oc[best[1]])
            d = L.distance(rw, hw); we += d; wt += len(rw); wers.append(d / max(1, len(rw)))
        if used:
            print(f"{layer:5} book {bid}: pages compared {used}, letter error {100*ce/ct:.2f}%, "
                  f"word error {100*we/wt:.1f}%, median page letter error {100*statistics.median(cers):.2f}%, "
                  f"worst decile {100*sorted(cers)[int(len(cers)*0.9)]:.1f}%")
        else:
            print(layer, "book", bid, "no comparable pages")
