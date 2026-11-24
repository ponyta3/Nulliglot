"""Score OCR engines on the benchmark against proofread Wikisource text.

    python ocr_bench_score.py <bench.json> <engine.json> [<engine.json> ...]

Engine files hold {book: {page: [[ [word, confidence], ... ] per line ]}}.
Letter error: edit distance on Tamil letters only (spacing, punctuation, digits ignored),
after dropping the first and last line of the page when short (running header, page number).
Words correct: share of OCR words, not at a line edge, that appear on the proofread page.
"""
import collections, json, re, sys
from rapidfuzz.distance import Levenshtein as L

def clean_wiki(t):
    t = re.sub(r"<noinclude>.*?</noinclude>", " ", t, flags=re.S)
    t = re.sub(r"<ref[^>]*>.*?</ref>|<[^>]+>", " ", t, flags=re.S)
    for _ in range(4):
        t = re.sub(r"\{\{(?:[^{}|]*\|)*([^{}|]*)\}\}", r"\1", t)
    t = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", t)
    t = re.sub(r"'{2,}|^[:;*#=]+|=+$|\{\||\|\}|^\|.*$", " ", t, flags=re.M)
    return t

nz = lambda t: t.replace("‌", "").replace("‍", "")
letters = lambda t: re.sub(r"[^஀-௿]", "", nz(t))
TAW = re.compile(r"[஀-௿]+")

def body(lines):
    ls = [l for l in lines if l]
    for end in (0, -1):
        if ls and len("".join(w for w, c in ls[end])) < 45 and len(ls) > 3:
            ls = ls[1:] if end == 0 else ls[:-1]
    return ls

if __name__ == "__main__":
    bench = json.load(open(sys.argv[1]))
    band = lambda c: "below 82" if c < 82 else "82-86" if c < 86 else "86-90" if c < 90 else "90+"
    names = ("all", "below 82", "82-86", "86-90", "90+")
    if all(v["conf"] > 1000 for v in bench.values()):          # old-print test: "conf" holds the year
        band = lambda y: "before 1940" if y < 1940 else f"{int(y) // 10 * 10}s"
        names = ("all", "before 1940", "1940s", "1950s", "1960s", "1970s")
    for f in sys.argv[2:]:
        eng = json.load(open(f)); S = collections.Counter(); per_book = {}
        for b, pg in eng.items():
            e = t = 0
            for n, lines in pg.items():
                ref_t = nz(clean_wiki(bench[b]["pages"][n])); ref = letters(ref_t); truth = set(TAW.findall(ref_t))
                stream = ref
                ls = body(lines); hyp = letters(" ".join(w for l in ls for w, c in l))
                # the proofread page may also carry or lack the header: take the better of with/without
                hyp_all = letters(" ".join(w for l in lines for w, c in l))
                d = min(L.distance(ref, hyp), L.distance(ref, hyp_all))
                e += d; t += len(ref)
                for l in ls:
                    for w, c in l[1:-1]:
                        w = letters(w)
                        if w:
                            for k in ("all", band(bench[b]["conf"])):
                                S[k, "w"] += 1; S[k, "ok"] += w in truth
                                S[k, "sp"] += w not in truth and len(w) >= 3 and w in stream
            per_book[b] = e / max(1, t)
            for k in ("all", band(bench[b]["conf"])):
                S[k, "e"] += e; S[k, "t"] += t; S[k, "books"] += 1
        print(f"\n{f}")
        for k in names:
            if S[k, "t"]:
                w = max(1, S[k, "w"])
                print(f"  books {k:11} n={S[k,'books']:2}  letter error {100*S[k,'e']/S[k,'t']:5.2f}%   words exact {100*S[k,'ok']/w:5.2f}%"
                      f"   real letter mistakes {100*(w-S[k,'ok']-S[k,'sp'])/w:5.2f}% of words")
        worst = sorted(per_book.items(), key=lambda x: -x[1])[:5]
        print("  worst books (letter error):", ", ".join(f"{b} {100*v:.1f}%" for b, v in worst))
