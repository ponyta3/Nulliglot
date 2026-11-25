"""Count documents, paragraphs and Tamil words in every source folder of the final corpus.
    python corpus_summary.py <final_dir>"""
import glob, os, re, sys
TAW = re.compile(r"[஀-௿]+")
OTH = re.compile(r"[^\W\d_஀-௿]+")
print("source\tdocuments\tparagraphs\ttamil_words\tother_script_words_pct\tmegabytes")
T = [0, 0, 0, 0, 0]
for d in sorted(glob.glob(sys.argv[1] + "/*/")):
    n = p = w = o = b = 0
    for f in glob.glob(d + "*.txt"):
        t = open(f, encoding="utf-8").read()
        if not t.strip(): continue
        n += 1; p += len([x for x in t.split("\n\n") if x.strip()]); w += len(TAW.findall(t)); o += len(OTH.findall(t)); b += len(t.encode())
    print(f"{os.path.basename(d[:-1])}\t{n}\t{p}\t{w}\t{100 * o / max(1, w + o):.2f}\t{b / 1e6:.0f}")
    for i, v in enumerate((n, p, w, o, b)): T[i] += v
print(f"TOTAL\t{T[0]}\t{T[1]}\t{T[2]}\t{100 * T[3] / max(1, T[2] + T[3]):.2f}\t{T[4] / 1e6:.0f}")
