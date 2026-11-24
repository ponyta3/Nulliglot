"""Per-book quality summary of the Tamil Digital Library OCR pilot.

    python tdl_pilot_report.py <pilot_dir> <books.json>

Signals (no ground truth needed):
  conf      mean Tesseract word confidence
  low       share of pages with mean confidence under 80
  bad_orth  share of Tamil words containing an impossible letter sequence
  frag      share of Tamil words that are a single letter
  latin     Latin letters as a share of Tamil + Latin letters
"""
import collections, json, re, sys, urllib.parse

VS = "ா-ௌௗ"                     # dependent vowel signs
BAD = re.compile(rf"(^[{VS}்])|([{VS}்][{VS}])|(்்)|([அ-ஔ][{VS}்])")
TAW = re.compile(r"[஀-௿]+")

def pages(path):
    return {int(m.group(1)): m.group(2) for m in re.finditer(
        r"\n=== page (\d+) ===\n(.*?)(?=\n=== page \d+ ===\n|\Z)", open(path).read(), re.S)}

d, books = sys.argv[1], json.load(open(sys.argv[2]))
stats = collections.defaultdict(list)
for l in open(f"{d}/page_stats.tsv"):
    r = l.rstrip("\n").split("\t"); stats[r[0]].append([float(x) for x in r[1:]])
print("id\tyear\tpages\twords\tconf\tlow\tbad_orth\tfrag\tlatin\tlayer\ttitle")
tot = 0
for n, row in enumerate(books):
    m = re.search(r"-(\d+)-", urllib.parse.unquote(row[-1])); bid = m.group(1) if m else f"book{n}"
    if bid not in stats:
        print(f"{bid}\t{row[3]}\t-\t-\t-\t-\t-\t-\t-\t-\t{row[1]} (no PDF on the site)"); continue
    s = stats[bid]; pg = pages(f"{d}/ocr/{bid}.txt")
    text = "\n".join(pg.values()).replace("‌", "")
    w = TAW.findall(text); body = [x for x in s if x[1] >= 200]     # pages with real text
    ta = sum(x[1] for x in s); la = sum(x[2] for x in s)
    conf = sum(x[4] for x in body) / max(1, len(body))
    low = sum(1 for x in body if x[4] < 80) / max(1, len(body))
    bad = sum(1 for x in w if BAD.search(x)) / max(1, len(w))
    frag = sum(1 for x in w if len(x) == 1) / max(1, len(w))
    layer = sum(1 for x in s if x[5] > 100) / len(s)
    tot += len(w)
    print(f"{bid}\t{row[3]}\t{len(s)}\t{len(w)}\t{conf:.1f}\t{100*low:.0f}%\t{100*bad:.2f}%\t{100*frag:.1f}%\t"
          f"{100*la/max(1,ta+la):.1f}%\t{100*layer:.0f}%\t{row[1].strip(' -0')}")
print("total Tamil words", tot)
