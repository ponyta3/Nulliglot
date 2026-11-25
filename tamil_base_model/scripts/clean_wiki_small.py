"""Clean sources 3 and 4: Tamil Wikinews, Wikibooks and Wikiquote dumps, one file per page.

    python clean_wiki_small.py <dump.xml.bz2> <out_dir> <wikinews|wikibooks|wikiquote> [--min-tamil 0.8]

Main-namespace pages that are not redirects. Wiki markup, templates, references, tables, code
and images are removed (wikitext.py); the sections that only list sources, references and
links are cut; paragraphs that are not mostly Tamil are dropped; pages left with under 30
Tamil words are dropped.
Writes <out_dir>/<page id>.txt, <out_dir>/clean_stats.tsv and <out_dir>/dropped_templates.tsv.
"""
import argparse, collections, os, re
import wikitext
from textutil import is_tamil, nwords, tidy

STOP = {"மூலம்", "மூலங்கள்", "மூலம்:", "தொடர்புள்ள செய்திகள்", "தொடர்புடைய செய்திகள்", "வெளி இணைப்புகள்",
        "வெளியிணைப்புகள்", "வெளி இணைப்பு", "புற இணைப்புகள்", "சான்றுகள்", "சான்றுகோள்கள்", "குறிப்புகள்",
        "உசாத்துணை", "உசாத்துணைகள்", "ஆதாரங்கள்", "ஆதாரம்", "இவற்றையும் பார்க்க", "இவற்றையும் பார்க்கவும்",
        "மேலும் பார்க்க", "சான்று", "அடிக்குறிப்புகள்", "see also", "references", "external links", "sources"}
HEAD = re.compile(r"^(={2,6})\s*(.*?)\s*\1\s*$", re.M)

def cut_sections(text):
    """Remove the sections whose heading is in STOP (up to the next heading of the same or a higher level)."""
    out, pos, skip = [], 0, 0
    for m in HEAD.finditer(text):
        if not skip: out.append(text[pos:m.start()])
        lvl = len(m.group(1)); name = re.sub(r"[\[\]']", "", m.group(2)).strip().lower()
        if skip and lvl <= skip: skip = 0
        if not skip and (name in STOP or "இணைப்பு" in name): skip = lvl
        if not skip: out.append(m.group(0))
        pos = m.end()
    if not skip: out.append(text[pos:])
    return "".join(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump"); ap.add_argument("out"); ap.add_argument("wiki")
    ap.add_argument("--min-tamil", type=float, default=0.8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tot = collections.Counter(); lost = collections.Counter()
    with open(f"{a.out}/clean_stats.tsv", "w", encoding="utf-8") as rep:
        rep.write("id\tstatus\tparas\twords\ttitle\n")
        for pid, title, ns, text, red in wikitext.dump_pages(a.dump):
            if ns != 0 or red: continue
            paras = ["\n".join(x for x in (tidy(l) for l in p.split("\n")) if x)
                     for p in wikitext.strip(cut_sections(text), lost)]
            kept = [p for p in paras if p and is_tamil(p, a.min_tamil)]
            while kept and nwords(kept[-1]) <= 3 and not re.search(r"[.!?:]", kept[-1]): kept.pop()   # heading with nothing under it
            words = nwords("\n".join(kept))
            status = "kept" if words >= 30 else "dropped: under 30 Tamil words"
            if status == "kept":
                head = [title] if is_tamil(title, a.min_tamil) and a.wiki != "wikibooks" else []
                open(f"{a.out}/{pid}.txt", "w", encoding="utf-8").write("\n\n".join(head + kept) + "\n")
            tot[status] += 1; tot["words kept"] += words if status == "kept" else 0
            rep.write(f"{pid}\t{status}\t{len(kept)}\t{words}\t{title}\n")
    with open(f"{a.out}/dropped_templates.tsv", "w", encoding="utf-8") as o:
        for k, v in lost.most_common(): o.write(f"{v}\t{k}\n")
    for k, v in sorted(tot.items()): print(k, v)
    print("Tamil words inside dropped templates:", sum(lost.values()), lost.most_common(20))

if __name__ == "__main__":
    main()
