"""Clean source 6, FreeTamilEbooks: EPUB files to plain Tamil text, one file per book.

    python clean_fte.py <raw/fte> <out_dir> <fte_translation_flagged.tsv> [--min-tamil 0.8]

A book is kept only when
  - an open licence is stated (CC0, public domain, CC BY, CC BY-SA) on the book's web page, in
    the EPUB metadata or on its copyright page, and none of the three states a non-commercial
    or no-derivatives licence;
  - it is not in the list of translations;
  - fewer than --max-bad of its words have an impossible Tamil letter sequence (damaged text).
Inside a kept book: paragraphs that occur in three or more books (the site's own pages and
notices) are dropped, as are paragraphs that are not mostly Tamil.
Writes <out_dir>/<slug>.txt and <out_dir>/clean_stats.tsv.
"""
import argparse, collections, csv, hashlib, os, posixpath, re, urllib.parse, zipfile
import xml.etree.ElementTree as ET
from textutil import html_paragraphs, is_tamil, nwords, badshape_share, TAW

RESTRICTED = re.compile(r"non[\s-]*com|\bNC\b|no[\s-]*deriv|\bND\b|all rights reserved|royalty", re.I)
OPEN = re.compile(r"CC0|CCO\b|CC[\s-]*zero|creative commons zero|public[\s-]*domain|piblic domain|பொதுக் ?கள|"
                  r"பொதுவுடைமை|CC[\s-]*BY|attribution|attibution|licenses/by(-sa)?/|publicdomain/zero", re.I)

def spine_docs(z):
    opf = [n for n in z.namelist() if n.lower().endswith(".opf")][0]
    root = ET.fromstring(z.read(opf)); base = posixpath.dirname(opf)
    els = lambda name: [e for e in root.iter() if e.tag.rsplit("}", 1)[-1] == name]
    items = {i.get("id"): i.get("href") for i in els("item")}
    rights = " ".join((e.text or "") for e in els("rights"))
    names = set(z.namelist()); docs = []
    for ref in els("itemref"):
        h = items.get(ref.get("idref"))
        if not h: continue
        n = posixpath.normpath(posixpath.join(base, urllib.parse.unquote(h.split("#")[0])))
        if n in names: docs.append(n)
    return docs, rights

def key(p):
    return hashlib.md5("".join(TAW.findall(p)).encode()).digest() if nwords(p) >= 4 else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw"); ap.add_argument("out"); ap.add_argument("flagged")
    ap.add_argument("--min-tamil", type=float, default=0.8)
    ap.add_argument("--max-bad", type=float, default=0.01)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    flagged = {r["slug"] for r in csv.DictReader(open(a.flagged, encoding="utf-8"), delimiter="\t")}
    books = [r for r in csv.DictReader(open(f"{a.raw}/books.tsv", encoding="utf-8"), delimiter="\t")
             if r["status"] == "downloaded"]
    data, seen_in = {}, collections.Counter()
    for r in books:
        slug = r["slug"]
        try:
            z = zipfile.ZipFile(f"{a.raw}/epub/{slug}.epub"); docs, rights = spine_docs(z)
            paras = []
            for n in docs:
                if re.search(r"toc|table-of-contents|nav\.|pressbooks-promo|cover", posixpath.basename(n), re.I): continue
                paras += html_paragraphs(z.read(n).decode("utf-8", "replace"))
        except Exception as e:
            data[slug] = (r, None, f"unreadable: {type(e).__name__}"); continue
        # licence evidence: web page, EPUB metadata, and the book's own licence statement (the
        # site's general pages, which describe every Creative Commons licence, are not counted)
        inbook = " ".join(p for p in paras if len(p) < 300 and re.search(
            r"^\s*உரிமை\s*[:–-]|licen[sc]ed under|உரிமத்தின் கீழ் (பகிர|வெளியி)|உரிமையில் வெளியி", p, re.I))
        ev = " || ".join([r["licence_text"], rights, inbook[:3000]])
        data[slug] = (r, paras, ev)
        for k in {key(p) for p in paras} - {None}: seen_in[k] += 1
    tot = collections.Counter()
    with open(f"{a.out}/clean_stats.tsv", "w", encoding="utf-8") as rep:
        rep.write("slug\tstatus\tparas\tparas_kept\twords\tbad_shape_pct\tlicence_evidence\ttitle\n")
        for slug, (r, paras, ev) in data.items():
            kept, words, bad = [], 0, 0.0
            if paras is None: status = ev; ev = ""
            elif slug in flagged: status = "dropped: translation"
            elif RESTRICTED.search(ev): status = "dropped: restricted licence"
            elif not OPEN.search(ev): status = "dropped: no licence stated"
            else:
                kept = [p for p in paras if seen_in.get(key(p), 0) < 3 and is_tamil(p, a.min_tamil)]
                text = "\n".join(kept); words = nwords(text); bad = badshape_share(text)
                status = "kept"
                if words < 200: status = "dropped: almost no Tamil text"
                elif bad > a.max_bad: status = "dropped: damaged text"
                if status == "kept":
                    open(f"{a.out}/{slug}.txt", "w", encoding="utf-8").write("\n\n".join(kept) + "\n")
            tot[status] += 1; tot["words"] += words if status == "kept" else 0
            rep.write(f"{slug}\t{status}\t{len(paras or [])}\t{len(kept)}\t{words}\t{100*bad:.2f}\t"
                      f"{re.sub(chr(9)+'|'+chr(10), ' ', ev)[:300]}\t{r['title']}\n")
    for k, v in sorted(tot.items()): print(k, v)

if __name__ == "__main__":
    main()
