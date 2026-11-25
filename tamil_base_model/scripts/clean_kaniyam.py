"""Clean source 5, Kaniyam magazine: posts from the site's API to plain Tamil text, one file per post.

    python clean_kaniyam.py <raw/kaniyam> <out_dir> <kaniyam_flagged_posts.tsv> [--min-tamil 0.8]

A post is dropped when it is in the flagged list (translations and the posts of one author
suspected of machine-assisted translation), or when less than half of its text survives the
paragraph filter (English-heavy posts), or when under 100 Tamil words are left.
Inside a kept post: code blocks, tables, image captions and embedded media are removed;
paragraphs found in three or more posts (notices, signatures) are dropped; a paragraph is kept
when at least --min-tamil of its words are in Tamil script. The threshold is lower than for the book
sources because these articles give English technical terms in brackets.
Writes <out_dir>/<post id>.txt and <out_dir>/clean_stats.tsv.
"""
import argparse, collections, csv, glob, hashlib, html, json, os
from textutil import html_paragraphs, is_tamil, nwords, other_share, TAW

SKIP_CLASSES = ("wp-caption", "embed-", "youtube", "blob-", "hljs", "sharedaddy", "jp-relatedposts",
                "wp-block-button", "wp-block-code", "wp-block-embed", "wp-block-table", "gallery", "latex")

def key(p):
    return hashlib.md5("".join(TAW.findall(p)).encode()).digest() if nwords(p) >= 4 else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("raw"); ap.add_argument("out"); ap.add_argument("flagged")
    ap.add_argument("--min-tamil", type=float, default=0.8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    flagged = {r["id"]: r["flag"] for r in csv.DictReader(open(a.flagged, encoding="utf-8"), delimiter="\t")}
    posts = [p for f in sorted(glob.glob(f"{a.raw}/posts_*.json")) for p in json.load(open(f, encoding="utf-8"))]
    data, seen = {}, collections.Counter()
    for p in posts:
        title = html.unescape(p["title"]["rendered"]).strip()
        paras = html_paragraphs(p["content"]["rendered"], SKIP_CLASSES, skip_tables=True)
        data[p["id"]] = (p, title, paras)
        for k in {key(x) for x in paras} - {None}: seen[k] += 1
    tot = collections.Counter(); alltext = []
    with open(f"{a.out}/clean_stats.tsv", "w", encoding="utf-8") as rep:
        rep.write("id\tdate\tauthor\tstatus\tparas\tparas_kept\twords\ttitle\n")
        for pid, (p, title, paras) in data.items():
            body = [x for x in paras if seen.get(key(x), 0) < 3]
            kept = [x for x in body if is_tamil(x, a.min_tamil)]
            chars = sum(len(x) for x in body); kchars = sum(len(x) for x in kept)
            words = nwords("\n".join(kept))
            if str(pid) in flagged: status = "dropped: flagged " + flagged[str(pid)].split("+")[0]
            elif not chars or kchars < 0.5 * chars: status = "dropped: mostly not Tamil"
            elif words < 100: status = "dropped: under 100 Tamil words"
            else:
                status = "kept"
                head = [title] if is_tamil(title, a.min_tamil) and (not kept or title != kept[0]) else []
                text = "\n\n".join(head + kept)
                open(f"{a.out}/{pid}.txt", "w", encoding="utf-8").write(text + "\n")
                alltext.append(text)
            tot[status.split(" ", 2)[0] + " " + status.split(" ", 2)[1] if status != "kept" else "kept"] += 1
            tot["words kept"] += words if status == "kept" else 0
            rep.write(f"{pid}\t{p['date'][:10]}\t{p['author']}\t{status}\t{len(paras)}\t{len(kept)}\t{words}\t"
                      f"{title.replace(chr(9), ' ')}\n")
    for k, v in sorted(tot.items()): print(k, v)
    print(f"words in kept text that are not in Tamil script: {100 * other_share(' '.join(alltext)):.1f}%")

if __name__ == "__main__":
    main()
