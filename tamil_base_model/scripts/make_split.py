"""Split the final corpus into train, validation and test by whole document.

    python make_split.py <final_dir> <out_dir> [--limit-docs N]

Each document (one .txt file) goes to one split, chosen from a hash of its path, so the split is
the same every time and does not change when other documents are added or removed: 98% train,
1% validation, 1% test. Whole documents are held out so that validation text never shares a book
or article with training text.

Writes train.txt, val.txt, test.txt (paths relative to <final_dir>) and SPLIT.tsv (documents and
Tamil words per source and split). --limit-docs keeps N documents per source, for tests.
"""
import argparse, glob, hashlib, os, re

TAW = re.compile(r"[஀-௿]+")


def split_of(rel):
    b = int(hashlib.sha1(rel.encode("utf-8")).hexdigest(), 16) % 100
    return "val" if b == 0 else "test" if b == 1 else "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("final_dir"); ap.add_argument("out_dir")
    ap.add_argument("--limit-docs", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    lists = {"train": [], "val": [], "test": []}
    rows = []
    for d in sorted(glob.glob(os.path.join(a.final_dir, "*", ""))):
        src = os.path.basename(d[:-1])
        files = sorted(glob.glob(d + "*.txt"))
        if a.limit_docs: files = files[:a.limit_docs]
        n = {k: 0 for k in lists}; w = {k: 0 for k in lists}
        for f in files:
            t = open(f, encoding="utf-8").read()
            if not t.strip(): continue
            rel = src + "/" + os.path.basename(f)
            s = split_of(rel)
            lists[s].append(rel); n[s] += 1; w[s] += len(TAW.findall(t))
        rows.append((src, n, w))
    for s, v in lists.items():
        with open(os.path.join(a.out_dir, s + ".txt"), "w", encoding="utf-8") as o:
            o.write("\n".join(v) + "\n")
    head = "source\ttrain_docs\tval_docs\ttest_docs\ttrain_words\tval_words\ttest_words"
    out = [head]
    T = [0] * 6
    for src, n, w in rows:
        v = [n["train"], n["val"], n["test"], w["train"], w["val"], w["test"]]
        T = [x + y for x, y in zip(T, v)]
        out.append(src + "\t" + "\t".join(map(str, v)))
    out.append("TOTAL\t" + "\t".join(map(str, T)))
    open(os.path.join(a.out_dir, "SPLIT.tsv"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
