"""Print random samples from a cleaned source folder, for checking by eye.
    python clean_report.py <clean_dir> [n_files] [chars] [seed]"""
import glob, random, sys, csv, collections
d = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 4; c = int(sys.argv[3]) if len(sys.argv) > 3 else 900
random.seed(int(sys.argv[4]) if len(sys.argv) > 4 else 1)
rows = list(csv.DictReader(open(d + "/clean_stats.tsv", encoding="utf-8"), delimiter="\t"))
print(collections.Counter(r["status"] for r in rows).most_common())
fs = sorted(glob.glob(d + "/*.txt"))
for f in random.sample(fs, min(n, len(fs))):
    t = open(f, encoding="utf-8").read(); i = random.randrange(0, max(1, len(t) - c))
    print("=====", f, len(t)); print(t[:c // 2]); print("  ...  "); print(t[i:i + c])
