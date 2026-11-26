"""Compare the trained tokenizers on held-out text (the validation split).

    python eval_tokenizer.py <final_dir> <val_list> <tokenizer_dir>

Reads every <tokenizer_dir>/bpe_*/ and writes
  REPORT.tsv   one row per vocabulary size
  SOURCES.tsv  tokens per Tamil word, per source and vocabulary size
  samples.txt  the same held-out paragraphs cut by every tokenizer, to read by eye
  vocab_<size>.txt  every token of that tokenizer, in id order

Columns of REPORT.tsv
  tok_per_word       tokens per Tamil word (word types of the validation text, weighted by count)
  one_tok_pct        Tamil words that are a single token
  le2_pct, ge5_pct   Tamil words cut into at most 2 / at least 5 tokens
  tok_per_1000_chars all tokens (spaces, punctuation, digits included) per 1,000 characters
  words_in_512, words_in_1024   Tamil words that fit in a context of that many tokens
  used_pct           tokens of the vocabulary that occur in the validation text
  chars              tokens that are a single character
  tamil, latin, digit, other    longer tokens by what they contain
  byte_use_pct       share of validation tokens that are byte tokens (characters with no token)
  roundtrip_fail     validation documents that do not decode back to the same text
  embed_M            millions of parameters in one embedding table at hidden size 768
"""
import glob, os, re, sys
from collections import Counter
from transformers import AutoTokenizer

final_dir, val_list, tok_dir = sys.argv[1:4]
WORD = re.compile(r" ?[஀-௿]+")
TA, LAT, DIG = re.compile(r"[஀-௿]"), re.compile(r"[A-Za-z]"), re.compile(r"\d")
NSPECIAL, NBYTE = 16, 256

dirs = sorted(glob.glob(os.path.join(tok_dir, "bpe_*", "")), key=lambda d: int(d.rstrip("/").split("_")[-1]))
norm = AutoTokenizer.from_pretrained(dirs[0]).backend_tokenizer.normalizer.normalize_str   # same in all
rels = [x for x in open(val_list, encoding="utf-8").read().split("\n") if x]
docs = [(r, norm(open(os.path.join(final_dir, r), encoding="utf-8").read().strip())) for r in rels]
docs = [(r, t) for r, t in docs if t]
words = Counter(); by_src = {}
for r, t in docs:
    c = Counter(WORD.findall(t)); words.update(c)
    by_src.setdefault(r.split("/")[0], Counter()).update(c)
types = list(words); nw = sum(words.values()); nchar = sum(len(t) for _, t in docs)
print(f"validation: {len(docs)} documents, {nw} Tamil words, {len(types)} distinct", flush=True)

# sample paragraphs: the first paragraph of 40 to 400 characters after the first five, from
# two documents of every source
samples = []
for src in sorted(by_src):
    k = 0
    for r, t in docs:
        if not r.startswith(src + "/"): continue
        ps = [p for p in t.split("\n\n")[5:] if 40 <= len(p) <= 400] or [p for p in t.split("\n\n") if 40 <= len(p) <= 400]
        if ps: samples.append((r, ps[0])); k += 1
        if k == 2: break

cols = ["vocab", "tok_per_word", "one_tok_pct", "le2_pct", "ge5_pct", "tok_per_1000_chars", "words_in_512",
        "words_in_1024", "used_pct", "chars", "tamil", "latin", "digit", "other", "byte_use_pct",
        "roundtrip_fail", "embed_M"]
report = ["\t".join(cols)]; src_rows = {}; cut = {}
print(report[0], flush=True)
for d in dirs:
    hf = AutoTokenizer.from_pretrained(d)
    tok = hf.backend_tokenizer
    V = tok.get_vocab_size()
    assert len(hf) == V and hf.bos_token_id == 1 and hf.eos_token_id == 2 and hf.pad_token_id == 0
    assert hf("அ")["input_ids"][0] == 1, "begin-of-text token is not added"
    s = "தமிழ் 😀 中文\n\nஅடுத்த பத்தி"                 # through the Hugging Face class, with byte fallback
    assert hf.decode(hf(s)["input_ids"], skip_special_tokens=True) == s, hf.decode(hf(s)["input_ids"])
    # per-word statistics
    enc = tok.encode_batch(types, add_special_tokens=False)
    ln = {w: len(e.ids) for w, e in zip(types, enc)}
    tot = sum(ln[w] * c for w, c in words.items())
    share = lambda f: 100 * sum(c for w, c in words.items() if f(ln[w])) / nw
    src_rows[V] = {s: sum(ln[w] * c for w, c in cw.items()) / sum(cw.values()) for s, cw in by_src.items()}
    # whole documents
    used = Counter(); ntok = 0; fails = 0
    for (r, t), e in zip(docs, tok.encode_batch([t for _, t in docs], add_special_tokens=False)):
        used.update(e.ids); ntok += len(e.ids)
        if tok.decode(e.ids, skip_special_tokens=False) != t: fails += 1
    piece = {i: tok.id_to_token(i) for i in range(V)}
    kinds = Counter()
    for i in range(NSPECIAL + NBYTE, V):
        s = piece[i]
        kinds["chars" if len(s) == 1 else "tamil" if TA.search(s) else "latin" if LAT.search(s)
              else "digit" if DIG.search(s) else "other"] += 1
    nbyte = sum(n for i, n in used.items() if NSPECIAL <= i < NSPECIAL + NBYTE)
    wpt = nw / ntok                                    # Tamil words per token of running text
    row = [V, f"{tot / nw:.3f}", f"{share(lambda n: n == 1):.1f}", f"{share(lambda n: n <= 2):.1f}",
           f"{share(lambda n: n >= 5):.1f}", f"{1000 * ntok / nchar:.1f}", f"{512 * wpt:.0f}", f"{1024 * wpt:.0f}",
           f"{100 * len(used) / V:.1f}", kinds["chars"], kinds["tamil"], kinds["latin"], kinds["digit"],
           kinds["other"], f"{100 * nbyte / ntok:.4f}", fails, f"{V * 768 / 1e6:.1f}"]
    report.append("\t".join(map(str, row))); print(report[-1], flush=True)
    cut[V] = ["|".join(piece[i].replace("\n", "\\n") for i in tok.encode(p, add_special_tokens=False).ids) for _, p in samples]
    with open(os.path.join(tok_dir, f"vocab_{V}.txt"), "w", encoding="utf-8") as o:
        for i in range(V): o.write(f"{i}\t{piece[i]!r}\n")

open(os.path.join(tok_dir, "REPORT.tsv"), "w").write("\n".join(report) + "\n")
srcs = sorted(by_src)
with open(os.path.join(tok_dir, "SOURCES.tsv"), "w") as o:
    o.write("vocab\t" + "\t".join(srcs) + "\n")
    for V, m in src_rows.items(): o.write(str(V) + "\t" + "\t".join(f"{m[s]:.3f}" for s in srcs) + "\n")
with open(os.path.join(tok_dir, "samples.txt"), "w", encoding="utf-8") as o:
    o.write("Tokens are separated by | ; a space at the start of a token belongs to the token.\n\n")
    for k, (r, p) in enumerate(samples):
        o.write(f"=== {r}\n{p}\n")
        for V in cut: o.write(f"--- {V}\n{cut[V][k]}\n")
        o.write("\n")
print(open(os.path.join(tok_dir, "SOURCES.tsv")).read())
