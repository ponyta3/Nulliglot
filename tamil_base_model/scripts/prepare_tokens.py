"""Turn the three splits of the final corpus into token-id files for pre-training.

    python prepare_tokens.py <final_dir> <splits_dir> <tokenizer_dir> <out_dir> [--limit-docs N]

For every split (train, val, test) writes
  <split>.bin      the token ids of all documents, one after the other, as 16-bit integers
  <split>.idx.npy  where each document starts in <split>.bin, plus the total length at the end
and once
  meta.json        counts per split and source, and the SHA-256 of the tokenizer file

Each document is stored as: begin-of-text, its tokens, end-of-text. So the model sees the stop
token after every document during pre-training (the Bulgarian model never saw it), and "a
document starts here" is marked the same way as at generation time.

The documents of a split are stored in a random order that is fixed by the seed, so a training
window that runs across a document border joins two unrelated documents, not two neighbours of
the same source.

Checks made on every document; the script stops on the first failure:
- the ids decode back to exactly the same text (after the tokenizer's own normalisation);
- no special token (ids 0 to 15) occurs inside a document. This would happen if the corpus text
  held a string such as "<|end_of_text|>".

--limit-docs keeps N documents per split, for tests.
"""
import argparse, hashlib, json, os, random, sys, time
import numpy as np
from tokenizers import Tokenizer

NSPECIAL, BOS, EOS = 16, 1, 2
BATCH_CHARS = 20_000_000          # documents are encoded in batches of about this many characters


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def encode_split(tok, final_dir, rels, out_bin):
    """Write the ids of all documents to out_bin. Returns the document starts and the counts."""
    norm = tok.normalizer.normalize_str
    starts, n, per_src = [], 0, {}
    stats = {"docs": 0, "tokens": 0, "chars": 0, "byte_tokens": 0}

    def flush(batch, o):
        nonlocal n
        encs = tok.encode_batch([t for _, t in batch], add_special_tokens=False)
        for (rel, t), e in zip(batch, encs):
            ids = np.asarray(e.ids, dtype=np.int64)
            if ids.size == 0 or ids.min() < NSPECIAL:
                sys.exit(f"{rel}: empty, or a special token occurs inside the document")
            if tok.decode(e.ids, skip_special_tokens=False) != t:
                sys.exit(f"{rel}: the token ids do not decode back to the same text")
            starts.append(n)
            np.concatenate(([BOS], ids, [EOS])).astype(np.uint16).tofile(o)
            n += ids.size + 2
            s = per_src.setdefault(rel.split("/")[0], {"docs": 0, "tokens": 0, "chars": 0})
            nb = int((ids < NSPECIAL + 256).sum())
            for d in (s, stats):
                d["docs"] += 1; d["tokens"] += ids.size + 2; d["chars"] += len(t)
            stats["byte_tokens"] += nb

    with open(out_bin, "wb") as o:
        batch, size = [], 0
        for rel in rels:
            t = norm(open(os.path.join(final_dir, rel), encoding="utf-8").read().strip())
            if not t: continue
            batch.append((rel, t)); size += len(t)
            if size >= BATCH_CHARS:
                flush(batch, o); batch, size = [], 0
        if batch: flush(batch, o)
    return np.asarray(starts + [n], dtype=np.int64), stats, per_src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("final_dir"); ap.add_argument("splits_dir"); ap.add_argument("tokenizer_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--limit-docs", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    tok_file = os.path.join(a.tokenizer_dir, "tokenizer.json")
    tok = Tokenizer.from_file(tok_file)
    V = tok.get_vocab_size()
    assert V <= 65536, "token ids do not fit in 16 bits"
    assert tok.token_to_id("<|begin_of_text|>") == BOS and tok.token_to_id("<|end_of_text|>") == EOS
    os.makedirs(a.out_dir, exist_ok=True)
    if os.path.exists(os.path.join(a.out_dir, "meta.json")):
        sys.exit(f"{a.out_dir} already holds token files; not overwriting")
    meta = {"tokenizer": os.path.abspath(a.tokenizer_dir), "tokenizer_sha256": sha256(tok_file),
            "vocab_size": V, "bos": BOS, "eos": EOS, "seed": a.seed, "limit_docs": a.limit_docs, "splits": {}}
    for split in ("train", "val", "test"):
        t0 = time.time()
        rels = sorted(x for x in open(os.path.join(a.splits_dir, split + ".txt"), encoding="utf-8").read().split("\n") if x)
        random.Random(f"{a.seed}-{split}").shuffle(rels)
        if a.limit_docs: rels = rels[:a.limit_docs]
        idx, stats, per_src = encode_split(tok, a.final_dir, rels, os.path.join(a.out_dir, split + ".bin"))
        np.save(os.path.join(a.out_dir, split + ".idx.npy"), idx)
        # read back: length, begin and end marks at every document border
        d = np.fromfile(os.path.join(a.out_dir, split + ".bin"), dtype=np.uint16)
        assert d.size == idx[-1] == stats["tokens"] and int(d.max()) < V
        assert (d[idx[:-1]] == BOS).all() and (d[idx[1:] - 1] == EOS).all()
        assert int((d == BOS).sum()) == int((d == EOS).sum()) == stats["docs"]
        stats["sources"] = per_src
        meta["splits"][split] = stats
        print(f"{split}: {stats['docs']} documents, {stats['tokens']} tokens, {stats['chars']} characters, "
              f"{1000 * stats['tokens'] / stats['chars']:.1f} tokens per 1,000 characters, "
              f"{100 * stats['byte_tokens'] / stats['tokens']:.4f}% byte tokens, {time.time() - t0:.0f} s", flush=True)
    with open(os.path.join(a.out_dir, "meta.json"), "w") as o:
        json.dump(meta, o, indent=1)


if __name__ == "__main__":
    main()
