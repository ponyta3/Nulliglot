"""Build the Stage B data: tasks whose answer is written in the prompt.

    python make_stage_b.py --source data/splits_raw/train.json --out-dir data/stage_b

Every row gives the model a Bulgarian passage as the input and asks for text
that can be read straight off it:

    copy            repeat the whole passage
    first_sentence  only its first sentence
    last_sentence   only its last sentence
    first_words     only its first three words
    last_words      only its last three words

Passages are one to three consecutive sentences cut from the answers and
inputs of the Alpaca training split (never its validation or test rows).
Source rows are divided between train, validation and test before cutting, so
no sentence is shared between them.

Writes train.json, val.json, test_copy.json (copy rows only, comparable with
the earlier copy probe) and test_other.json (the four extraction tasks).
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

INSTRUCTIONS = {
    "copy": "Повторете следния текст дословно.",
    "first_sentence": "Напишете само първото изречение от следния текст.",
    "last_sentence": "Напишете само последното изречение от следния текст.",
    "first_words": "Напишете само първите три думи от следния текст.",
    "last_words": "Напишете само последните три думи от следния текст.",
}
SENT_END = re.compile(r"(?<=[.!?])\s+")


def passages(text, min_chars, max_chars):
    """All windows of 1-3 consecutive sentences within the length limits."""
    sents = [s.strip() for s in SENT_END.split(" ".join(text.split())) if s.strip()]
    for i in range(len(sents)):
        for n in (1, 2, 3):
            if i + n <= len(sents):
                p = " ".join(sents[i:i + n])
                if min_chars <= len(p) <= max_chars:
                    yield p, sents[i:i + n]


def make_row(task, passage, sents):
    words = passage.split()
    out = {
        "copy": passage,
        "first_sentence": sents[0],
        "last_sentence": sents[-1],
        "first_words": " ".join(words[:3]),
        "last_words": " ".join(words[-3:]),
    }[task]
    return {"instruction": INSTRUCTIONS[task], "input": passage, "output": out, "task": task}


def build(texts, rng, min_chars, max_chars, seen):
    rows = []
    for text in texts:
        for p, sents in passages(text, min_chars, max_chars):
            if p in seen or len(p.split()) < 6:
                continue
            seen.add(p)
            if len(sents) == 1:
                task = rng.choices(["copy", "first_words", "last_words"], [6, 2, 2])[0]
            else:
                task = rng.choices(list(INSTRUCTIONS), [4, 2, 2, 1, 1])[0]
            rows.append(make_row(task, p, sents))
    rng.shuffle(rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="data/splits_raw/train.json")
    ap.add_argument("--out-dir", default="data/stage_b")
    ap.add_argument("--n-train", type=int, default=128000)
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--n-test", type=int, default=200, help="rows in each test file")
    ap.add_argument("--held-out-sources", type=int, default=1500)
    ap.add_argument("--min-chars", type=int, default=30)
    ap.add_argument("--max-chars", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    src = json.loads(Path(args.source).read_text(encoding="utf-8"))
    texts = sorted({t.strip() for r in src for t in (r["output"], r.get("input", "")) if t.strip()})
    rng.shuffle(texts)
    k = args.held_out_sources
    seen = set()
    test_rows = build(texts[:k], rng, args.min_chars, args.max_chars, seen)
    val_rows = build(texts[k:2 * k], rng, args.min_chars, args.max_chars, seen)
    train_rows = build(texts[2 * k:], rng, args.min_chars, args.max_chars, seen)

    parts = {
        "train": train_rows[:args.n_train],
        "val": val_rows[:args.n_val],
        "test_copy": [r for r in test_rows if r["task"] == "copy"][:args.n_test],
        "test_other": [r for r in test_rows if r["task"] != "copy"][:args.n_test],
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, part in parts.items():
        (out / f"{name}.json").write_text(
            json.dumps(part, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{name}: {len(part)} rows  {dict(Counter(r['task'] for r in part))}")
    print(f"source texts: {len(texts)}; train passages available: {len(train_rows)}")


if __name__ == "__main__":
    main()
