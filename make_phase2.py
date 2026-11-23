"""Build the Phase 2 data: the answer depends on the instruction, not only the passage.

    python make_phase2.py --source data/splits_raw/train.json --out-dir data/phase2

Phase 1 taught the model to read a passage. Here the same passage needs a
different answer for each instruction, and many instructions carry a word the
model must find ("replace the word X with Y", "which word follows X?"). Some
tasks put the passage inside the instruction with no input block, the way most
Alpaca rows are written. Every task has several wordings.

All text is Bulgarian, cut from the Alpaca training split with the same
source-row division as make_stage_b.py, so the Phase 1 validation and test
passages stay unseen.

Writes train.json, val.json and test.json. In test.json every held-out
passage appears once per applicable task (field "passage_id"), which shows
directly whether the answer changes when only the instruction changes.
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

from make_stage_b import passages

WORD = re.compile(r"\w+", re.UNICODE)
CYR = re.compile(r"^[А-Яа-яѝЍ]{4,}$")

# task -> (weight, wordings). {p} passage, {w} a word in it, {y} a replacement word.
TASKS = {
    # Phase 1 tasks, kept so the skill is not forgotten
    "copy": (4, ["Повторете следния текст дословно.", "Препишете текста без промени.",
                 "Напишете отново същия текст."]),
    "first_sentence": (2, ["Напишете само първото изречение от следния текст.",
                           "Кое е първото изречение в текста?"]),
    "last_sentence": (2, ["Напишете само последното изречение от следния текст.",
                          "Кое е последното изречение в текста?"]),
    "first_words": (1, ["Напишете само първите три думи от следния текст.",
                        "Кои са първите три думи в текста?"]),
    "last_words": (2, ["Напишете само последните три думи от следния текст.",
                       "Кои са последните три думи в текста?"]),
    # new: position questions
    "second_sentence": (2, ["Напишете само второто изречение от следния текст.",
                            "Кое е второто изречение в текста?"]),
    "first_word": (1, ["Коя е първата дума в текста?", "Напишете само първата дума от текста."]),
    "last_word": (2, ["Коя е последната дума в текста?",
                      "Напишете само последната дума от текста."]),
    # new: the instruction names a word to find
    "sentence_with_word": (4, ["Напишете изречението, което съдържа думата „{w}“.",
                               "В кое изречение се среща думата „{w}“?"]),
    "word_after": (3, ["Коя дума следва след думата „{w}“ в текста?",
                       "Напишете думата, която идва веднага след „{w}“."]),
    "word_before": (3, ["Коя дума стои преди думата „{w}“ в текста?",
                        "Напишете думата, която идва точно преди „{w}“."]),
    # new: edits
    "replace_word": (4, ["Заменете думата „{w}“ с „{y}“ в следния текст.",
                         "Препишете текста, като вместо „{w}“ напишете „{y}“."]),
    "delete_word": (3, ["Премахнете думата „{w}“ от следния текст.",
                        "Препишете текста без думата „{w}“."]),
    "delete_first_sentence": (3, ["Премахнете първото изречение от следния текст.",
                                  "Препишете текста без първото изречение."]),
    "delete_last_sentence": (3, ["Премахнете последното изречение от следния текст.",
                                 "Препишете текста без последното изречение."]),
    "reverse_sentences": (3, ["Напишете изреченията от текста в обратен ред.",
                              "Разменете реда на изреченията в текста."]),
    # new: passage inside the instruction, no input block
    "copy_inline": (4, ["Повторете изречението: {p}", "Напишете отново следното: {p}"]),
    "first_word_inline": (1, ["Коя е първата дума в изречението: {p}"]),
    "word_after_inline": (3, ["Коя дума следва след „{w}“ в изречението: {p}",
                              "Напишете думата след „{w}“ в следното изречение: {p}"]),
    "replace_inline": (4, ["Заменете „{w}“ с „{y}“ в изречението: {p}",
                           "Напишете следното изречение, но с „{y}“ вместо „{w}“: {p}"]),
}
INLINE = {"copy_inline", "first_word_inline", "word_after_inline", "replace_inline"}


def tidy(text):
    return re.sub(r"\s+([,.;:!?])", r"\1", " ".join(text.split())).strip()


def candidates(passage, sents, rng, vocab):
    """Every task this passage supports -> (instruction arguments, output)."""
    words = WORD.findall(passage)
    low = Counter(w.lower() for w in words)
    unique = [i for i, w in enumerate(words) if CYR.match(w) and low[w.lower()] == 1
              and len(re.findall(rf"(?<!\w){re.escape(w)}(?!\w)", passage)) == 1]
    out = {
        "copy": ({}, passage),
        "first_words": ({}, " ".join(passage.split()[:3])),
        "last_words": ({}, " ".join(passage.split()[-3:])),
        "first_word": ({}, words[0]),
        "last_word": ({}, words[-1]),
    }
    if len(passage) <= 200:
        out["copy_inline"] = ({}, passage)
        out["first_word_inline"] = ({}, words[0])
    if len(sents) >= 2:
        out["first_sentence"] = ({}, sents[0])
        out["last_sentence"] = ({}, sents[-1])
        out["delete_first_sentence"] = ({}, " ".join(sents[1:]))
        out["delete_last_sentence"] = ({}, " ".join(sents[:-1]))
        out["reverse_sentences"] = ({}, " ".join(reversed(sents)))
    if len(sents) >= 3:
        out["second_sentence"] = ({}, sents[1])
    if unique:
        i = rng.choice(unique)
        w = words[i]
        y = rng.choice(vocab)
        pattern = rf"(?<!\w){re.escape(w)}(?!\w)"
        out["replace_word"] = ({"w": w, "y": y}, re.sub(pattern, y, passage))
        deleted = tidy(re.sub(pattern, "", passage))
        if deleted:
            out["delete_word"] = ({"w": w}, deleted)
        if i + 1 < len(words):
            out["word_after"] = ({"w": w}, words[i + 1])
        if i > 0:
            out["word_before"] = ({"w": w}, words[i - 1])
        if len(sents) >= 2:
            out["sentence_with_word"] = ({"w": w}, next(s for s in sents if re.search(pattern, s)))
        if len(passage) <= 200:
            out["replace_inline"] = out["replace_word"]
            if "word_after" in out:
                out["word_after_inline"] = out["word_after"]
    return out


def make_row(task, passage, cand, rng, passage_id=None):
    fmt, output = cand
    instruction = rng.choice(TASKS[task][1]).format(p=passage, **fmt)
    row = {"instruction": instruction, "input": "" if task in INLINE else passage,
           "output": output, "task": task}
    if passage_id is not None:
        row["passage_id"] = passage_id
    return row


def unique_passages(texts, min_chars, max_chars, seen):
    for text in texts:
        for p, sents in passages(text, min_chars, max_chars):
            # every sentence must be a real one: list numbers such as "3." are not
            if (p not in seen and len(p.split()) >= 6
                    and all(len(WORD.findall(x)) >= 3 for x in sents)):
                seen.add(p)
                yield p, sents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="data/splits_raw/train.json")
    ap.add_argument("--out-dir", default="data/phase2")
    ap.add_argument("--n-train", type=int, default=640000)
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--n-test-passages", type=int, default=30)
    ap.add_argument("--held-out-sources", type=int, default=1500)
    ap.add_argument("--min-chars", type=int, default=30)
    ap.add_argument("--max-chars", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Same source-text order and division as make_stage_b.py.
    rng = random.Random(args.seed)
    src = json.loads(Path(args.source).read_text(encoding="utf-8"))
    texts = sorted({t.strip() for r in src for t in (r["output"], r.get("input", "")) if t.strip()})
    rng.shuffle(texts)
    k = args.held_out_sources
    held_test, held_val, train_texts = texts[:k], texts[k:2 * k], texts[2 * k:]

    vocab = sorted({w for t in train_texts[:5000] for w in WORD.findall(t) if CYR.match(w)})
    seen = set()

    def one_task_rows(source_texts, limit):
        rows = []
        pool = list(unique_passages(source_texts, args.min_chars, args.max_chars, seen))
        rng.shuffle(pool)
        for p, sents in pool:
            cands = candidates(p, sents, rng, vocab)
            names = list(cands)
            task = rng.choices(names, [TASKS[n][0] for n in names])[0]
            rows.append(make_row(task, p, cands[task], rng))
            if len(rows) >= limit:
                break
        return rows

    # Test: passages with three sentences, every applicable task for each.
    test_rows, pid = [], 0
    pool = [x for x in unique_passages(held_test, args.min_chars, args.max_chars, seen)
            if len(x[1]) == 3 and len(x[0]) <= 200]
    rng.shuffle(pool)
    for p, sents in pool:
        cands = candidates(p, sents, rng, vocab)
        if len(cands) < len(TASKS):
            continue
        test_rows += [make_row(t, p, cands[t], rng, pid) for t in TASKS]
        pid += 1
        if pid >= args.n_test_passages:
            break

    parts = {"test": test_rows,
             "val": one_task_rows(held_val, args.n_val),
             "train": one_task_rows(train_texts, args.n_train)}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, part in parts.items():
        (out / f"{name}.json").write_text(
            json.dumps(part, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{name}: {len(part)} rows")
    counts = Counter(r["task"] for r in parts["train"])
    for t in TASKS:
        print(f"  {t:22s} {counts[t]:7d}  {100 * counts[t] / len(parts['train']):5.1f}%")
    print(f"test passages: {pid}, each with all {len(TASKS)} tasks")


if __name__ == "__main__":
    main()
