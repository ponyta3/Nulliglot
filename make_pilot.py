"""Build the Phase 3 pilot data: Alpaca questions mixed with Phase 2 reading tasks.

    python make_pilot.py --out-dir data/pilot

Two versions of the same rows are written:

    plain/     the answer is the reference answer
    restate/   the answer first repeats the question, then gives the answer:
                   Въпрос: {instruction}
                   Отговор: {answer}

Both hold the Alpaca training rows plus a share of Phase 2 rows, so the skill
of reading the prompt is kept while question answering is learned. Validation
and test rows are the Alpaca validation and test rows.
"""

import argparse
import json
import random
from pathlib import Path

Q, A = "Въпрос: ", "\nОтговор: "


def restate_prefix(instruction):
    """The text a restate-format answer starts with, up to the answer itself."""
    return Q + " ".join(instruction.split()) + A


def restated(row):
    return {**row, "output": restate_prefix(row["instruction"]) + row["output"].strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpaca-dir", default="data/splits_raw")
    ap.add_argument("--reading", default="data/phase2/train.json")
    ap.add_argument("--out-dir", default="data/pilot")
    ap.add_argument("--reading-share", type=float, default=0.25,
                    help="share of training rows that are Phase 2 reading tasks")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    load = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))
    alpaca = {s: load(f"{args.alpaca_dir}/{s}.json") for s in ("train", "val", "test")}
    reading = load(args.reading)
    n_read = round(len(alpaca["train"]) * args.reading_share / (1 - args.reading_share))
    train = [{**r, "task": "alpaca"} for r in alpaca["train"]] + rng.sample(reading, n_read)
    rng.shuffle(train)

    parts = {"train": train, "val": alpaca["val"], "test": alpaca["test"]}
    for fmt, f in (("plain", lambda r: r), ("restate", restated)):
        out = Path(args.out_dir) / fmt
        out.mkdir(parents=True, exist_ok=True)
        for name, rows in parts.items():
            (out / f"{name}.json").write_text(
                json.dumps([f(r) for r in rows], ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"{fmt}/{name}: {len(rows)} rows")
    print(f"training rows: {len(alpaca['train'])} Alpaca + {n_read} reading")


if __name__ == "__main__":
    main()
