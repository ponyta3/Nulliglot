"""Shuffle the instruction data and hold out validation and test rows.

    python split_data.py

Writes data/splits/{train,val,test}.json plus split_info.json.

Two things beyond a plain shuffle:

* Held-out rows are drawn only from rows that fit in the model's 512-token
  context once rendered with the prompt template, so every test row can
  actually be evaluated. Over-length rows go to train.json, where the training
  script's own length filter drops them.
* Rows sharing the same (instruction, input) are kept on the same side of the
  split, so no held-out prompt also appears in the training set.
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from instruct_prompt import encode_example


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/alpaca_bg_clean_final.json")
    ap.add_argument("--out-dir", default="data/splits")
    ap.add_argument("--model", default="iamshnoo/bul_100m")
    ap.add_argument("--lang", default="bul", choices=["bul", "deu"])
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = json.loads(Path(args.data).read_text(encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(args.model)

    # Group duplicates of the same prompt; a group is one unit in the shuffle.
    groups = defaultdict(list)
    for row in rows:
        groups[(row["instruction"].strip(), row["input"].strip())].append(row)
    units = list(groups.values())

    def fits(unit):
        return all(len(encode_example(tok, r, args.lang)[0]) <= args.max_len for r in unit)

    random.Random(args.seed).shuffle(units)

    test, val, train = [], [], []
    n_over = 0
    for unit in units:
        ok = fits(unit)
        n_over += 0 if ok else len(unit)
        if ok and len(unit) == 1 and len(test) < args.n_test:
            test += unit
        elif ok and len(unit) == 1 and len(val) < args.n_val:
            val += unit
        else:
            train += unit

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("val", val), ("test", test)):
        (out_dir / f"{name}.json").write_text(
            json.dumps(part, ensure_ascii=False, indent=1), encoding="utf-8"
        )

    info = {
        "source": args.data,
        "model": args.model,
        "seed": args.seed,
        "max_len": args.max_len,
        "rows_total": len(rows),
        "train": len(train),
        "train_over_max_len": n_over,
        "train_usable": len(train) - n_over,
        "val": len(val),
        "test": len(test),
    }
    (out_dir / "split_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
