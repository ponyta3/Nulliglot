"""Build the learnability probe: a task whose answer is fully given in the prompt.

    python make_copy_probe.py --source data/splits/train.json --out-dir data/probe_copy

Every row asks the model to repeat a Bulgarian passage:

    instruction  "Повторете следния текст дословно."
    input        <passage>
    output       <the same passage>

The passages are answers from the Alpaca training split, so the language is the
same as in the main run and nothing comes from the validation or test rows.
If a model trained on this still cannot copy, it cannot read its prompt and no
adapter setting will make its answers relevant.
"""

import argparse
import json
import random
from pathlib import Path

INSTRUCTION = "Повторете следния текст дословно."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="data/splits/train.json")
    ap.add_argument("--out-dir", default="data/probe_copy")
    ap.add_argument("--n-train", type=int, default=10000)
    ap.add_argument("--n-val", type=int, default=300)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--min-chars", type=int, default=40)
    ap.add_argument("--max-chars", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = json.loads(Path(args.source).read_text(encoding="utf-8"))
    texts = sorted({r["output"].strip() for r in rows
                    if args.min_chars <= len(r["output"].strip()) <= args.max_chars})
    random.Random(args.seed).shuffle(texts)
    need = args.n_train + args.n_val + args.n_test
    if len(texts) < need:
        raise SystemExit(f"only {len(texts)} passages, need {need}")

    def make(ts):
        return [{"instruction": INSTRUCTION, "input": t, "output": t} for t in ts]

    parts = {
        "test": make(texts[:args.n_test]),
        "val": make(texts[args.n_test:args.n_test + args.n_val]),
        "train": make(texts[args.n_test + args.n_val:need]),
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, part in parts.items():
        (out / f"{name}.json").write_text(
            json.dumps(part, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{name}: {len(part)} rows")
    print(f"passages available: {len(texts)}; written to {out}")


if __name__ == "__main__":
    main()
