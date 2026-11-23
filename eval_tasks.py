"""Per-task scores for a tuned model on a test file whose rows carry a "task" field.

    python eval_tasks.py --adapter adapters/phase2_lora/best \
        --test data/phase2/test.json --out adapters/phase2_lora/test_tasks_eval.json

For every task it reports

  exact      share of answers identical to the reference
  overlap    word-overlap F1 with the reference
  as_copy    the overlap a model would get by just repeating the passage

A model that follows the instruction scores well above as_copy on the tasks
whose answer is not the whole passage. Only the tuned model is run.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from eval_test import generate_all, overlap_f1
from stop_token import load_instruct_model
from train_lora import pick_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="iamshnoo/bul_100m")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang", default="bul", choices=["bul", "deu"])
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    rows = json.loads(Path(args.test).read_text(encoding="utf-8"))[:args.limit]
    model = load_instruct_model(args.model, args.adapter).to(device).eval()
    gens = generate_all(model, tok, rows, args.lang, args.max_len, args.max_new, device)

    stats = defaultdict(lambda: defaultdict(float))
    per_row = []
    for r, g in zip(rows, gens):
        passage = r.get("input") or r["instruction"]
        s = stats[r.get("task", "all")]
        s["n"] += 1
        s["exact"] += g["text"].strip() == r["output"].strip()
        s["overlap"] += overlap_f1(g["text"], r["output"])
        s["as_copy"] += overlap_f1(passage, r["output"])
        s["stopped"] += g["stopped"]
        per_row.append({"task": r.get("task"), "instruction": r["instruction"],
                        "input": r.get("input", ""), "reference": r["output"], "answer": g})

    summary = {t: {k: (v / s["n"] if k != "n" else int(v)) for k, v in s.items()}
               for t, s in stats.items()}
    n = len(rows)
    summary["ALL"] = {"n": n, **{k: sum(s[k] for s in stats.values()) / n
                                 for k in ("exact", "overlap", "as_copy", "stopped")}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"args": vars(args), "summary": summary, "rows": per_row},
                                         ensure_ascii=False, indent=1), encoding="utf-8")

    print("%-24s %4s %7s %8s %8s %8s" % ("task", "n", "exact", "overlap", "as_copy", "stopped"))
    for t, s in summary.items():
        print("%-24s %4d %7.3f %8.3f %8.3f %8.3f"
              % (t, s["n"], s["exact"], s["overlap"], s["as_copy"], s["stopped"]))
    print(f"\nsaved {n} rows to {args.out}")


if __name__ == "__main__":
    main()
