"""Evaluate the base model against base + adapter on the held-out test rows.

    python eval_test.py --model models/bul_100m --adapter adapters/bul_instruct \
        --test data/splits/test.json --out adapters/bul_instruct/test_eval.json

For each system it reports, over the test rows:

  stopped        share of generations that end with <|end_of_text|>
  cyrillic_only  share of generations with no Latin letters
  avg_tokens     mean number of generated tokens
  overlap_true   word-overlap F1 between the generation and its reference answer
  overlap_wrong  the same against another row's reference (chance level)
  loss_true      loss on the reference answer given its own prompt
  loss_wrong     loss on the reference answer given another row's prompt

A model that uses the instruction shows overlap_true well above overlap_wrong
and loss_wrong well above loss_true. All generations are saved for reading.
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from instruct_prompt import encode_example, encode_prompt, special_ids
from stop_token import load_instruct_model
from train_lora import evaluate, pick_device

LATIN = re.compile(r"[A-Za-z]")
WORD = re.compile(r"\w+", re.UNICODE)


def overlap_f1(a, b):
    """Word-level F1 between two texts (lower-cased, order ignored)."""
    wa, wb = Counter(WORD.findall(a.lower())), Counter(WORD.findall(b.lower()))
    common = sum((wa & wb).values())
    if common == 0:
        return 0.0
    p, r = common / sum(wa.values()), common / sum(wb.values())
    return 2 * p * r / (p + r)


@torch.no_grad()
def generate_all(model, tok, rows, lang, max_len, max_new, device):
    _, eos_id = special_ids(tok)
    outs = []
    for row in rows:
        ids = encode_prompt(tok, row["instruction"], row.get("input", ""), lang)
        budget = min(max_new, max_len - len(ids))
        out = model.generate(
            input_ids=torch.tensor([ids], device=device),
            attention_mask=torch.ones(1, len(ids), dtype=torch.long, device=device),
            max_new_tokens=budget,
            do_sample=False,
            repetition_penalty=1.1,
            eos_token_id=eos_id,
            pad_token_id=eos_id,
        )[0][len(ids):].tolist()
        outs.append({
            "text": tok.decode(out, skip_special_tokens=True).strip(),
            "stopped": eos_id in out,
            "new_tokens": len(out),
        })
    return outs


def score(model, tok, rows, wrong, lang, max_len, max_new, device):
    _, eos_id = special_ids(tok)
    model.to(device).eval()
    gens = generate_all(model, tok, rows, lang, max_len, max_new, device)

    def examples(rs):
        ex = [encode_example(tok, r, lang) for r in rs]
        return [e for e in ex if len(e[0]) <= max_len]

    wrong_prompt_rows = [
        {"instruction": w["instruction"], "input": w.get("input", ""), "output": r["output"]}
        for r, w in zip(rows, wrong)
    ]
    n = len(rows)
    summary = {
        "n": n,
        "stopped": sum(g["stopped"] for g in gens) / n,
        "cyrillic_only": sum(not LATIN.search(g["text"]) for g in gens) / n,
        "empty": sum(not g["text"] for g in gens) / n,
        "avg_tokens": sum(g["new_tokens"] for g in gens) / n,
        "overlap_true": sum(overlap_f1(g["text"], r["output"]) for g, r in zip(gens, rows)) / n,
        "overlap_wrong": sum(overlap_f1(g["text"], w["output"]) for g, w in zip(gens, wrong)) / n,
        "loss_true": evaluate(model, examples(rows), 16, eos_id, device),
        "loss_wrong": evaluate(model, examples(wrong_prompt_rows), 16, eos_id, device),
    }
    summary["loss_gap"] = summary["loss_wrong"] - summary["loss_true"]
    return summary, gens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="iamshnoo/bul_100m")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--test", default="data/splits/test.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang", default="bul", choices=["bul", "deu"])
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N rows")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    bos_id, eos_id = special_ids(tok)
    rows = json.loads(Path(args.test).read_text(encoding="utf-8"))[:args.limit]

    # Pair every row with a different row (a rotation of a shuffled order).
    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    partner = {order[i]: order[(i + 1) % len(order)] for i in range(len(order))}
    wrong = [rows[partner[i]] for i in range(len(rows))]

    base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    base.generation_config.bos_token_id = bos_id
    base.generation_config.eos_token_id = eos_id
    base.generation_config.pad_token_id = eos_id
    systems = {"base": base}
    results, generations = {}, {}
    for name in ("base", "adapter"):
        if name == "adapter":
            del systems["base"]
            systems["adapter"] = load_instruct_model(args.model, args.adapter)
        results[name], generations[name] = score(
            systems[name], tok, rows, wrong, args.lang, args.max_len, args.max_new, device)
        print(name, json.dumps(results[name], indent=1))

    per_row = [
        {
            "instruction": r["instruction"], "input": r.get("input", ""), "reference": r["output"],
            "base": generations["base"][i], "adapter": generations["adapter"][i],
        }
        for i, r in enumerate(rows)
    ]
    Path(args.out).write_text(
        json.dumps({"args": vars(args), "summary": results, "rows": per_row},
                   ensure_ascii=False, indent=1),
        encoding="utf-8")

    print("\n%-15s %10s %10s" % ("metric", "base", "adapter"))
    for k in results["base"]:
        print("%-15s %10.4f %10.4f" % (k, results["base"][k], results["adapter"][k]))
    print(f"\nsaved {len(rows)} rows with both generations to {args.out}")


if __name__ == "__main__":
    main()
