"""Compare decoding settings for one adapter on the Alpaca test rows.

    python eval_decode.py --adapter adapters/phase3_lora/best \
        --out adapters/phase3_lora/decode_eval.json

Each setting is NAME:rep_penalty:no_repeat:boost:own_script, for example
"guided:1.1:3:2:1". The same measures as eval_pilot.py are reported, plus

  loops   share of answers in which some 3-word sequence occurs 3+ times

Boosting the question's words raises on_topic by construction, so judge a
boost by overlap_true against overlap_wrong and by reading the answers.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from eval_pilot import summarize
from eval_test import WORD
from generate import answer, latin_token_ids
from stop_token import load_instruct_model
from train_lora import pick_device

DEFAULT = ["baseline:1.1:0:0:0", "norepeat:1.1:3:0:0", "script:1.1:3:0:1",
           "boost1:1.1:3:1:1", "boost2:1.1:3:2:1", "boost4:1.1:3:4:1"]


def loops(text):
    w = WORD.findall(text.lower())
    grams = Counter(tuple(w[i:i + 3]) for i in range(len(w) - 2))
    return bool(grams) and max(grams.values()) >= 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="iamshnoo/bul_100m")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--test", default="data/pilot/plain/test.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--setting", action="append", default=[])
    ap.add_argument("--lang", default="bul", choices=["bul", "deu"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    rows = json.loads(Path(args.test).read_text(encoding="utf-8"))[:args.limit]
    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    partner = {order[i]: order[(i + 1) % len(order)] for i in range(len(order))}
    wrong = [rows[partner[i]] for i in range(len(rows))]
    model = load_instruct_model(args.model, args.adapter).to(device).eval()
    latin = latin_token_ids(tok)

    results, per_row = {}, [{"instruction": r["instruction"], "input": r.get("input", ""),
                             "reference": r["output"]} for r in rows]
    for spec in args.setting or DEFAULT:
        name, rep, ngram, boost, own = spec.split(":")
        gens = [answer(model, tok, r["instruction"], r.get("input", ""), args.lang,
                       rep_penalty=float(rep), no_repeat=int(ngram), boost=float(boost),
                       banned_ids=latin if int(own) else None, device=device) for r in rows]
        texts = [g["text"] for g in gens]
        results[name] = {"setting": spec, **summarize(rows, wrong, texts, gens),
                         "loops": sum(loops(t) for t in texts) / len(rows)}
        for p, t in zip(per_row, texts):
            p[name] = t
        s = results[name]
        print("%-10s stop %.3f cyr %.3f loops %.3f tok %5.1f ovl %.3f/%.3f topic %.3f/%.3f"
              % (name, s["stopped"], s["cyrillic_only"], s["loops"], s["avg_tokens"],
                 s["overlap_true"], s["overlap_wrong"], s["on_topic"], s["on_topic_wrong"]),
              flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"args": vars(args), "summary": results, "rows": per_row},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
