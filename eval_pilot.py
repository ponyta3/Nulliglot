"""Score a Phase 3 pilot adapter on the Alpaca test rows (plain reference answers).

    python eval_pilot.py --adapter adapters/pilot_plain/best \
        --test data/pilot/plain/test.json --out adapters/pilot_plain/test_eval.json
    python eval_pilot.py --adapter adapters/pilot_restate/best --restate \
        --test data/pilot/plain/test.json --out adapters/pilot_restate/test_eval.json

With --restate the model was trained to repeat the question before answering.
Everything is then measured on the answer part only, so repeating the question
cannot raise a score:

  loss_true / loss_wrong   loss on the reference answer after its own question
                           (prompt and restated question) or another row's
  overlap_true / _wrong    word overlap of the generated answer with its own
                           reference or another row's
  on_topic                 share of the question's content words (4+ letters)
                           that appear in the generated answer
  on_topic_wrong           the same against another row's question
  restated                 word overlap between the restated question the model
                           wrote and the real one (--restate only)

Two generations are made with --restate: "free" (the model writes the restated
question itself) and "given" (it is filled in, the model writes the answer).
"""

import argparse
import json
import random
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer

from eval_test import LATIN, WORD, overlap_f1
from instruct_prompt import encode_prompt, special_ids
from make_pilot import A, restate_prefix
from stop_token import load_instruct_model
from train_lora import evaluate, pick_device


def content_words(text):
    return {w.lower() for w in WORD.findall(text) if len(w) >= 4}


def on_topic(answer, question):
    q = content_words(question)
    return len(q & content_words(answer)) / len(q) if q else 0.0


def example(tok, prompt_row, answer, restate, lang, bos_eos):
    """ids and labels with the loss on the answer and stop token only."""
    _, eos_id = bos_eos
    head = encode_prompt(tok, prompt_row["instruction"], prompt_row.get("input", ""), lang)
    if restate:
        head = head + tok(restate_prefix(prompt_row["instruction"]),
                          add_special_tokens=False).input_ids
    ans = tok(answer.strip(), add_special_tokens=False).input_ids
    return head + ans + [eos_id], [-100] * len(head) + ans + [eos_id]


@torch.no_grad()
def generate(model, tok, row, prefill, lang, max_len, max_new, device, eos_id):
    ids = encode_prompt(tok, row["instruction"], row.get("input", ""), lang)
    if prefill:
        ids = ids + tok(prefill, add_special_tokens=False).input_ids
    budget = min(max_new, max_len - len(ids))
    if budget <= 0:
        return {"text": "", "stopped": False, "new_tokens": 0}
    out = model.generate(
        input_ids=torch.tensor([ids], device=device),
        attention_mask=torch.ones(1, len(ids), dtype=torch.long, device=device),
        max_new_tokens=budget, do_sample=False, repetition_penalty=1.1,
        eos_token_id=eos_id, pad_token_id=eos_id)[0][len(ids):].tolist()
    return {"text": tok.decode(out, skip_special_tokens=True).strip(),
            "stopped": eos_id in out, "new_tokens": len(out)}


def split_restated(text):
    """(restated question, answer) from a free generation in the restate format."""
    if A in text:
        q, a = text.split(A, 1)
    elif A.strip() in text:
        q, a = text.split(A.strip(), 1)
    else:
        return text, ""
    return re.sub(r"^Въпрос:\s*", "", q).strip(), a.strip()


def summarize(rows, wrong, answers, gens):
    n = len(rows)
    return {
        "stopped": sum(g["stopped"] for g in gens) / n,
        "cyrillic_only": sum(not LATIN.search(a) for a in answers) / n,
        "empty": sum(not a for a in answers) / n,
        "avg_tokens": sum(g["new_tokens"] for g in gens) / n,
        "overlap_true": sum(overlap_f1(a, r["output"]) for a, r in zip(answers, rows)) / n,
        "overlap_wrong": sum(overlap_f1(a, w["output"]) for a, w in zip(answers, wrong)) / n,
        "on_topic": sum(on_topic(a, r["instruction"]) for a, r in zip(answers, rows)) / n,
        "on_topic_wrong": sum(on_topic(a, w["instruction"]) for a, w in zip(answers, wrong)) / n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="iamshnoo/bul_100m")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--test", default="data/pilot/plain/test.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--restate", action="store_true")
    ap.add_argument("--lang", default="bul", choices=["bul", "deu"])
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    bos_eos = special_ids(tok)
    eos_id = bos_eos[1]
    rows = json.loads(Path(args.test).read_text(encoding="utf-8"))[:args.limit]
    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)
    partner = {order[i]: order[(i + 1) % len(order)] for i in range(len(order))}
    wrong = [rows[partner[i]] for i in range(len(rows))]

    model = load_instruct_model(args.model, args.adapter).to(device).eval()

    def loss(prompt_rows):
        ex = [example(tok, p, r["output"], args.restate, args.lang, bos_eos)
              for p, r in zip(prompt_rows, rows)]
        return evaluate(model, [e for e in ex if len(e[0]) <= args.max_len], 16, eos_id, device)

    results = {"n": len(rows), "loss_true": loss(rows), "loss_wrong": loss(wrong)}
    results["loss_gap"] = results["loss_wrong"] - results["loss_true"]

    gen = lambda r, prefill, extra=0: generate(
        model, tok, r, prefill, args.lang, args.max_len, args.max_new + extra, device, eos_id)
    per_row = [{"instruction": r["instruction"], "input": r.get("input", ""),
                "reference": r["output"]} for r in rows]
    if args.restate:
        free = [gen(r, None, 80) for r in rows]
        parts = [split_restated(g["text"]) for g in free]
        results["free"] = summarize(rows, wrong, [a for _, a in parts], free)
        results["free"]["restated"] = sum(
            overlap_f1(q, r["instruction"]) for (q, _), r in zip(parts, rows)) / len(rows)
        results["free"]["format_ok"] = sum(A.strip() in g["text"] for g in free) / len(rows)
        given = [gen(r, restate_prefix(r["instruction"])) for r in rows]
        results["given"] = summarize(rows, wrong, [g["text"] for g in given], given)
        for p, f, (q, a), g in zip(per_row, free, parts, given):
            p.update(free=f, free_answer=a, given=g)
    else:
        plain = [gen(r, None) for r in rows]
        results["plain"] = summarize(rows, wrong, [g["text"] for g in plain], plain)
        for p, g in zip(per_row, plain):
            p.update(plain=g)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"args": vars(args), "summary": results, "rows": per_row},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(results, indent=1))
    print(f"saved {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
