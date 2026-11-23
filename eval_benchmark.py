"""Multiple-choice accuracy on the Bulgarian MMLU and ARC test sets.

    python eval_benchmark.py --model models/bul_100m \
        --system lora=adapters/bul_instruct/best \
        --out adapters/benchmark_results.json

The model is never asked to generate. For every question each choice is scored
by the log-likelihood the model gives to the choice text after the prompt, and
the highest-scoring choice is the model's answer.

The base model is always scored; every --system name=dir adds a tuned model
(a LoRA adapter folder or a full fine-tune folder). Each is scored three ways:

  template     question inside the instruction template used for training
  plain        "Въпрос: {question}\nОтговор: {choice}", no template
  no_question  the choice text alone, the question is not shown

A model that reads the question scores well above no_question. Accuracy is
reported for the summed log-likelihood ("sum") and for the per-token mean
("mean"), next to the chance level.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from instruct_prompt import prompt_text, special_ids
from stop_token import load_instruct_model
from train_lora import pick_device

BENCHMARKS = ("mmlu", "arc_easy", "arc_challenge")
STYLES = ("template", "plain", "no_question")


def build_sequences(tok, rows, style, lang, max_len, bos_id):
    """One (ids, n_choice_tokens) per choice; all choices of a question share a prompt."""
    seqs, owner = [], []
    for qi, r in enumerate(rows):
        if style == "template":
            prompt, lead = prompt_text(r["question"], "", lang), ""
        elif style == "plain":
            prompt, lead = f"Въпрос: {r['question']}\nОтговор:", " "
        else:
            prompt, lead = "", ""
        p_ids = tok(prompt, add_special_tokens=False).input_ids if prompt else []
        c_ids = [tok(lead + c, add_special_tokens=False).input_ids[:max_len - 2]
                 for c in r["choices"]]
        room = max_len - 1 - max(len(c) for c in c_ids)
        p_ids = p_ids[-room:] if room > 0 else []  # keep the end of a long question
        for c in c_ids:
            seqs.append(([bos_id] + p_ids + c, len(c)))
            owner.append(qi)
    return seqs, owner


@torch.no_grad()
def score_sequences(model, seqs, pad_id, device, max_batch_tokens):
    """Sum of log-probabilities of each sequence's last n tokens."""
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]))
    out = [0.0] * len(seqs)
    i = 0
    while i < len(order):
        j, longest = i, 0
        while j < len(order):
            longest_next = max(longest, len(seqs[order[j]][0]))
            if j > i and longest_next * (j - i + 1) > max_batch_tokens:
                break
            longest, j = longest_next, j + 1
        idx = order[i:j]
        ids = torch.full((len(idx), longest), pad_id, dtype=torch.long)
        mask = torch.zeros((len(idx), longest), dtype=torch.long)
        for b, k in enumerate(idx):
            s = seqs[k][0]
            ids[b, :len(s)] = torch.tensor(s)
            mask[b, :len(s)] = 1
        logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
        for b, k in enumerate(idx):
            s, n = seqs[k]
            pos = torch.arange(len(s) - n, len(s), device=device)
            lp = torch.log_softmax(logits[b, pos - 1].float(), dim=-1)
            out[k] = lp.gather(1, ids[b, pos.cpu()].to(device).unsqueeze(1)).sum().item()
        del logits
        if device == "mps":
            torch.mps.empty_cache()
        i = j
    return out


def accuracy(rows, seqs, owner, scores):
    by_q = defaultdict(list)
    for (ids, n), qi, s in zip(seqs, owner, scores):
        by_q[qi].append((s, s / max(n, 1)))
    hit_sum = hit_mean = 0
    per_subject = defaultdict(lambda: [0, 0])
    for qi, r in enumerate(rows):
        sums = [x[0] for x in by_q[qi]]
        means = [x[1] for x in by_q[qi]]
        ok_sum = sums.index(max(sums)) == r["answer"]
        hit_sum += ok_sum
        hit_mean += means.index(max(means)) == r["answer"]
        per_subject[r["subject"]][0] += ok_sum
        per_subject[r["subject"]][1] += 1
    n = len(rows)
    return {"sum": hit_sum / n, "mean": hit_mean / n,
            "by_subject_sum": {k: v[0] / v[1] for k, v in sorted(per_subject.items())}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="iamshnoo/bul_100m")
    ap.add_argument("--system", action="append", default=[], metavar="NAME=DIR",
                    help="tuned model to score besides the base (repeatable)")
    ap.add_argument("--data-dir", default="data/benchmarks")
    ap.add_argument("--benchmarks", default=",".join(BENCHMARKS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang", default="bul", choices=["bul", "deu"])
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--max-batch-tokens", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=None, help="score only the first N questions")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    bos_id, eos_id = special_ids(tok)
    systems = [("base", None)] + [tuple(s.split("=", 1)) for s in args.system]

    data, chance = {}, {}
    for name in args.benchmarks.split(","):
        rows = json.loads((Path(args.data_dir) / f"{name}.json").read_text(encoding="utf-8"))
        data[name] = rows[:args.limit]
        chance[name] = sum(1 / len(r["choices"]) for r in data[name]) / len(data[name])
        print(f"{name}: {len(data[name])} questions, chance {chance[name]:.4f}", flush=True)

    results = {}
    for sys_name, path in systems:
        if path is None:
            model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
        else:
            model = load_instruct_model(args.model, path)
        model.to(device).eval()
        results[sys_name] = {}
        for name, rows in data.items():
            results[sys_name][name] = {}
            for style in STYLES:
                seqs, owner = build_sequences(tok, rows, style, args.lang, args.max_len, bos_id)
                scores = score_sequences(model, seqs, eos_id, device, args.max_batch_tokens)
                acc = accuracy(rows, seqs, owner, scores)
                results[sys_name][name][style] = acc
                print(f"{sys_name:10s} {name:14s} {style:12s} "
                      f"sum {acc['sum']:.4f}  mean {acc['mean']:.4f}", flush=True)
        del model

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"args": vars(args), "n": {k: len(v) for k, v in data.items()},
         "chance": chance, "results": results}, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\naccuracy by summed log-likelihood")
    print("%-14s %-12s %8s" % ("benchmark", "style", "chance")
          + "".join(" %10s" % s for s, _ in systems))
    for name in data:
        for style in STYLES:
            print("%-14s %-12s %8.4f" % (name, style, chance[name])
                  + "".join(" %10.4f" % results[s][name][style]["sum"] for s, _ in systems))
    print(f"\nsaved to {args.out}")


if __name__ == "__main__":
    main()
