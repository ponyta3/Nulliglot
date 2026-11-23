"""Answer an instruction with a tuned bul_100m / deu_100m adapter.

    python generate.py --adapter adapters/phase3_lora/best \
        --instruction "Назовете три вида плодове."
    python generate.py --adapter adapters/phase3_lora/best \
        --instruction "Кое е първото изречение в текста?" --input "Вали дъжд. Студено е."

Decoding options (no training involved):

  --no-repeat N      never repeat a sequence of N tokens within the answer (stops loops)
  --rep-penalty X    penalty on tokens already written
  --boost B          add B to the score of tokens that spell the content words
                     of the instruction and input (keeps the answer on topic)
  --own-script       forbid tokens with Latin letters (Bulgarian: Cyrillic only)
"""

import argparse
import re

import torch
from transformers import AutoTokenizer, LogitsProcessor, LogitsProcessorList

from instruct_prompt import encode_prompt, special_ids
from stop_token import load_instruct_model
from train_lora import pick_device

WORD = re.compile(r"\w+", re.UNICODE)
LATIN = re.compile(r"[A-Za-z]")


class AddBias(LogitsProcessor):
    """Add a fixed value to the scores of chosen tokens."""

    def __init__(self, token_ids, value):
        self.ids = torch.tensor(sorted(token_ids), dtype=torch.long)
        self.value = value

    def __call__(self, input_ids, scores):
        if len(self.ids):
            scores[:, self.ids.to(scores.device)] += self.value
        return scores


class NoRepeatInAnswer(LogitsProcessor):
    """Forbid repeating a sequence of n tokens inside the answer.

    The built-in no_repeat_ngram_size also counts the prompt, which would stop
    the model from reusing the question's own words.
    """

    def __init__(self, n, prompt_len):
        self.n, self.prompt_len = n, prompt_len

    def __call__(self, input_ids, scores):
        for b in range(input_ids.shape[0]):
            gen = input_ids[b, self.prompt_len:].tolist()
            if len(gen) < self.n:
                continue
            last = tuple(gen[len(gen) - self.n + 1:]) if self.n > 1 else ()
            banned = [gen[i + self.n - 1] for i in range(len(gen) - self.n + 1)
                      if tuple(gen[i:i + self.n - 1]) == last]
            if banned:
                scores[b, banned] = float("-inf")
        return scores


def latin_token_ids(tok):
    """Ordinary vocabulary tokens that contain a Latin letter."""
    first_special = min(tok.all_special_ids)
    return [i for i, t in enumerate(tok.convert_ids_to_tokens(list(range(first_special))))
            if LATIN.search(t)]


def topic_token_ids(tok, text, min_letters=4):
    """Tokens that spell the content words of the instruction and input."""
    ids = set()
    for w in {w for w in WORD.findall(text) if len(w) >= min_letters and not w.isdigit()}:
        for form in (w, " " + w, " " + w.lower()):
            ids.update(tok(form, add_special_tokens=False).input_ids)
    return ids


@torch.no_grad()
def answer(model, tok, instruction, inp="", lang="bul", max_len=512, max_new=200,
           rep_penalty=1.1, no_repeat=0, boost=0.0, banned_ids=None, device="cpu"):
    """Greedy answer. Returns {"text", "stopped", "new_tokens"}."""
    _, eos_id = special_ids(tok)
    ids = encode_prompt(tok, instruction, inp, lang)
    budget = min(max_new, max_len - len(ids))
    if budget <= 0:
        return {"text": "", "stopped": False, "new_tokens": 0}
    procs = LogitsProcessorList()
    if no_repeat:
        procs.append(NoRepeatInAnswer(no_repeat, len(ids)))
    if boost:
        procs.append(AddBias(topic_token_ids(tok, instruction + " " + (inp or "")), boost))
    if banned_ids:
        procs.append(AddBias(banned_ids, float("-inf")))
    out = model.generate(
        input_ids=torch.tensor([ids], device=device),
        attention_mask=torch.ones(1, len(ids), dtype=torch.long, device=device),
        max_new_tokens=budget, do_sample=False, repetition_penalty=rep_penalty,
        logits_processor=procs,
        eos_token_id=eos_id, pad_token_id=eos_id)[0][len(ids):].tolist()
    return {"text": tok.decode(out, skip_special_tokens=True).strip(),
            "stopped": eos_id in out, "new_tokens": len(out)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="iamshnoo/bul_100m")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--instruction", required=True)
    ap.add_argument("--input", default="")
    ap.add_argument("--lang", default="bul", choices=["bul", "deu"])
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--rep-penalty", type=float, default=1.1)
    ap.add_argument("--no-repeat", type=int, default=0)
    ap.add_argument("--boost", type=float, default=0.0)
    ap.add_argument("--own-script", action="store_true")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = pick_device(args.device)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = load_instruct_model(args.model, args.adapter).to(device).eval()
    banned = latin_token_ids(tok) if args.own_script else None
    out = answer(model, tok, args.instruction, args.input, args.lang, max_new=args.max_new,
                 rep_penalty=args.rep_penalty, no_repeat=args.no_repeat, boost=args.boost,
                 banned_ids=banned, device=device)
    print(out["text"])
    if not out["stopped"]:
        print("[stopped at the length limit]")


if __name__ == "__main__":
    main()
