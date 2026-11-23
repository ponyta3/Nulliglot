#!/usr/bin/env python3
"""Thin agent wrappers around the monolingual base models.

An Agent here is a courier, not an intelligence. It holds exactly one model,
accepts text, runs generation, and returns the output verbatim with the
measurements attached. There is deliberately no LLM inside it choosing what to
send or interpreting what comes back -- that would put a competent model's
judgement into the middle of your transcript, and you could no longer tell
which words were bul_100m's and which were the wrapper's.

    you --"Опишете Луната"--> agent_bul --> bul_100m --> text --> you

Modes:
  ask     one agent, one round trip                        (step 1)
  chain   agent_bul answers; its text becomes agent_deu's input
  repl    interactive: talk to either agent, or chain by hand

Reminder about prompts: these are base models with no instruction tuning.
"Опишете Луната" is an imperative, and imperatives of that form live in
textbook exercises, so the likely continuation is more exercise prompts rather
than a description. A continuation seed works far better:

    --prompt "Луната е"                                 ("The Moon is...")
    --prompt "Луната е естествен спътник на"            ("...a natural satellite of")

Usage:
    python agents.py ask --prompt "Луната е"
    python agents.py chain --prompt "Луната е" --max-new-tokens 200
    python agents.py repl
"""

import argparse
import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch
from transformers import AutoTokenizer

from dialogue_sim import MAX_CTX, generate, load_speaker, pick_device, surprisal


@dataclass
class Reply:
    """Everything one round trip produced, including what it cost to measure."""
    agent: str
    model_id: str
    lang: str
    received: str
    text: str
    collapsed: bool
    ppl_input: Optional[float]    # how surprising the input was to THIS model
    ppl_output: Optional[float]   # how natural this model finds its own output
    prompt_tokens: int
    new_tokens: int
    passes: int
    seconds: float
    # --- trace detail: what the agent actually fed the model, and got back
    raw: str = ""                 # decoded output BEFORE stripping
    prompt_ids: list = field(default_factory=list)
    new_ids: list = field(default_factory=list)
    eos_emitted: bool = False
    decode_used: dict = field(default_factory=dict)


class Agent:
    """One named agent owning one model. No reasoning of its own."""

    def __init__(self, name, model_id, lang, tok, device,
                 nllb_code="", **decode):
        self.name = name
        self.lang = lang
        self.model_id = model_id
        self.tok = tok
        self.device = device
        self.speaker = load_speaker(name, lang, nllb_code, model_id, tok, device)
        self.cfg = SimpleNamespace(**decode)

    def send(self, text: str, **overrides) -> Reply:
        """Hand `text` to the model and return its continuation untouched.

        Decoding params default to those fixed at construction; `overrides`
        lets a caller (the UI) change them per request without rebuilding the
        agent, which would mean reloading the model.
        """
        cfg = SimpleNamespace(**{**vars(self.cfg), **overrides}) \
            if overrides else self.cfg

        ids = [self.speaker.bos_id] + self.tok(
            text, add_special_tokens=False).input_ids

        t0 = time.time()
        new_ids, prompt_len, passes = generate(
            self.speaker, ids, cfg, self.device,
            long_output=getattr(cfg, "long_output", False))
        dt = time.time() - t0

        raw = self.tok.decode(new_ids, skip_special_tokens=True)
        out = raw.strip()
        collapsed = not out

        _, ppl_in = surprisal(self.speaker, text, self.tok, self.device)
        _, ppl_out = surprisal(self.speaker, out, self.tok, self.device) \
            if out else (None, None)

        return Reply(
            agent=self.name, model_id=self.model_id, lang=self.lang,
            received=text, text=out if out else "(empty)", collapsed=collapsed,
            ppl_input=ppl_in, ppl_output=ppl_out,
            prompt_tokens=prompt_len, new_tokens=len(new_ids),
            passes=passes, seconds=round(dt, 2),
            raw=raw, prompt_ids=ids, new_ids=list(new_ids),
            eos_emitted=self.speaker.eos_id in new_ids,
            decode_used=dict(vars(cfg)),
        )

    def score(self, text: str) -> Optional[float]:
        """Perplexity of someone else's text under this model."""
        _, ppl = surprisal(self.speaker, text, self.tok, self.device)
        return ppl


# ---------------------------------------------------------------- presentation

def loggable(r: Reply) -> dict:
    """Reply as a dict, minus the bulky token id lists."""
    d = asdict(r)
    d.pop("prompt_ids", None)
    d.pop("new_ids", None)
    return d


def _ppl(v):
    return f"{v:,.0f}" if v else "n/a"


def show(r: Reply, listener: Optional[Agent] = None):
    print(f"\n{'─' * 72}")
    print(f"{r.agent}  ({r.model_id}, {r.lang})")
    print(f"{'─' * 72}")
    print(f"  received : {r.received}")
    print(f"\n  RESPONSE :\n")
    for line in (r.text or "(empty)").splitlines() or ["(empty)"]:
        print(f"    {line}")
    print(f"\n  input ppl (to {r.agent}) : {_ppl(r.ppl_input)}")
    print(f"  own output ppl          : {_ppl(r.ppl_output)}")
    if listener is not None:
        print(f"  output ppl to {listener.name:<10}: "
              f"{_ppl(listener.score(r.text))}   <- the language barrier")
    print(f"  {r.new_tokens} new tokens, prompt {r.prompt_tokens}tok, "
          f"{r.passes} pass(es), {r.seconds}s")
    if r.collapsed:
        print("  ! model produced nothing")


# ---------------------------------------------------------------- modes

def mode_ask(agents, args, log):
    """you -> agent_bul -> bul_100m -> text -> you"""
    a = agents[args.to]
    other = agents["agent_deu" if args.to == "agent_bul" else "agent_bul"]
    r = a.send(args.prompt)
    show(r, listener=other)
    log(loggable(r) | {"mode": "ask"})


def mode_chain(agents, args, log):
    """you -> agent_bul -> text -> agent_deu -> text -> you"""
    bul, deu = agents["agent_bul"], agents["agent_deu"]

    print(f"\nPROMPT ({bul.lang}): {args.prompt}")
    r1 = bul.send(args.prompt)
    show(r1, listener=deu)
    log(loggable(r1) | {"mode": "chain", "hop": 1})

    if r1.collapsed:
        print("\nchain stops: agent_bul returned nothing to pass on")
        return

    print(f"\n          │")
    print(f"          └─ passing agent_bul's text to agent_deu verbatim")
    r2 = deu.send(r1.text)
    show(r2, listener=bul)
    log(loggable(r2) | {"mode": "chain", "hop": 2})


def mode_repl(agents, args, log):
    print("\ncommands:  bul <text> | deu <text> | chain <text> | quit")
    print("(prompts should be in the target model's language)\n")
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            continue
        if line in ("quit", "exit"):
            break

        cmd, _, text = line.partition(" ")
        text = text.strip()
        if not text:
            print("  need some text after the command")
            continue

        if cmd == "bul":
            show(agents["agent_bul"].send(text), listener=agents["agent_deu"])
        elif cmd == "deu":
            show(agents["agent_deu"].send(text), listener=agents["agent_bul"])
        elif cmd == "chain":
            ns = SimpleNamespace(prompt=text)
            mode_chain(agents, ns, log)
        else:
            print("  unknown command; use bul / deu / chain / quit")


# ---------------------------------------------------------------- entry point

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["ask", "chain", "repl"])
    p.add_argument("--prompt", default="Луната е",
                   help="text in the receiving model's language")
    p.add_argument("--to", default="agent_bul",
                   choices=["agent_bul", "agent_deu"],
                   help="ask mode: which agent to address")
    p.add_argument("--model-bul", default="iamshnoo/bul_100m")
    p.add_argument("--model-deu", default="iamshnoo/deu_100m")
    p.add_argument("--max-new-tokens", type=int, default=200,
                   help="prompt + this must fit in 512 unless --long-output")
    p.add_argument("--long-output", action="store_true",
                   help="roll the context past 512; memory stays 512 tokens")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--rep-penalty", type=float, default=1.15)
    p.add_argument("--no-repeat-ngram", type=int, default=4)
    p.add_argument("--greedy", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--torch-seed", type=int, default=0)
    p.add_argument("--out", default="agents_log.jsonl")
    args = p.parse_args()

    torch.manual_seed(args.torch_seed)
    device = pick_device(args.device)
    print(f"device: {device}   mode: {args.mode}")

    tok = AutoTokenizer.from_pretrained(args.model_bul, clean_up_tokenization_spaces=False)
    tok_b = AutoTokenizer.from_pretrained(args.model_deu, clean_up_tokenization_spaces=False)
    if tok.get_vocab() != tok_b.get_vocab():
        raise SystemExit("models do not share a tokenizer")
    del tok_b

    decode = dict(
        max_new_tokens=args.max_new_tokens, long_output=args.long_output,
        temperature=args.temperature, top_p=args.top_p,
        rep_penalty=args.rep_penalty, no_repeat_ngram=args.no_repeat_ngram,
        greedy=args.greedy,
    )

    print("loading agents:")
    agents = {
        "agent_bul": Agent("agent_bul", args.model_bul, "Bulgarian",
                           tok, device, nllb_code="bul_Cyrl", **decode),
        "agent_deu": Agent("agent_deu", args.model_deu, "German",
                           tok, device, nllb_code="deu_Latn", **decode),
    }

    out_path = Path(args.out)
    fh = out_path.open("a", encoding="utf-8")

    def log(rec):
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()

    log({"kind": "run_meta", "mode": args.mode, "decode": decode,
         "max_ctx": MAX_CTX, "torch_seed": args.torch_seed,
         "models": {"agent_bul": args.model_bul, "agent_deu": args.model_deu}})

    try:
        {"ask": mode_ask, "chain": mode_chain,
         "repl": mode_repl}[args.mode](agents, args, log)
    finally:
        fh.close()
    print(f"\nlog -> {out_path}")


if __name__ == "__main__":
    main()
