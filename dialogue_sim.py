#!/usr/bin/env python3
"""Simulate a turn-taking dialogue between two monolingual base LMs.

Default pairing: iamshnoo/bul_100m (Bulgarian) <-> iamshnoo/deu_100m (German).

These are *base* models: no instruction tuning, no chat template, 512-token
context, ~113M non-embedding params. They cannot take turns on their own, so
turns are induced by dialogue-script prompting:

    A: <line>
    B: <line>
    A:              <- model continues here, we cut at the newline

Both models share the Llama-3 tokenizer, so token ids pass between them
losslessly; what does *not* transfer is meaning. Every turn is therefore
scored for surprisal under both models, which is the actual measurable:
how unintelligible is B's German to the Bulgarian model, and vice versa.

Two modes:

  --mode relay     A broken-telephone chain. One prompt goes to model A, its
                   completion is handed to model B as B's prompt, and so on.
                   This is the mode for "prompt bul_100m, observe, pass to
                   deu_100m, observe". Note that B does not *interpret* what
                   it receives -- it continues it. That is the finding.

  --mode dialogue  Multi-turn script-prompted conversation ("A: ...\nB: ...").
                   Turns are short and the transcript slides inside 512 tokens.

Prompts must be in the speaker's own language. These models never saw English,
so use a Bulgarian prefix the model wants to continue, not an English
instruction it has no way to recognise:

    --prompt "Луната е"              ("The moon is...")
    --prompt "Описание на Луната:"   ("Description of the Moon:")

Usage:
    python dialogue_sim.py --mode relay --prompt "Луната е" --max-new-tokens 200
    python dialogue_sim.py --mode relay --mediator nllb --hops 4
    python dialogue_sim.py --mode dialogue --turns 12 --out logs/bul_deu.jsonl
"""

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MAX_CTX = 512  # both models' max_position_embeddings; hard limit


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Speaker:
    tag: str            # "A" / "B"
    lang: str           # human-readable, for logs
    nllb_code: str      # FLORES-200 code, used only by the NLLB mediator
    model_id: str
    model: object
    bos_id: int
    eos_id: int


def load_speaker(tag, lang, nllb_code, model_id, tok, device) -> Speaker:
    print(f"  loading {model_id} ({lang}) ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float32 if device == "cpu" else torch.bfloat16,
    ).to(device)
    model.eval()

    # Both repos ship config.json with bos_token_id=1 / eos_token_id=2, which do
    # not match their Llama-3 tokenizer. Unpatched, generate() never halts.
    bos_id = tok.convert_tokens_to_ids("<|begin_of_text|>")
    eos_id = tok.convert_tokens_to_ids("<|end_of_text|>")
    model.generation_config.bos_token_id = bos_id
    model.generation_config.eos_token_id = eos_id
    model.generation_config.pad_token_id = eos_id

    return Speaker(tag, lang, nllb_code, model_id, model, bos_id, eos_id)


# ---------------------------------------------------------------- prompt build

def build_prompt(tok, header, turns, next_tag, bos_id, max_new):
    """Render the transcript as token ids, newest-first within the 512 budget.

    The header is always kept; older turns are dropped until the prompt plus the
    generation budget fits in MAX_CTX. Returns (ids, n_turns_kept).
    """
    head = tok(header, add_special_tokens=False).input_ids if header else []
    cue = tok(f"\n{next_tag}:", add_special_tokens=False).input_ids
    budget = MAX_CTX - max_new - len(head) - len(cue) - 1  # -1 for BOS

    kept, used = [], 0
    for tag, text in reversed(turns):
        piece = tok(f"\n{tag}: {text}", add_special_tokens=False).input_ids
        if used + len(piece) > budget:
            break
        kept.append(piece)
        used += len(piece)
    kept.reverse()

    ids = [bos_id] + head
    for piece in kept:
        ids += piece
    ids += cue
    return ids, len(kept)


def truncate_turn(text: str, tags) -> str:
    """Keep only this speaker's single line.

    Base models run straight past the turn boundary and start writing the other
    speaker's reply, so cut at the first newline or any speaker cue.
    """
    text = text.split("\n")[0]
    for tag in tags:
        for cue in (f"{tag}:", f" {tag}:"):
            idx = text.find(cue)
            if idx != -1:
                text = text[:idx]
    return text.strip()


# ---------------------------------------------------------------- generation

def _gen_kwargs(speaker, args, n):
    kw = dict(
        max_new_tokens=n,
        repetition_penalty=args.rep_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram,
        eos_token_id=speaker.eos_id,
        pad_token_id=speaker.eos_id,
    )
    if args.greedy:
        kw["do_sample"] = False
    else:
        kw.update(do_sample=True, temperature=args.temperature, top_p=args.top_p)
    return kw


@torch.inference_mode()
def generate(speaker, prompt_ids, args, device, long_output=False):
    """Generate a continuation, optionally rolling past the 512-token window.

    Without long_output the whole thing must fit in MAX_CTX: the prompt is
    left-truncated and the budget is whatever room is left. With long_output we
    fill the window, then carry the tail forward as the next prompt and keep
    going -- length becomes unbounded but the model's memory is still only 512
    tokens, so it forgets how its own text began.
    """
    want = args.max_new_tokens

    if not long_output:
        ids = prompt_ids[-(MAX_CTX - 1):]
        room = MAX_CTX - len(ids)
        n = max(1, min(want, room))
        inp = torch.tensor([ids], device=device)
        out = speaker.model.generate(inp, **_gen_kwargs(speaker, args, n))
        return out[0][inp.shape[-1]:].tolist(), len(ids), 1

    produced, cur, passes = [], list(prompt_ids), 0
    prompt_len = len(cur[-(MAX_CTX - 1):])
    while len(produced) < want:
        cur = cur[-(MAX_CTX - 1):]
        room = MAX_CTX - len(cur)
        if room < 16:                      # no headroom: drop the oldest half
            cur = cur[-(MAX_CTX // 2):]
            room = MAX_CTX - len(cur)
        n = min(room, want - len(produced))
        inp = torch.tensor([cur], device=device)
        out = speaker.model.generate(inp, **_gen_kwargs(speaker, args, n))
        new = out[0][inp.shape[-1]:].tolist()
        passes += 1
        if not new:
            break
        produced += new
        cur += new
        if speaker.eos_id in new:
            break
    return produced, prompt_len, passes


# ---------------------------------------------------------------- measurement

@torch.inference_mode()
def surprisal(speaker: Speaker, text: str, tok, device):
    """Mean NLL per token (nats) and perplexity of `text` under one model.

    Scored standalone (BOS + text) rather than in-context, so the number
    reflects the model's view of the utterance itself, not of the transcript.
    """
    if not text.strip():
        return None, None
    ids = tok(text, add_special_tokens=False).input_ids[: MAX_CTX - 1]
    if len(ids) < 1:
        return None, None
    inp = torch.tensor([[speaker.bos_id] + ids], device=device)
    out = speaker.model(inp, labels=inp)
    nll = out.loss.item()
    return nll, math.exp(min(nll, 20.0))


# ---------------------------------------------------------------- mediator

class Passthrough:
    """Relay each turn verbatim. The models see each other's raw language."""

    name = "none"

    def relay(self, text, src, dst):
        return text


class NllbMediator:
    """Translate each turn into the listener's language before relaying it.

    Separates two confounded questions: can the pair sustain a dialogue at all,
    versus can they do it across a language barrier. Lazy-loads a ~2.4GB model.
    """

    name = "nllb"

    def __init__(self, device):
        from transformers import AutoModelForSeq2SeqLM

        mid = "facebook/nllb-200-distilled-600M"
        print(f"  loading mediator {mid} ...", flush=True)
        self.tok = AutoTokenizer.from_pretrained(mid)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(mid).to(device)
        self.model.eval()
        self.device = device

    @torch.inference_mode()
    def relay(self, text, src, dst):
        if not text.strip():
            return text
        self.tok.src_lang = src
        batch = self.tok(text, return_tensors="pt", truncation=True,
                         max_length=256).to(self.device)
        out = self.model.generate(
            **batch,
            forced_bos_token_id=self.tok.convert_tokens_to_ids(dst),
            max_new_tokens=128,
        )
        return self.tok.batch_decode(out, skip_special_tokens=True)[0].strip()


# ---------------------------------------------------------------- relay mode

def run_relay(args, tok, A, B, mediator, device, log):
    """Broken-telephone chain: prompt -> A -> B -> A -> ... one hop at a time.

    Each model receives its predecessor's completion as its entire prompt. No
    speaker tags, no transcript -- just text handed across the language barrier.
    """
    text = args.prompt
    print(f"\n{'=' * 70}\nPROMPT ({A.lang}): {text}\n{'=' * 70}")
    log.write(json.dumps({"kind": "prompt", "text": text,
                          "lang": A.lang}, ensure_ascii=False) + "\n")

    for hop in range(1, args.hops + 1):
        speaker = A if hop % 2 == 1 else B
        listener = B if speaker is A else A

        prompt_ids = [speaker.bos_id] + tok(text, add_special_tokens=False).input_ids

        t0 = time.time()
        new_ids, prompt_len, passes = generate(
            speaker, prompt_ids, args, device, long_output=args.long_output)
        dt = time.time() - t0

        reply = tok.decode(new_ids, skip_special_tokens=True).strip()
        collapsed = not reply
        if collapsed:
            reply = "(empty)"

        # What the next model will actually receive.
        relayed = mediator.relay(reply, speaker.nllb_code, listener.nllb_code)

        in_self, in_self_ppl = surprisal(speaker, text, tok, device)
        out_self, out_self_ppl = surprisal(speaker, reply, tok, device)
        out_list, out_list_ppl = surprisal(listener, relayed, tok, device)

        rec = {
            "kind": "hop", "hop": hop,
            "speaker": speaker.tag, "model": speaker.model_id, "lang": speaker.lang,
            "received": text, "reply": reply, "relayed": relayed,
            "collapsed": collapsed,
            "ppl_input_under_speaker": in_self_ppl,
            "ppl_reply_under_speaker": out_self_ppl,
            "ppl_reply_under_listener": out_list_ppl,
            "prompt_tokens": prompt_len, "new_tokens": len(new_ids),
            "passes": passes, "seconds": round(dt, 2),
        }
        log.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log.flush()

        def fmt(v):
            return f"{v:,.0f}" if v else "n/a"

        print(f"\n--- hop {hop}: {speaker.model_id} ({speaker.lang}) ---")
        print(f"  received : {text[:200]}{'...' if len(text) > 200 else ''}")
        print(f"  REPLY    : {reply}")
        if relayed != reply:
            print(f"  relayed  : {relayed}")
        print(f"  ppl: input={fmt(in_self_ppl)} (to speaker) "
              f"reply={fmt(out_self_ppl)} (to speaker) "
              f"{fmt(out_list_ppl)} (to listener)")
        print(f"  {len(new_ids)} new tokens, prompt {prompt_len}tok, "
              f"{passes} pass(es), {dt:.1f}s")

        if collapsed:
            print("  ! model produced nothing; chain stops here")
            break
        text = relayed


# ---------------------------------------------------------------- dialogue mode

def run_dialogue(args, tok, A, B, mediator, device, log):
    """Script-prompted multi-turn conversation inside the 512-token window."""
    tags = (A.tag, B.tag)

    turns = [(A.tag, args.prompt)]
    _, ppl_a = surprisal(A, args.prompt, tok, device)
    _, ppl_b = surprisal(B, args.prompt, tok, device)
    log.write(json.dumps({
        "kind": "turn", "index": 0, "speaker": "A", "generated": False,
        "text": args.prompt, "ppl_under_A": ppl_a, "ppl_under_B": ppl_b,
    }, ensure_ascii=False) + "\n")
    print(f"[0] A ({A.lang}, seed): {args.prompt}")

    for i in range(1, args.turns + 1):
        speaker, listener = (B, A) if i % 2 == 1 else (A, B)

        ids, kept = build_prompt(tok, args.header, turns, speaker.tag,
                                 speaker.bos_id, args.max_new_tokens)
        inp = torch.tensor([ids], device=device)

        t0 = time.time()
        with torch.inference_mode():
            out = speaker.model.generate(
                inp, **_gen_kwargs(speaker, args, args.max_new_tokens))
        dt = time.time() - t0

        raw = tok.decode(out[0][inp.shape[-1]:], skip_special_tokens=True)
        text = truncate_turn(raw, tags)
        collapsed = not text
        if collapsed:
            text = "(empty)"

        relayed = mediator.relay(text, speaker.nllb_code, listener.nllb_code)
        s_nll, s_ppl = surprisal(speaker, text, tok, device)
        l_nll, l_ppl = surprisal(listener, relayed, tok, device)
        turns.append((speaker.tag, relayed))

        log.write(json.dumps({
            "kind": "turn", "index": i, "speaker": speaker.tag,
            "generated": True, "lang": speaker.lang,
            "raw": raw, "text": text, "relayed": relayed,
            "collapsed": collapsed,
            "prompt_tokens": len(ids), "turns_in_window": kept,
            "ppl_self": s_ppl, "ppl_listener": l_ppl,
            "nll_self": s_nll, "nll_listener": l_nll,
            "seconds": round(dt, 2),
        }, ensure_ascii=False) + "\n")
        log.flush()

        shown = text if relayed == text else f"{text}  ->  {relayed}"
        print(f"[{i}] {speaker.tag} ({speaker.lang}): {shown}")
        print(f"      ppl self={s_ppl:.0f} listener={l_ppl:.0f} "
              f"| ctx={len(ids)}tok, {kept} turns, {dt:.1f}s"
              if s_ppl and l_ppl else f"      ctx={len(ids)}tok, {dt:.1f}s")


# ---------------------------------------------------------------- entry point

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["relay", "dialogue"], default="relay",
                   help="relay: prompt -> A -> B -> ... ; dialogue: scripted turns")
    p.add_argument("--model-a", default="iamshnoo/bul_100m")
    p.add_argument("--model-b", default="iamshnoo/deu_100m")
    p.add_argument("--lang-a", default="Bulgarian")
    p.add_argument("--lang-b", default="German")
    p.add_argument("--nllb-a", default="bul_Cyrl")
    p.add_argument("--nllb-b", default="deu_Latn")
    p.add_argument("--prompt", default="Луната е",
                   help="opening text, in model A's language (NOT an English "
                        "instruction -- these are base models)")
    p.add_argument("--hops", type=int, default=2,
                   help="relay mode: how many models to pass through")
    p.add_argument("--turns", type=int, default=12,
                   help="dialogue mode: turns to generate")
    p.add_argument("--header", default="",
                   help="dialogue mode: preamble always kept in context")
    p.add_argument("--max-new-tokens", type=int, default=200,
                   help="tokens to generate per hop/turn. Prompt + this must "
                        "fit in 512 unless --long-output is set")
    p.add_argument("--long-output", action="store_true",
                   help="roll the context to exceed 512 total tokens; the model "
                        "still only remembers the last 512")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--rep-penalty", type=float, default=1.15)
    p.add_argument("--no-repeat-ngram", type=int, default=4)
    p.add_argument("--greedy", action="store_true", help="disable sampling")
    p.add_argument("--mediator", choices=["none", "nllb"], default="none",
                   help="none: hand text over untranslated (the control). "
                        "nllb: translate into the listener's language")
    p.add_argument("--device", default="auto")
    p.add_argument("--torch-seed", type=int, default=0)
    p.add_argument("--out", default="dialogue_log.jsonl")
    args = p.parse_args()

    torch.manual_seed(args.torch_seed)
    device = pick_device(args.device)
    print(f"device: {device}  mode: {args.mode}")

    # Shared tokenizer: verify rather than assume.
    tok = AutoTokenizer.from_pretrained(args.model_a, clean_up_tokenization_spaces=False)
    tok_b = AutoTokenizer.from_pretrained(args.model_b, clean_up_tokenization_spaces=False)
    if tok.get_vocab() != tok_b.get_vocab():
        raise SystemExit("models do not share a tokenizer; relay text, not ids")
    del tok_b

    print("loading models:")
    A = load_speaker("A", args.lang_a, args.nllb_a, args.model_a, tok, device)
    B = load_speaker("B", args.lang_b, args.nllb_b, args.model_b, tok, device)

    mediator = NllbMediator(device) if args.mediator == "nllb" else Passthrough()
    print(f"mediator: {mediator.name}")

    out_path = Path(args.out)
    if out_path.parent != Path("."):
        out_path.parent.mkdir(parents=True, exist_ok=True)
    log = out_path.open("w", encoding="utf-8")

    log.write(json.dumps({
        "kind": "run_meta", "mode": args.mode,
        "speakers": {"A": A.model_id, "B": B.model_id},
        "langs": {"A": A.lang, "B": B.lang},
        "mediator": mediator.name, "max_ctx": MAX_CTX,
        "long_output": args.long_output,
        "decoding": {
            "greedy": args.greedy, "temperature": args.temperature,
            "top_p": args.top_p, "repetition_penalty": args.rep_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram,
            "max_new_tokens": args.max_new_tokens,
        },
        "torch_seed": args.torch_seed,
    }, ensure_ascii=False) + "\n")

    try:
        if args.mode == "relay":
            run_relay(args, tok, A, B, mediator, device, log)
        else:
            run_dialogue(args, tok, A, B, mediator, device, log)
    finally:
        log.close()
    print(f"\ntranscript -> {out_path}")


if __name__ == "__main__":
    main()
