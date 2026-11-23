#!/usr/bin/env python3
"""Local Gradio UI for the two monolingual agents.

Both models are loaded once at import time and kept resident, so a generation
costs ~0.2s instead of the ~10s a cold load would. Run:

    .venv/bin/python ui.py          then open http://127.0.0.1:7860

Every run also appends to agents_log.jsonl, so UI experimentation and CLI runs
accumulate into the same comparable file.
"""

import json
import random
import time
from pathlib import Path

import gradio as gr
import torch
from transformers import AutoTokenizer

from agents import Agent, Reply, loggable
from dialogue_sim import MAX_CTX, pick_device

MODEL_BUL = "iamshnoo/bul_100m"
MODEL_DEU = "iamshnoo/deu_100m"
LOG = Path("agents_log.jsonl")

DEFAULTS = dict(max_new_tokens=200, long_output=False, temperature=0.8,
                top_p=0.9, rep_penalty=1.15, no_repeat_ngram=4, greedy=False)

print("loading models (once) ...")
DEVICE = pick_device("auto")
TOK = AutoTokenizer.from_pretrained(MODEL_BUL, clean_up_tokenization_spaces=False)
AGENTS = {
    "agent_bul": Agent("agent_bul", MODEL_BUL, "Bulgarian", TOK, DEVICE,
                       nllb_code="bul_Cyrl", **DEFAULTS),
    "agent_deu": Agent("agent_deu", MODEL_DEU, "German", TOK, DEVICE,
                       nllb_code="deu_Latn", **DEFAULTS),
}
print(f"ready on {DEVICE}")


# ---------------------------------------------------------------- input clip

def clip(text, max_in, side):
    """Trim `text` to at most `max_in` tokens. Returns (text, before, after).

    Which end you keep is a real choice, not a detail. A causal LM continues
    from the END of its prompt, so keeping the tail is the default. Keeping the
    head is the one you want when the input is another agent's long reply and
    you care about how it OPENED.
    """
    ids = TOK(text, add_special_tokens=False).input_ids
    before = len(ids)
    if before <= max_in:
        return text, before, before
    ids = ids[-max_in:] if side.startswith("keep end") else ids[:max_in]
    return TOK.decode(ids), before, len(ids)


# ---------------------------------------------------------------- trace render

def _ppl(v):
    return f"{v:,.1f}" if v else "n/a"


def trace_for(r: Reply, a: Agent, listener: Agent, label: str) -> str:
    """Narrate one round trip: what the agent did, then what the model did."""
    sp = a.speaker
    head_ids = r.prompt_ids[:12]
    head_toks = [TOK.decode([i]) for i in head_ids]
    gen_head = [(i, TOK.decode([i])) for i in r.new_ids[:10]]

    over = {k: v for k, v in r.decode_used.items() if DEFAULTS.get(k) != v}
    tps = (r.new_tokens / r.seconds) if r.seconds else 0

    return f"""### {label} — {a.name}

**AGENT LAYER** (a courier; no reasoning of its own)

| step | value |
|---|---|
| received text | `{r.received[:120]}` |
| 1. prepend BOS | id `{sp.bos_id}` = `<|begin_of_text|>` |
| 2. tokenize body | {len(r.prompt_ids) - 1} tokens, `add_special_tokens=False` |
| 3. full prompt | {len(r.prompt_ids)} tokens → `{head_ids}{' …' if len(r.prompt_ids) > 12 else ''}` |
| 4. decoded back | `{' | '.join(head_toks)}` |
| 5. call model | `generate()` with the params below |
| 6. post-process | `.strip()` only — no cleanup, no interpretation |
| 7. relay target | {listener.name} (receives this text verbatim) |

**MODEL LAYER** — `{r.model_id}`

| property | value |
|---|---|
| device / dtype | {DEVICE} / {next(sp.model.parameters()).dtype} |
| params | {sp.model.num_parameters()/1e6:.1f}M total, ~113M transformer body |
| context limit | {MAX_CTX} tokens (prompt + generated, hard) |
| bos / eos used | {sp.bos_id} / {sp.eos_id} &nbsp;*(patched — config.json ships the wrong 1 / 2)* |
| prompt tokens | {r.prompt_tokens} of {MAX_CTX} budget |
| generated | {r.new_tokens} tokens in {r.passes} pass(es) |
| hit EOS? | {'yes — stopped on its own' if r.eos_emitted else 'no — ran out of budget'} |
| wall time | {r.seconds}s ({tps:.0f} tok/s) |
| collapsed? | {'YES — produced nothing' if r.collapsed else 'no'} |

**first generated tokens**

```
{chr(10).join(f'  {i:>6}  {d!r}' for i, d in gen_head) or '  (none)'}
```

**decoding params actually used** {f'(changed from default: {", ".join(over)})' if over else '(all defaults)'}

```json
{json.dumps(r.decode_used, indent=2, ensure_ascii=False)}
```

**raw output before `.strip()`**

```
{r.raw[:600]!r}
```

**perplexity** — how surprised each model is by this text

| text | under {a.name} | under {listener.name} |
|---|---|---|
| the input it received | {_ppl(r.ppl_input)} | {_ppl(listener.score(r.received))} |
| the output it produced | {_ppl(r.ppl_output)} | **{_ppl(listener.score(r.text))}** ← barrier |
"""


# ---------------------------------------------------------------- handler

def run(prompt, mode, max_new, max_in, trunc_side, long_out, temp, top_p,
        rep_pen, no_rep, greedy, seed, random_seed):
    if not prompt.strip():
        return "", "", "_enter a prompt first_", ""
    max_in = int(max_in)

    if random_seed:
        seed = random.randint(0, 2**31 - 1)
    seed = int(seed)
    torch.manual_seed(seed)

    over = dict(max_new_tokens=int(max_new), long_output=bool(long_out),
                temperature=float(temp), top_p=float(top_p),
                rep_penalty=float(rep_pen), no_repeat_ngram=int(no_rep),
                greedy=bool(greedy))

    prompt, raw_in, kept_in = clip(prompt, max_in, trunc_side)
    notes = []
    if kept_in < raw_in:
        notes.append(f"input clipped {raw_in} → {kept_in} tokens ({trunc_side})")

    budget = kept_in + int(max_new) + 1
    if budget > MAX_CTX and not long_out:
        notes.append(f"⚠️ {kept_in} in + {int(max_new)} out + 1 BOS = {budget} > "
                     f"{MAX_CTX}; output was cut to fit — lower max input, or "
                     f"tick **roll past 512**")
    warn = ("\n\n> " + "  \n> ".join(notes)) if notes else ""

    bul, deu = AGENTS["agent_bul"], AGENTS["agent_deu"]
    fh = LOG.open("a", encoding="utf-8")
    t0 = time.time()

    try:
        if mode == "ask agent_bul":
            r1 = bul.send(prompt, **over)
            out_bul, out_deu = r1.text, ""
            trace = trace_for(r1, bul, deu, "hop 1")
            fh.write(json.dumps(loggable(r1) | {"mode": "ui-ask", "seed": seed},
                                ensure_ascii=False) + "\n")

        elif mode == "ask agent_deu":
            r1 = deu.send(prompt, **over)
            out_bul, out_deu = "", r1.text
            trace = trace_for(r1, deu, bul, "hop 1")
            fh.write(json.dumps(loggable(r1) | {"mode": "ui-ask", "seed": seed},
                                ensure_ascii=False) + "\n")

        else:  # chain: bul -> deu
            r1 = bul.send(prompt, **over)
            out_bul = r1.text
            trace = trace_for(r1, bul, deu, "hop 1")
            fh.write(json.dumps(loggable(r1) | {"mode": "ui-chain", "hop": 1,
                                                "seed": seed},
                                ensure_ascii=False) + "\n")
            if r1.collapsed:
                out_deu = "(chain stopped: agent_bul returned nothing)"
            else:
                relayed, rb, rk = clip(r1.text, max_in, trunc_side)
                if rk < rb:
                    warn += (f"\n> hop 2 input clipped {rb} → {rk} tokens "
                             f"({trunc_side})")
                r2 = deu.send(relayed, **over)
                out_deu = r2.text
                trace += ("\n\n---\n\n> **handoff:** agent_bul's text above was "
                          "passed to agent_deu verbatim — not translated, "
                          "not summarised.\n\n")
                trace += trace_for(r2, deu, bul, "hop 2")
                fh.write(json.dumps(loggable(r2) | {"mode": "ui-chain",
                                                    "hop": 2, "seed": seed},
                                    ensure_ascii=False) + "\n")
    finally:
        fh.close()

    status = (f"**{mode}** · seed `{seed}` · {time.time() - t0:.2f}s · "
              f"{DEVICE} · logged to `{LOG}`{warn}")
    return out_bul, out_deu, status, trace


# ---------------------------------------------------------------- layout

with gr.Blocks(title="Monolingual agents: bul_100m ↔ deu_100m") as demo:
    gr.Markdown(
        "# Monolingual agents — `bul_100m` ↔ `deu_100m`\n"
        "Both are **base models**: no instruction tuning, no chat template, "
        "512-token context, ~113M transformer-body params. They continue text; "
        "they do not answer questions. Use a continuation seed "
        "(`патицата е`, `Луната е`) rather than an instruction "
        "(`Опишете Луната`), which lands you in textbook-exercise text."
    )

    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(label="Prompt (in the receiving model's language)",
                                value="патицата е", lines=3)
            mode = gr.Radio(["ask agent_bul", "ask agent_deu",
                             "chain: agent_bul → agent_deu"],
                            value="ask agent_bul", label="Mode")
            go = gr.Button("Run", variant="primary")
        with gr.Column(scale=1):
            max_in = gr.Slider(1, 511, value=511, step=1,
                               label="max INPUT tokens (prompt is clipped)")
            trunc_side = gr.Radio(["keep end (continuation)", "keep start"],
                                  value="keep end (continuation)",
                                  label="when clipping input")
            max_new = gr.Slider(8, 505, value=200, step=1,
                                label="max OUTPUT tokens (new)")
            long_out = gr.Checkbox(False, label="roll past 512 (memory stays 512)")
            greedy = gr.Checkbox(False, label="greedy (ignores temp / top-p)")
            with gr.Row():
                seed = gr.Number(0, label="seed", precision=0)
                random_seed = gr.Checkbox(False, label="random")

    with gr.Accordion("Decoding parameters", open=False):
        with gr.Row():
            temp = gr.Slider(0.1, 2.0, value=0.8, step=0.05, label="temperature")
            top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.01, label="top-p")
        with gr.Row():
            rep_pen = gr.Slider(1.0, 2.0, value=1.15, step=0.01,
                                label="repetition penalty")
            no_rep = gr.Slider(0, 10, value=4, step=1,
                               label="no-repeat n-gram (0 = off)")

    status = gr.Markdown()

    with gr.Row():
        out_bul = gr.Textbox(label="agent_bul → bul_100m (Bulgarian)",
                             lines=10, buttons=["copy"])
        out_deu = gr.Textbox(label="agent_deu → deu_100m (German)",
                             lines=10, buttons=["copy"])
    with gr.Accordion("What the agent and the model actually did", open=True):
        trace = gr.Markdown()

    inputs = [prompt, mode, max_new, max_in, trunc_side, long_out, temp,
              top_p, rep_pen, no_rep, greedy, seed, random_seed]
    outputs = [out_bul, out_deu, status, trace]
    go.click(run, inputs, outputs)
    prompt.submit(run, inputs, outputs)

    gr.Examples(
        [["патицата е", "ask agent_bul"],
         ["Луната е", "ask agent_bul"],
         ["Луната е естествен спътник на", "chain: agent_bul → agent_deu"],
         ["Опишете Луната", "ask agent_bul"],
         ["Der Mond ist", "ask agent_deu"]],
        inputs=[prompt, mode], label="Try these")


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", server_port=7860)
