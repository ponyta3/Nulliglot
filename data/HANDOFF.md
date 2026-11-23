# Handoff: Bulgarian instruction-tuning data for bul_100m

Context carried over from a Claude Cowork session on 2026-11-12. Read this first, then continue from "Next steps".

## Goal

Train a small LoRA adapter on a frozen monolingual base model so it follows instructions and stops, using only data in that model's language. Two adapters are planned: Bulgarian (`iamshnoo/bul_100m`) first, then German (`deu_100m`).

## Files in this folder (`~/Downloads/alpaca-cleaned-bg`)

| File | What it is |
|---|---|
| `alpaca_bg_clean_final.json` | **The training data.** 32,040 rows, zero Latin letters, Alpaca format (`instruction`, `input`, `output`). 12,455 rows have a non-empty `input`. |
| `alpaca_bg_clean.json` | Intermediate: 44,553 rows, damaged rows removed but Latin-script names still present. Not for training. |
| `alpaca_bg_removed.json` | The 7,207 rows dropped by the cleaner, each with a `reasons` list. For review only. |
| `clean_alpaca_bg.py` | The cleaning script. `python3 clean_alpaca_bg.py <original.json>` reproduces `alpaca_bg_clean.json` and `alpaca_bg_removed.json`. |

Source dataset: `pinzhenchen/alpaca-cleaned-bg` on Hugging Face (51,760 rows, machine-translated from `yahma/alpaca-cleaned`, license CC BY-NC 4.0, non-commercial).

## What was found in the source data

- The characters `=`, `^`, `<`, `>`, `*`, `{`, `}`, backtick, `#`, `\` and `|` appear nowhere in the file. The translation pipeline stripped them or replaced them with `?` or `и`, so every formula and every code snippet is damaged.
- Code keywords were translated (for example "```python" became "Шпитон", `import` became "внос на").
- Translation and grammar-correction tasks are often meaningless after translation; 249 rows have output identical to input.
- Real English sentences are rare (about 0.24% of rows).

## How the final file was made

1. `clean_alpaca_bg.py` dropped rows flagged as code (2,388), math (3,311), language tasks (2,146), English or 20%+ Latin script (1,425), junk (15). Rows can carry several flags; 7,207 unique rows dropped.
2. From the remaining 44,553 rows, every row containing any Latin letter was dropped, leaving 32,040.

The rules are regex heuristics: some good rows were dropped and a few odd rows may remain.

## Prompt template (use identically in training and inference)

With `input`:

```
<|begin_of_text|>### Инструкция:
{instruction}

### Вход:
{input}

### Отговор:
{output}<|end_of_text|>
```

Without `input`, omit the `### Вход:` block. At inference, end the prompt at `### Отговор:\n` and generate until `<|end_of_text|>`.

German markers: `### Anweisung:`, `### Eingabe:`, `### Antwort:`.

## Training plan

- Freeze the base model; LoRA on `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`.
- rank 16, alpha 32, dropout 0.05, learning rate 2e-4, batch size 16, 2-3 epochs.
- Loss on answer tokens only (labels `-100` for BOS + prompt).
- Drop examples over 512 tokens; do not truncate, so every example ends with the stop token.
- Use the tokenizer's real `<|begin_of_text|>` / `<|end_of_text|>` ids, as `load_speaker` in `dialogue_sim.py` does. The ids in the model repos' `config.json` are reported to be wrong. **Not verified in the Cowork session.**
- `peft` and `datasets` need installing in the project `.venv`.

```python
if inp.strip():
    prompt_text = f"### Инструкция:\n{instr}\n\n### Вход:\n{inp}\n\n### Отговор:\n"
else:
    prompt_text = f"### Инструкция:\n{instr}\n\n### Отговор:\n"
prompt = tok(prompt_text, add_special_tokens=False).input_ids
answer = tok(out, add_special_tokens=False).input_ids
ids    = [bos_id] + prompt + answer + [eos_id]
labels = [-100] * (1 + len(prompt)) + answer + [eos_id]
```

## Next steps

1. Shuffle `alpaca_bg_clean_final.json`; hold out 200 test rows and about 300 validation rows; never train on them.
2. Install `peft` and `datasets`.
3. Write the training script (template above, answer-only loss, 512-token filter).
4. Trial run: 5,000 rows, 1 epoch. Check that loss falls and the model stops with `<|end_of_text|>`.
5. Full run: 2-3 epochs, save to `adapters/bul_instruct`.
6. Evaluate base vs base + adapter on the 200 test rows: stops, output fully Cyrillic, answers relevant (read 30-50 by hand).
7. Add optional `--adapter-bul` / `--adapter-deu` flags to `agents.py`.
8. German: download `pinzhenchen/alpaca-cleaned-de`, scan it for the same damage, and adapt `clean_alpaca_bg.py` (its keyword rules are Bulgarian). A no-Latin filter cannot be used for German; use language ID plus the damage rules.

## Not done / not verified

- The project code (`dialogue_sim.py`, `agents.py`) was never seen in the Cowork session.
- The base model and tokenizer were not inspected; target module names and token ids are taken from earlier notes.
- The German dataset has not been examined.
