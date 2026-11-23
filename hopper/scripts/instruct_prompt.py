"""Prompt template for the instruction-tuned adapters.

Single source of truth: the split, training, evaluation and inference code must
all build prompts through this module so the adapter sees the same token
sequence everywhere.

    <|begin_of_text|>### Инструкция:
    {instruction}

    ### Вход:            (block omitted when input is empty)
    {input}

    ### Отговор:
    {output}<|end_of_text|>
"""

MARKERS = {
    "bul": ("### Инструкция:", "### Вход:", "### Отговор:"),
    "deu": ("### Anweisung:", "### Eingabe:", "### Antwort:"),
}


def special_ids(tok):
    """Real BOS/EOS ids from the tokenizer.

    The model repos' config.json says bos=1 / eos=2, which is wrong for their
    Llama-3 tokenizer (128000 / 128001). Always take them from here.
    """
    bos_id = tok.convert_tokens_to_ids("<|begin_of_text|>")
    eos_id = tok.convert_tokens_to_ids("<|end_of_text|>")
    if bos_id is None or eos_id is None or bos_id == eos_id:
        raise ValueError("tokenizer lacks <|begin_of_text|> / <|end_of_text|>")
    return bos_id, eos_id


def prompt_text(instruction, inp="", lang="bul"):
    """Prompt up to and including the answer marker's newline (no BOS)."""
    m_instr, m_inp, m_ans = MARKERS[lang]
    instruction, inp = instruction.strip(), (inp or "").strip()
    if inp:
        return f"{m_instr}\n{instruction}\n\n{m_inp}\n{inp}\n\n{m_ans}\n"
    return f"{m_instr}\n{instruction}\n\n{m_ans}\n"


def marker_token_ids(tok, lang="bul"):
    """Ids of every token the template itself contributes to a prompt."""
    filler = "\x00"
    ids = set()
    for text in (prompt_text(filler, "", lang), prompt_text(filler, filler, lang)):
        for piece in text.split(filler):
            ids.update(tok(piece, add_special_tokens=False).input_ids)
    return sorted(ids)


def encode_prompt(tok, instruction, inp="", lang="bul"):
    """Token ids for inference: BOS + prompt. Generate until EOS."""
    bos_id, _ = special_ids(tok)
    text = prompt_text(instruction, inp, lang)
    return [bos_id] + tok(text, add_special_tokens=False).input_ids


def encode_example(tok, row, lang="bul"):
    """Token ids and labels for training. Loss is on answer + EOS only.

    Prompt and answer are tokenized separately so the prompt ids are exactly
    what encode_prompt() produces at inference.
    """
    bos_id, eos_id = special_ids(tok)
    prompt = tok(
        prompt_text(row["instruction"], row.get("input", ""), lang),
        add_special_tokens=False,
    ).input_ids
    answer = tok(row["output"].strip(), add_special_tokens=False).input_ids
    ids = [bos_id] + prompt + answer + [eos_id]
    labels = [-100] * (1 + len(prompt)) + answer + [eos_id]
    return ids, labels
