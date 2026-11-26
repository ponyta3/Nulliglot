"""Train a Tamil-only tokenizer on the training split of the final corpus.

    python train_tokenizer.py <final_dir> <train_list> <out_dir> --vocab 16000 32000 ...
    python train_tokenizer.py <final_dir> <train_list> <out_dir> --census-only

One tokenizer is written per vocabulary size, to <out_dir>/bpe_<size>/, in the Hugging Face
format. <out_dir>/chars.tsv lists every character of the training text with its count.

Design
- Trained only on this corpus. Nothing is taken from another tokenizer.
- BPE over Unicode characters. Every token is made of whole characters, so the model cannot
  write half a Tamil letter. (A byte-level BPE, as in GPT-2 and Llama 3, was tested first: about
  150 of its tokens were pieces of a character, some glued to a whole letter.)
- These characters get a token: every character of the Unicode Tamil block (also the rare ones,
  such as the old number and date signs), every printable ASCII character, the line break, and
  any other character seen at least --min-char-count times in the training text (curly quotes,
  dashes, the ellipsis). Any other character (a rare symbol, another script, an emoji) is
  written as its UTF-8 bytes with 256 byte tokens, so every text can be encoded and decoded back
  exactly and there is no unknown token. This is the SentencePiece "byte fallback" used by
  Llama 2 and Gemma. The rule is by kind of character and not by count alone because the
  training text holds a little damaged text (stray Latin-1 letters, private-use codes: under 10
  characters per million) that should not get tokens while rare Tamil signs should.
- Text is put in Unicode NFC form first, so the two-part Tamil vowel signs (ொ ோ ௌ) always have
  one spelling. Zero-width characters and soft hyphens are removed, as in the corpus cleaning.
- Words are cut before BPE with a pattern in which a word is a run of letters AND combining
  marks. The usual GPT-2 / Llama-3 pattern counts letters only; Tamil vowel signs and the pulli
  are marks, so that pattern would cut every Tamil word at each vowel sign. A word keeps one
  leading space, so "the start of a word" and "inside a word" are different tokens and no
  separate space token is spent between words. Digits go in groups of up to 3.
- Token ids: 0 pad, 1 begin of text, 2 end of text, 3-15 reserved for later use (instruction
  tuning) so the vocabulary never has to be resized, 16-271 the byte tokens, then characters and
  merged tokens in the order they were learned.
"""
import argparse, json, os, re, sys, time, unicodedata
from collections import Counter
from tokenizers import Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers

PAD, BOS, EOS = "<|pad|>", "<|begin_of_text|>", "<|end_of_text|>"
SPECIALS = [PAD, BOS, EOS] + [f"<|reserved_{i}|>" for i in range(13)]
BYTES = [f"<0x{b:02X}>" for b in range(256)]
SPLIT = (r" ?[\p{L}\p{M}]+"                    # a word: letters and marks, with one leading space
         r"|\p{N}{1,3}"                        # digits, at most 3 together
         r"| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*"    # punctuation and symbols
         r"|\s*[\r\n]+"                        # line breaks
         r"|\s+(?!\S)|\s+")                    # other white space
# invisible characters the corpus cleaners remove (zero-width space and joiners, soft hyphen,
# byte-order mark); removed here too, so text typed by a user is seen like the training text
INVISIBLE = "[​‌‍­﻿]"
INV = re.compile(INVISIBLE)
CHECKS = ["தமிழ்நாடு அரசு 2026-ஆம் ஆண்டு அறிவித்தது.", "கொ கோ கௌ ஸ்ரீ க்ஷ ௧௨௩ ஃ", "a  b\tc\r\nd\n\n\ne ",
          "Hello (File) 12345 — “ஆம்” ‘இல்லை’ …", "😀 中文 हिन्दी ελληνικά \u0001 <0x41> <|pad|x ▁", " முன் இடைவெளி",
          "கொ"[:1] + "ொ" + " பிரிந்த ஒகரம்", "சொல்‌லின் நடுவில்‍ மறை­எழுத்து", ""]


CORE = ["\n"] + [chr(o) for o in range(0x20, 0x7F)] + \
       [chr(o) for o in range(0x0B80, 0x0C00) if unicodedata.name(chr(o), "")]


def norm(t):
    return INV.sub("", unicodedata.normalize("NFC", t))


def new_tokenizer():
    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.Sequence([normalizers.NFC(), normalizers.Replace(Regex(INVISIBLE), "")])
    tok.pre_tokenizer = pre_tokenizers.Split(Regex(SPLIT), behavior="isolated")
    return tok


def chunks(final_dir, rels, size=200_000):
    """Training text in pieces of about `size` characters, cut only between paragraphs, with the
    blank line between paragraphs kept so that it gets a token of its own."""
    for rel in rels:
        t = open(os.path.join(final_dir, rel), encoding="utf-8").read().strip()
        if not t: continue
        buf, n = [], 0
        for p in t.split("\n\n"):
            buf.append(p); n += len(p)
            if n >= size:
                yield "\n\n".join(buf) + "\n\n"; buf, n = [], 0
        if buf: yield "\n\n".join(buf) + "\n\n"


def census(final_dir, rels, out_dir):
    c = Counter()
    for piece in chunks(final_dir, rels, 5_000_000): c.update(norm(piece))
    with open(os.path.join(out_dir, "chars.tsv"), "w", encoding="utf-8") as o:
        o.write("char\tcodepoint\tname\tcount\n")
        for ch, n in c.most_common():
            o.write(f"{ch if ch.isprintable() else repr(ch)[1:-1]}\tU+{ord(ch):04X}\t{unicodedata.name(ch, '?')}\t{n}\n")
    return c


def finish(tok):
    """Add the 256 byte tokens after the special tokens, switch on the byte fallback, and set the
    decoder and the begin-of-text rule."""
    j = json.loads(tok.to_str())
    v = j["model"]["vocab"]; order = sorted(v, key=v.get)
    assert order[:len(SPECIALS)] == SPECIALS and not set(BYTES) & set(order)
    j["model"]["vocab"] = {t: i for i, t in enumerate(SPECIALS + BYTES + order[len(SPECIALS):])}
    j["model"]["byte_fallback"] = True
    tok = Tokenizer.from_str(json.dumps(j, ensure_ascii=False))
    tok.decoder = decoders.Sequence([decoders.ByteFallback(), decoders.Fuse()])
    tok.post_processor = processors.TemplateProcessing(
        single=f"{BOS} $A", pair=f"{BOS} $A {BOS} $B", special_tokens=[(BOS, tok.token_to_id(BOS))])
    return tok


def check(tok, V, alphabet):
    assert tok.get_vocab_size() == V, tok.get_vocab_size()
    for i, s in enumerate(SPECIALS + BYTES): assert tok.token_to_id(s) == i, s
    for ch in alphabet: assert tok.token_to_id(ch) is not None, repr(ch)
    for s in CHECKS:
        e = tok.encode(s)
        assert e.ids[0] == 1 and tok.decode(e.ids[1:], skip_special_tokens=False) == norm(s), (s, e.tokens)
    e = tok.encode("கொ"[:1] + "ொ"); assert e.ids == tok.encode("கொ").ids       # NFC
    assert tok.encode("<|end_of_text|>").ids == [1, 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("final_dir"); ap.add_argument("train_list"); ap.add_argument("out_dir")
    ap.add_argument("--vocab", type=int, nargs="*", default=[])
    ap.add_argument("--min-char-count", type=int, default=1000)
    ap.add_argument("--census-only", action="store_true")
    a = ap.parse_args()
    rels = [x for x in open(a.train_list, encoding="utf-8").read().split("\n") if x]
    os.makedirs(a.out_dir, exist_ok=True)
    for V in a.vocab:
        if os.path.exists(os.path.join(a.out_dir, f"bpe_{V}", "tokenizer.json")):
            sys.exit(f"{a.out_dir}/bpe_{V} already has a tokenizer; not overwriting")
    t0 = time.time()
    c = census(a.final_dir, rels, a.out_dir)
    extra = sorted(ch for ch, n in c.items() if n >= a.min_char_count and ch not in CORE)
    alphabet = sorted(CORE + extra)
    tot = sum(c.values()); cov = sum(c[ch] for ch in alphabet)
    print(f"training documents: {len(rels)}; characters: {tot}; distinct: {len(c)}; census took {time.time() - t0:.0f} s\n"
          f"characters with a token: {len(alphabet)} = {len(CORE)} Tamil block, ASCII and line break "
          f"({sum(1 for ch in CORE if not c[ch])} of them never seen) + {len(extra)} others seen at least "
          f"{a.min_char_count} times: {' '.join(extra)}\n"
          f"they cover {100 * cov / tot:.5f}% of the text; the rest goes through the byte tokens", flush=True)
    if a.census_only: return
    from transformers import PreTrainedTokenizerFast
    for V in a.vocab:
        out = os.path.join(a.out_dir, f"bpe_{V}"); t0 = time.time()
        tok = new_tokenizer()
        trainer = trainers.BpeTrainer(vocab_size=V - len(BYTES), special_tokens=SPECIALS, min_frequency=2,
                                      initial_alphabet=alphabet, limit_alphabet=len(alphabet), show_progress=False)
        tok.train_from_iterator(chunks(a.final_dir, rels), trainer=trainer)
        tok = finish(tok); check(tok, V, alphabet)
        os.makedirs(out, exist_ok=True)
        PreTrainedTokenizerFast(tokenizer_object=tok, bos_token=BOS, eos_token=EOS, pad_token=PAD,
                                clean_up_tokenization_spaces=False,
                                model_input_names=["input_ids", "attention_mask"]).save_pretrained(out)
        print(f"vocab {V}: {time.time() - t0:.0f} s -> {out}", flush=True)


if __name__ == "__main__":
    main()
