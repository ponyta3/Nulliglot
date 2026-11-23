"""Interactive text-continuation REPL for iamshnoo/bul_100m.

NOTE: bul_100m is a *base* model (pretrained only, no instruction tuning,
no chat template, 512-token context). It continues text; it does not answer
questions or follow instructions. Feed it a prefix, not a prompt.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "iamshnoo/bul_100m"
MAX_CTX = 512  # model's max_position_embeddings

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"loading {MODEL_ID} on {device} ...")
tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float32 if device == "cpu" else torch.bfloat16,
).to(device)
model.eval()

# The repo's config.json ships bos_token_id=1 / eos_token_id=2, which do not
# match its Llama-3 tokenizer (<|begin_of_text|>=128000, <|end_of_text|>=128001).
# Left as-is, generation never emits EOS and always runs to max_new_tokens.
bos_id = tok.convert_tokens_to_ids("<|begin_of_text|>")
eos_id = tok.convert_tokens_to_ids("<|end_of_text|>")
model.generation_config.bos_token_id = bos_id
model.generation_config.eos_token_id = eos_id
model.generation_config.pad_token_id = eos_id
print(f"fixed special ids -> bos={bos_id} eos={eos_id}")

print(f"\nparams: {model.num_parameters()/1e6:.1f}M total, "
      f"{model.num_parameters(exclude_embeddings=True)/1e6:.1f}M non-embedding")
print("Enter a text prefix in Bulgarian to continue. Ctrl-D or 'quit' to exit.\n")

while True:
    try:
        prefix = input("prefix> ").strip()
    except EOFError:
        break
    if prefix in ("quit", "exit"):
        break
    if not prefix:
        continue

    ids = tok(prefix, return_tensors="pt", add_special_tokens=False).input_ids
    ids = torch.cat([torch.tensor([[bos_id]]), ids], dim=-1)[:, -MAX_CTX:].to(device)

    budget = min(200, MAX_CTX - ids.shape[-1])
    if budget <= 0:
        print("  (prefix fills the whole 512-token context; shorten it)\n")
        continue

    with torch.inference_mode():
        out = model.generate(
            ids,
            max_new_tokens=budget,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.1,
        )

    print("\n" + tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True).strip() + "\n")
