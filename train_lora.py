"""Train a LoRA instruction adapter on a frozen monolingual base model.

Trial run (5,000 rows, 1 epoch):
    python train_lora.py --max-train-rows 5000 --epochs 1 --out adapters/bul_instruct_trial

Full run:
    python train_lora.py --epochs 3 --out adapters/bul_instruct

The base model stays frozen; only the LoRA matrices train. Loss is computed on
the answer tokens and the closing <|end_of_text|> only. Examples longer than
the context are dropped, never truncated, so every example ends with the stop
token.
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from instruct_prompt import encode_example, encode_prompt, marker_token_ids, special_ids
from stop_token import MarkerEmbeddings, StopTokenRow

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def pick_device(name):
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_examples(path, tok, lang, max_len, limit=None, seed=0):
    """Tokenize a split file. Returns (examples, n_dropped_for_length)."""
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    examples, dropped = [], 0
    for row in rows:
        ids, labels = encode_example(tok, row, lang)
        if len(ids) > max_len:
            dropped += 1
        else:
            examples.append((ids, labels))
    if limit is not None and limit < len(examples):
        examples = random.Random(seed).sample(examples, limit)
    return examples, dropped


def make_batches(examples, batch_size, rng=None):
    """Batches of similar length to limit padding.

    With rng: shuffle, sort by length inside chunks of 50 batches, then shuffle
    the batch order. Without rng (evaluation): plain sort by length.
    """
    order = list(range(len(examples)))
    if rng is None:
        order.sort(key=lambda i: len(examples[i][0]))
    else:
        rng.shuffle(order)
        chunk = batch_size * 50
        order = [
            i
            for start in range(0, len(order), chunk)
            for i in sorted(order[start:start + chunk], key=lambda i: len(examples[i][0]))
        ]
    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    if rng is not None:
        rng.shuffle(batches)
    return batches


PAD_MULTIPLE = 32


def free_device_cache(device):
    """Return cached GPU buffers to the system.

    The MPS allocator keeps a buffer pool per tensor shape and never shrinks it
    on its own; with variable-length batches it grew past 120 GB within ~100
    steps and the run stalled in swap. Called every few steps on MPS.
    """
    if device == "mps":
        torch.mps.empty_cache()


def collate(examples, idxs, pad_id, device):
    # Round the width up so batches share a small set of shapes.
    width = max(len(examples[i][0]) for i in idxs)
    width = -(-width // PAD_MULTIPLE) * PAD_MULTIPLE
    ids = torch.full((len(idxs), width), pad_id, dtype=torch.long)
    labels = torch.full((len(idxs), width), -100, dtype=torch.long)
    mask = torch.zeros((len(idxs), width), dtype=torch.long)
    for r, i in enumerate(idxs):
        x, y = examples[i]
        ids[r, :len(x)] = torch.tensor(x)
        labels[r, :len(y)] = torch.tensor(y)
        mask[r, :len(x)] = 1
    return ids.to(device), mask.to(device), labels.to(device)


@torch.no_grad()
def evaluate(model, examples, batch_size, pad_id, device):
    """Mean loss per answer token over the whole set."""
    model.eval()
    total, n_tokens = 0.0, 0
    for idxs in make_batches(examples, batch_size):
        ids, mask, labels = collate(examples, idxs, pad_id, device)
        n = int((labels != -100).sum())
        loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
        total += loss.item() * n
        n_tokens += n
        del ids, mask, labels, loss
        free_device_cache(device)
    model.train()
    return total / max(n_tokens, 1)


@torch.no_grad()
def sample_generations(model, tok, rows, lang, max_len, device, max_new=200):
    """Greedy-decode a few held-out prompts; report whether each one stops."""
    _, eos_id = special_ids(tok)
    model.eval()
    results = []
    for row in rows:
        ids = encode_prompt(tok, row["instruction"], row.get("input", ""), lang)
        budget = min(max_new, max_len - len(ids))
        if budget <= 0:
            continue
        out = model.generate(
            input_ids=torch.tensor([ids], device=device),
            attention_mask=torch.ones(1, len(ids), dtype=torch.long, device=device),
            max_new_tokens=budget,
            do_sample=False,
            repetition_penalty=1.1,
            eos_token_id=eos_id,
            pad_token_id=eos_id,
        )[0][len(ids):].tolist()
        results.append({
            "instruction": row["instruction"],
            "input": row.get("input", ""),
            "generated": tok.decode(out, skip_special_tokens=True).strip(),
            "stopped": eos_id in out,
            "new_tokens": len(out),
        })
    model.train()
    return results


def load_saved_delta(module, directory, filename, token_ids=None):
    """Start a stop-token / marker module from the values saved with an adapter."""
    if not directory:
        return
    path = Path(directory) / filename
    if not path.exists():
        print(f"  {filename} not found in {directory}; starting it from zero")
        return
    saved = torch.load(path, map_location="cpu")
    if token_ids is not None and list(saved["token_ids"]) != list(token_ids):
        raise SystemExit(f"{path} was trained for different template tokens")
    with torch.no_grad():
        module.delta.copy_(saved["delta"].to(module.delta.device, module.delta.dtype))
    print(f"  loaded {filename} from {directory}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="iamshnoo/bul_100m")
    ap.add_argument("--lang", default="bul", choices=["bul", "deu"])
    ap.add_argument("--train", default="data/splits/train.json")
    ap.add_argument("--val", default="data/splits/val.json")
    ap.add_argument("--out", default="adapters/bul_instruct")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=None,
                    help="default 2e-4 for LoRA, 5e-5 with --full-finetune")
    ap.add_argument("--full-finetune", action="store_true",
                    help="train every weight of the base model instead of a LoRA adapter; "
                         "saves a full model copy (~1.25 GB each for final and best)")
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--init-adapter", default=None,
                    help="continue training from this saved LoRA adapter folder (with its "
                         "stop_token.pt / marker_embeddings.pt if present); --rank, --alpha "
                         "and --dropout are then taken from the saved adapter")
    ap.add_argument("--train-stop-token", action="store_true",
                    help="also train the lm_head row of <|end_of_text|> (saved as stop_token.pt; "
                         "load the result with stop_token.load_instruct_model)")
    ap.add_argument("--stop-token-lr", type=float, default=2e-3)
    ap.add_argument("--train-marker-embeddings", action="store_true",
                    help="also train the input embeddings of the prompt-template tokens "
                         "(saved as marker_embeddings.pt)")
    ap.add_argument("--marker-lr", type=float, default=1e-3)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--max-train-rows", type=int, default=None,
                    help="train on a random subset of this many rows (trial runs)")
    ap.add_argument("--eval-every", type=int, default=200, help="steps between validation passes")
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--n-samples", type=int, default=8,
                    help="validation prompts to generate from after training")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    if args.lr is None:
        args.lr = 5e-5 if args.full_finetune else 2e-4
    if args.full_finetune and (args.train_stop_token or args.train_marker_embeddings):
        raise SystemExit("--full-finetune already trains the stop-token and marker rows; "
                         "drop the other --train-* flags")
    if args.full_finetune and args.init_adapter:
        raise SystemExit("--init-adapter continues a LoRA adapter; it cannot be combined "
                         "with --full-finetune")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    bos_id, eos_id = special_ids(tok)
    pad_id = eos_id  # padding is masked out of attention and loss

    train_ex, train_dropped = load_examples(
        args.train, tok, args.lang, args.max_len, args.max_train_rows, args.seed)
    val_ex, val_dropped = load_examples(args.val, tok, args.lang, args.max_len)
    print(f"train: {len(train_ex)} examples ({train_dropped} dropped as > {args.max_len} tokens)")
    print(f"val:   {len(val_ex)} examples ({val_dropped} dropped)")

    # float32: the 100M model is small, and bf16 training is unreliable on MPS.
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32)
    if args.max_len > model.config.max_position_embeddings:
        raise SystemExit(f"--max-len exceeds the model context "
                         f"({model.config.max_position_embeddings})")
    model.generation_config.bos_token_id = bos_id
    model.generation_config.eos_token_id = eos_id
    model.generation_config.pad_token_id = eos_id

    if args.full_finetune:
        print(f"full fine-tuning: {model.num_parameters() / 1e6:.1f}M trainable params")
    elif args.init_adapter:
        model = PeftModel.from_pretrained(model, args.init_adapter, is_trainable=True)
        print(f"continuing from adapter {args.init_adapter}")
        model.print_trainable_parameters()
    else:
        model = get_peft_model(model, LoraConfig(
            task_type="CAUSAL_LM",
            r=args.rank,
            lora_alpha=args.alpha,
            lora_dropout=args.dropout,
            target_modules=TARGET_MODULES,
            bias="none",
        ))
        model.print_trainable_parameters()
    model.to(device)
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    groups = [{"params": params, "lr": args.lr}]
    stop_row = None
    if args.train_stop_token:
        stop_row = StopTokenRow(model, eos_id).to(device)
        load_saved_delta(stop_row, args.init_adapter, "stop_token.pt")
        groups.append({"params": [stop_row.delta], "lr": args.stop_token_lr})
        params = params + [stop_row.delta]
        print(f"training stop-token output row: {stop_row.delta.numel()} extra params")
    markers = None
    if args.train_marker_embeddings:
        ids = marker_token_ids(tok, args.lang)
        markers = MarkerEmbeddings(model, ids).to(device)
        load_saved_delta(markers, args.init_adapter, "marker_embeddings.pt", ids)
        groups.append({"params": [markers.delta], "lr": args.marker_lr})
        params = params + [markers.delta]
        print(f"training marker embeddings for {len(ids)} tokens: "
              f"{[tok.decode([i]) for i in ids]}")
    opt = torch.optim.AdamW(groups, weight_decay=args.weight_decay)

    def save_adapter(directory):
        model.save_pretrained(directory)
        if stop_row is not None:
            stop_row.save(directory)
        if markers is not None:
            markers.save(directory)

    steps_per_epoch = math.ceil(len(train_ex) / args.batch_size)
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    warmup = max(1, int(total_steps * args.warmup_ratio))

    def lr_at(step):  # linear warmup, cosine decay to zero
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total_steps - warmup)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    (out_dir / "train_args.json").write_text(json.dumps(vars(args), indent=2))
    log_path = out_dir / "train_log.jsonl"
    log_path.write_text("")

    def log(record):
        with log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")

    base_val = evaluate(model, val_ex, args.batch_size, pad_id, device)
    print(f"step 0/{total_steps}  val_loss {base_val:.4f}  (untrained adapter = base model)")
    log({"step": 0, "val_loss": base_val})

    rng = random.Random(args.seed)
    best_val, step, running, t0 = float("inf"), 0, [], time.time()
    while step < total_steps:
        for idxs in make_batches(train_ex, args.batch_size, rng):
            ids, mask, labels = collate(train_ex, idxs, pad_id, device)
            loss = model(input_ids=ids, attention_mask=mask, labels=labels).loss
            if not torch.isfinite(loss):
                raise SystemExit(f"non-finite loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            running.append(loss.item())
            del ids, mask, labels, loss
            if step % 4 == 0:
                free_device_cache(device)

            if step % args.log_every == 0 or step == total_steps:
                mean = sum(running) / len(running)
                running = []
                rate = step / (time.time() - t0)
                mem = (f"  gpu {torch.mps.driver_allocated_memory() / 2**30:.1f} GB"
                       if device == "mps" else "")
                print(f"step {step}/{total_steps}  epoch {step / steps_per_epoch:.2f}  "
                      f"loss {mean:.4f}  lr {sched.get_last_lr()[0]:.2e}  "
                      f"eta {(total_steps - step) / rate / 60:.1f} min{mem}", flush=True)
                log({"step": step, "train_loss": mean, "lr": sched.get_last_lr()[0]})

            if step % args.eval_every == 0 or step == total_steps:
                val = evaluate(model, val_ex, args.batch_size, pad_id, device)
                note = ""
                if val < best_val:
                    best_val = val
                    save_adapter(out_dir / "best")
                    note = "  (saved best)"
                print(f"step {step}/{total_steps}  val_loss {val:.4f}{note}", flush=True)
                log({"step": step, "val_loss": val})

            if step >= total_steps:
                break

    save_adapter(out_dir)  # final weights; best-by-val is in out/best
    tok.save_pretrained(out_dir)
    print(f"\nval_loss: base {base_val:.4f} -> best {best_val:.4f}")
    print(f"{'model' if args.full_finetune else 'adapter'} saved to {out_dir} (final) and {out_dir / 'best'} (lowest val loss)")

    if args.n_samples > 0:
        val_rows = json.loads(Path(args.val).read_text(encoding="utf-8"))[:args.n_samples]
        samples = sample_generations(model, tok, val_rows, args.lang, args.max_len, device)
        (out_dir / "samples.json").write_text(
            json.dumps(samples, ensure_ascii=False, indent=1), encoding="utf-8")
        stopped = sum(s["stopped"] for s in samples)
        print(f"\nstopped with <|end_of_text|>: {stopped}/{len(samples)} sample generations")
        for s in samples[:3]:
            print(f"\n--- {s['instruction'][:100]}")
            print(s["generated"][:400])
            print(f"[stopped={s['stopped']}, {s['new_tokens']} tokens]")


if __name__ == "__main__":
    main()
