"""Pre-train the Tamil base model from scratch on the token files made by prepare_tokens.py.

    python pretrain.py <tokens_dir> <tokenizer_dir> <out_dir> [options]

Running the same command again continues an interrupted run from its last checkpoint.

Model
- A Llama-style decoder (LlamaForCausalLM of `transformers`), the same family and layer shape as
  the Bulgarian bul_100m: 12 layers, width 768, 12 heads, MLP width 3072. It is saved in the
  Hugging Face format and loads with AutoModelForCausalLM, with no custom code.
- Different from bul_100m: the Tamil 32,000 vocabulary with one shared input/output embedding
  table (bul_100m: 128,256 tokens, two tables, 197M of its 310M parameters), and a context of
  1,024 tokens (bul_100m: 512). About 138M parameters.
- Weights start from a normal distribution with std 0.02; the two layers that write into the
  residual stream (o_proj, down_proj) start smaller, at 0.02 / sqrt(2 * layers), as in GPT-2.

Data
- The training text is one long stream of token ids: begin-of-text, a document, end-of-text, the
  next document. Every training window is `context` tokens of continuous text taken from this
  stream, so the model always sees a long stretch of one document and has to use it. No
  sentence is shuffled and nothing is padded.
- One epoch cuts the stream into windows that do not overlap and reads every window once, in
  random order. Every epoch shifts the cut by a different random amount, so a token is seen
  with a different amount of text before it each time.
- The order depends only on the seed and the epoch number, so a continued run reads exactly
  what the interrupted run would have read.

Training
- AdamW (betas 0.9 and 0.95, weight decay 0.1 on the weight matrices, none on the norm
  weights), gradient clipping at 1.0, linear warm-up, then a cosine fall to a tenth of the peak
  learning rate. On a GPU: 16-bit (bfloat16) mixed precision and torch.compile.
- Only the training split is trained on, and only the validation split is measured. The test
  split is never opened here.

Measured at every evaluation, written to eval.tsv
  val_loss, val_ppl   loss per token on the whole validation split, and its perplexity
  val_bpc             the same as bits per character (comparable between vocabulary sizes)
  train_loss          loss on a fixed sample of training windows, measured the same way; its
                      distance to val_loss shows over-fitting
  ctx_<k>             loss on the same 64 validation tokens when the model is given only the k
                      tokens before them. If the model uses its context, the loss falls as k
                      grows. (bul_100m: 3.72 with none, 3.50 with 8, 3.41 with 128.)
  rep_plain, rep_copy loss on a 256-token passage after 256 tokens of its own document, and
                      after a copy of the passage itself. A model that reads its context has
                      rep_copy far below rep_plain. (bul_100m: 3.50 and 3.43.)
  samples.txt holds text generated after four validation openings at every evaluation.

Written to <out_dir>
  run.json     all settings, sizes and package versions
  log.tsv      training loss, learning rate, gradient norm and speed
  eval.tsv     the measurements above
  best/        the model with the lowest val_loss so far, with the tokenizer
  final/       the model after the last step, with the tokenizer
  ckpt_last.pt everything needed to continue the run
"""
import argparse, hashlib, json, math, os, shutil, signal, sys, time
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

BOS, EOS, PAD = 1, 2, 0
# settings that must be the same when a run is continued
FIXED = ["layers", "hidden", "heads", "intermediate", "context", "dropout", "epochs", "batch_size",
         "grad_accum", "lr", "min_lr_ratio", "warmup_steps", "weight_decay", "beta2", "clip",
         "max_steps", "seed"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def build_model(a, V):
    import transformers
    from transformers import LlamaConfig, LlamaForCausalLM
    transformers.utils.logging.disable_progress_bar()
    cfg = LlamaConfig(
        vocab_size=V, hidden_size=a.hidden, intermediate_size=a.intermediate, num_hidden_layers=a.layers,
        num_attention_heads=a.heads, num_key_value_heads=a.heads, max_position_embeddings=a.context,
        hidden_act="silu", rms_norm_eps=1e-5, rope_theta=10000.0, attention_bias=False, mlp_bias=False,
        attention_dropout=a.dropout, initializer_range=0.02, tie_word_embeddings=True,
        bos_token_id=BOS, eos_token_id=EOS, pad_token_id=PAD, use_cache=True)
    torch.manual_seed(a.seed)
    model = LlamaForCausalLM(cfg)
    n = 0
    for name, p in model.named_parameters():
        if name.endswith(("o_proj.weight", "down_proj.weight")):
            torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * a.layers)); n += 1
    assert n == 2 * a.layers, n
    assert model.get_output_embeddings().weight is model.get_input_embeddings().weight, "embeddings are not shared"
    return model


def make_optimizer(model, a, dev):
    decay = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    plain = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    groups = [{"params": decay, "weight_decay": a.weight_decay}, {"params": plain, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=a.lr, betas=(0.9, a.beta2), eps=1e-8, fused=(dev.type == "cuda"))


class Plan:
    """Which windows of the training stream make up each step."""
    def __init__(self, n_tokens, T, per_step, epochs, seed, max_steps):
        self.N, self.T, self.B, self.seed = n_tokens, T, per_step, seed
        self.steps = [self._rn(e)[2] // per_step for e in range(epochs)]
        assert min(self.steps) > 0, "the training stream is shorter than one step"
        self.cum = np.cumsum([0] + self.steps)
        self.total = int(self.cum[-1]) if not max_steps else min(max_steps, int(self.cum[-1]))
        self._e, self._b = -1, None

    def _rn(self, e):
        rng = np.random.default_rng([self.seed, e])
        r = int(rng.integers(0, self.T))               # shift of the cut in this epoch
        return rng, r, (self.N - 1 - r) // self.T      # each window needs T + 1 tokens

    def epoch_of(self, step):
        return int(np.searchsorted(self.cum, step, side="right") - 1)

    def starts(self, step):
        """Start positions of the B windows of this step."""
        e = self.epoch_of(step)
        if e != self._e:
            rng, r, n = self._rn(e)
            self._b = (r + self.T * rng.permutation(n)[:self.steps[e] * self.B]).reshape(self.steps[e], self.B)
            self._e = e
        return self._b[step - int(self.cum[e])]


def lr_at(step, a, total, warmup):
    if step < warmup: return a.lr * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return a.lr * (a.min_lr_ratio + (1 - a.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))


def windows(stream, starts, length, dev):
    ix = np.asarray(starts)[:, None] + np.arange(length)[None, :]
    return torch.from_numpy(stream[ix].astype(np.int64)).to(dev, non_blocking=True)


class Meter:
    """Loss measurements on fixed data, the same at every evaluation."""
    def __init__(self, a, train, val, val_idx, val_chars, dev, amp):
        self.a, self.val, self.val_chars, self.dev, self.amp = a, val, val_chars, dev, amp
        T = a.context
        rng = np.random.default_rng([a.seed, 10**6])
        # fixed sample of training windows, about as many tokens as a tenth of validation
        k = max(1, min((len(train) - 1) // T, max(8, len(val) // (10 * T))))
        self.train_x = train[(rng.choice((len(train) - 1) // T, size=k, replace=False) * T)[:, None]
                             + np.arange(T + 1)[None, :]].astype(np.int64)
        # context test: stretches of T tokens that lie inside one validation document
        self.L, self.P = max(2, T // 16), max(2, T // 4)
        self.ks = [k for k in sorted({T - self.L, 128, 32, 8, 1}) if k <= T - self.L]
        lo, hi = val_idx[:-1] + 1, val_idx[1:] - 1                 # document bodies, without begin/end marks
        room = np.maximum(0, hi - lo - T + 1)
        self.W = None
        if room.sum() > 0:
            n = int(min(256, room.sum()))
            docs = rng.choice(len(room), size=n, p=room / room.sum())
            st = lo[docs] + (rng.random(n) * room[docs]).astype(np.int64)
            self.W = val[st[:, None] + np.arange(T)[None, :]].astype(np.int64)
        n = len(val) - 1                                # predictions in the validation stream
        self.val_x = val[(np.arange(n // T) * T)[:, None] + np.arange(T + 1)[None, :]].astype(np.int64)
        self.val_rest = val[None, (n // T) * T:].astype(np.int64) if (n // T) * T < n else None
        self.cols = ["val_loss", "val_ppl", "val_bpc", "train_loss"]
        if self.W is not None: self.cols += [f"ctx_{k}" for k in self.ks] + ["rep_plain", "rep_copy"]
        # openings of validation documents, for the text samples
        self.prompts = [val[s + 1:s + 17].astype(np.int64) for s, e in zip(val_idx[:-1], val_idx[1:]) if e - s >= 18][:4]

    def _nll(self, net, xy, n_tgt=None):
        """Summed loss of the last n_tgt tokens of every row (all tokens but the first if None)."""
        tot = 0.0
        for i in range(0, len(xy), self.a.eval_batch):
            b = torch.from_numpy(xy[i:i + self.a.eval_batch]).to(self.dev)
            with self.amp():
                logits = net(input_ids=b[:, :-1], use_cache=False).logits
            k = b.shape[1] - 1 if n_tgt is None else n_tgt
            tot += F.cross_entropy(logits[:, -k:].float().reshape(-1, logits.shape[-1]), b[:, -k:].reshape(-1),
                                   reduction="sum").item()
        return tot

    @torch.no_grad()
    def measure(self, net):
        net.eval()
        T, n = self.a.context, len(self.val) - 1
        tot = self._nll(net, self.val_x)
        if self.val_rest is not None: tot += self._nll(net, self.val_rest)
        r = {"val_loss": tot / n, "val_ppl": math.exp(min(20.0, tot / n)), "val_bpc": tot / (self.val_chars * math.log(2)),
             "train_loss": self._nll(net, self.train_x) / (self.train_x.shape[0] * T)}
        if self.W is not None:
            W, L, P, m = self.W, self.L, self.P, len(self.W)
            for k in self.ks: r[f"ctx_{k}"] = self._nll(net, W[:, T - L - k:], L) / (m * L)
            r["rep_plain"] = self._nll(net, W[:, T - 2 * P:], P) / (m * P)
            r["rep_copy"] = self._nll(net, np.concatenate([W[:, T - P:], W[:, T - P:]], 1), P) / (m * P)
        net.train()
        return r

    @torch.no_grad()
    def samples(self, net, tok, n_new=120, temperature=0.8, top_p=0.9):
        if not self.prompts: return []
        net.eval()
        gen = torch.Generator().manual_seed(self.a.seed)
        x = torch.from_numpy(np.stack([np.concatenate(([BOS], p)) for p in self.prompts])).to(self.dev)
        for _ in range(n_new):
            with self.amp():
                logits = net(input_ids=x[:, -self.a.context:], use_cache=False).logits[:, -1]
            p = torch.softmax(logits.float().cpu() / temperature, -1)
            sp, si = torch.sort(p, descending=True)
            sp[(torch.cumsum(sp, -1) - sp) > top_p] = 0.0
            nxt = si.gather(1, torch.multinomial(sp / sp.sum(-1, keepdim=True), 1, generator=gen))
            x = torch.cat([x, nxt.to(self.dev)], 1)
        net.train()
        out = []
        for row, p in zip(x.cpu().tolist(), self.prompts):
            new = row[1 + len(p):]
            if EOS in new: new = new[:new.index(EOS)]
            out.append((tok.decode(list(p), skip_special_tokens=False), tok.decode(new, skip_special_tokens=False)))
        return out


def save_model(net, tok_dir, path):
    """Write the model and the tokenizer to `path`, replacing an older copy only when complete."""
    tmp = path + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    net.save_pretrained(tmp)
    for f in ("tokenizer.json", "tokenizer_config.json"): shutil.copy(os.path.join(tok_dir, f), tmp)
    if os.path.isdir(path):
        os.rename(path, path + ".old")
    os.rename(tmp, path)
    shutil.rmtree(path + ".old", ignore_errors=True)


def keep_rows(path, step):
    """Drop the rows of a log written after `step` (a continued run writes them again)."""
    if not os.path.exists(path): return
    rows = open(path, encoding="utf-8").read().split("\n")
    kept = [rows[0]] + [r for r in rows[1:] if r and int(r.split("\t")[0]) <= step]
    open(path, "w", encoding="utf-8").write("\n".join(kept) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tokens_dir"); ap.add_argument("tokenizer_dir"); ap.add_argument("out_dir")
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--hidden", type=int, default=768)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--intermediate", type=int, default=3072)
    ap.add_argument("--context", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.0, help="dropout on the attention weights")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32, help="windows per forward pass")
    ap.add_argument("--grad-accum", type=int, default=2, help="forward passes per step; windows per step = batch-size x grad-accum")
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--min-lr-ratio", type=float, default=0.1)
    ap.add_argument("--warmup-steps", type=int, default=1000)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (for tests); 0 = all epochs")
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-compile", action="store_true")
    a = ap.parse_args()

    dev = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device)
    cuda = dev.type == "cuda"
    amp = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if cuda else (lambda: torch.autocast("cpu", enabled=False))
    if cuda:
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True

    # ---- data
    meta = json.load(open(os.path.join(a.tokens_dir, "meta.json")))
    tok_file = os.path.join(a.tokenizer_dir, "tokenizer.json")
    if sha256(tok_file) != meta["tokenizer_sha256"]:
        sys.exit("the token files were made with a different tokenizer than " + tok_file)
    tok = Tokenizer.from_file(tok_file)
    V, T = meta["vocab_size"], a.context
    assert tok.get_vocab_size() == V and meta["bos"] == BOS and meta["eos"] == EOS
    train = np.fromfile(os.path.join(a.tokens_dir, "train.bin"), dtype=np.uint16)
    val = np.fromfile(os.path.join(a.tokens_dir, "val.bin"), dtype=np.uint16)
    val_idx = np.load(os.path.join(a.tokens_dir, "val.idx.npy"))
    assert len(train) == meta["splits"]["train"]["tokens"] and len(val) == meta["splits"]["val"]["tokens"] == val_idx[-1]
    assert len(val) > T + 1, "the validation stream is shorter than one window"
    per_step = a.batch_size * a.grad_accum
    plan = Plan(len(train), T, per_step, a.epochs, a.seed, a.max_steps)
    total = plan.total
    warmup = min(a.warmup_steps, max(1, total // 10))

    # ---- model, optimizer, checkpoint
    os.makedirs(a.out_dir, exist_ok=True)
    P = lambda f: os.path.join(a.out_dir, f)
    net = build_model(a, V).to(dev)
    n_params = sum(p.numel() for p in net.parameters())
    opt = make_optimizer(net, a, dev)
    step, best, seconds = 0, {"val_loss": float("inf"), "step": -1}, 0.0
    settings = {k: getattr(a, k) for k in FIXED} | {"tokenizer_sha256": meta["tokenizer_sha256"], "train_tokens": len(train)}
    if os.path.exists(P("ckpt_last.pt")):
        ck = torch.load(P("ckpt_last.pt"), map_location=dev, weights_only=True)
        diff = {k: (ck["settings"].get(k), v) for k, v in settings.items() if ck["settings"].get(k) != v}
        if diff: sys.exit(f"{a.out_dir} holds a run with other settings (saved, now): {diff}")
        net.load_state_dict(ck["model"]); opt.load_state_dict(ck["optimizer"])
        step, best, seconds = ck["step"], ck["best"], ck["seconds"]
        keep_rows(P("log.tsv"), step); keep_rows(P("eval.tsv"), step)
        print(f"continuing from step {step} of {total}", flush=True)
    elif os.listdir(a.out_dir):
        sys.exit(f"{a.out_dir} is not empty and holds no checkpoint; not overwriting")
    fwd = net
    if cuda and not a.no_compile: fwd = torch.compile(net)

    meter = Meter(a, train, val, val_idx, meta["splits"]["val"]["chars"], dev, amp)
    if step == 0:
        import transformers
        run = {"settings": vars(a), "parameters": n_params, "vocab_size": V, "device": str(dev),
               "gpu": torch.cuda.get_device_name(0) if cuda else None,
               "train_tokens": len(train), "val_tokens": len(val), "windows_per_step": per_step,
               "tokens_per_step": per_step * T, "steps_per_epoch": plan.steps, "total_steps": total,
               "warmup_steps": warmup, "tokenizer_sha256": meta["tokenizer_sha256"],
               "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                            "numpy": np.__version__, "python": sys.version.split()[0]}}
        json.dump(run, open(P("run.json"), "w"), indent=1)
        open(P("log.tsv"), "w").write("step\tepoch\tlr\tloss\tgrad_norm\ttokens_per_s\thours\n")
        open(P("eval.tsv"), "w").write("step\tepoch\t" + "\t".join(meter.cols) + "\tbest\n")
        open(P("samples.txt"), "w", encoding="utf-8").write(
            "Text generated after four validation openings (temperature 0.8, top-p 0.9). [ ] marks the opening.\n")
    print(f"{n_params / 1e6:.1f}M parameters; {len(train)} training tokens; {per_step * T} tokens per step; "
          f"{plan.steps[0]} steps per epoch; {total} steps; warm-up {warmup}; device {dev}", flush=True)

    def checkpoint():
        tmp = P("ckpt_last.pt.tmp")
        torch.save({"model": net.state_dict(), "optimizer": opt.state_dict(), "step": step, "best": best,
                    "seconds": seconds, "settings": settings}, tmp)
        os.replace(tmp, P("ckpt_last.pt"))

    def evaluate():
        nonlocal best
        r = meter.measure(net)
        if not math.isfinite(r["val_loss"]): sys.exit(f"step {step}: validation loss is not a number; stopping")
        is_best = r["val_loss"] < best["val_loss"]
        if is_best and step > 0:
            best = {"val_loss": r["val_loss"], "step": step}
            save_model(net, a.tokenizer_dir, P("best"))
            json.dump(best | {k: r[k] for k in meter.cols}, open(P("best.json"), "w"), indent=1)
        ep = step / plan.steps[0]
        with open(P("eval.tsv"), "a") as o:
            o.write(f"{step}\t{ep:.3f}\t" + "\t".join(f"{r[c]:.4f}" for c in meter.cols) + f"\t{int(is_best and step > 0)}\n")
        with open(P("samples.txt"), "a", encoding="utf-8") as o:
            o.write(f"\n=== step {step} (epoch {ep:.2f}, val_loss {r['val_loss']:.3f})\n")
            for p, g in meter.samples(net, tok): o.write(f"[{p}]{g}\n---\n")
        print(f"eval step {step} epoch {ep:.2f}: " + " ".join(f"{c} {r[c]:.3f}" for c in meter.cols)
              + (" (best)" if is_best and step > 0 else ""), flush=True)

    stop = {"now": False}
    for s in (signal.SIGTERM, signal.SIGINT): signal.signal(s, lambda *_: stop.update(now=True))

    if step == 0:
        evaluate()                                     # untrained model: loss should be near ln(vocabulary)
    net.train()
    t0 = time.time(); acc_loss, acc_n, acc_s = 0.0, 0, 0.0
    while step < total:
        t_step = time.time()
        lr = lr_at(step, a, total, warmup)
        for g in opt.param_groups: g["lr"] = lr
        st = plan.starts(step)
        loss_step = 0.0
        for m in range(a.grad_accum):
            xy = windows(train, st[m * a.batch_size:(m + 1) * a.batch_size], T + 1, dev)
            with amp():
                logits = fwd(input_ids=xy[:, :-1], use_cache=False).logits
            loss = F.cross_entropy(logits.float().reshape(-1, V), xy[:, 1:].reshape(-1))
            (loss / a.grad_accum).backward()
            loss_step += loss.item() / a.grad_accum
        if not math.isfinite(loss_step):
            sys.exit(f"step {step}: training loss is not a number; stopping. The last checkpoint is kept.")
        gn = torch.nn.utils.clip_grad_norm_(net.parameters(), a.clip if a.clip > 0 else float("inf")).item()
        opt.step(); opt.zero_grad(set_to_none=True)
        step += 1
        acc_loss += loss_step; acc_n += 1; acc_s += time.time() - t_step
        if step % a.log_every == 0 or step == total:
            hours = (seconds + time.time() - t0) / 3600      # wall time, evaluations included
            speed = acc_n * per_step * T / acc_s             # training steps only
            with open(P("log.tsv"), "a") as o:
                o.write(f"{step}\t{step / plan.steps[0]:.3f}\t{lr:.3e}\t{acc_loss / acc_n:.4f}\t{gn:.3f}\t"
                        f"{speed:.0f}\t{hours:.3f}\n")
            print(f"step {step}/{total} epoch {step / plan.steps[0]:.2f} lr {lr:.2e} loss {acc_loss / acc_n:.4f} "
                  f"grad {gn:.2f} {speed:.0f} tok/s {hours:.2f} h"
                  + (f" gpu {torch.cuda.max_memory_allocated() / 2**30:.1f} GB" if cuda else ""), flush=True)
            acc_loss, acc_n, acc_s = 0.0, 0, 0.0
        due_eval = step % a.eval_every == 0 or step == total
        if due_eval: evaluate()
        if due_eval or step % a.ckpt_every == 0 or stop["now"]:
            seconds += time.time() - t0; t0 = time.time()
            checkpoint()
        if stop["now"] and step < total:
            print(f"stopped by a signal at step {step}; run the same command again to continue", flush=True)
            sys.exit(0)

    save_model(net, a.tokenizer_dir, P("final"))
    json.dump({"steps": step, "hours": round(seconds / 3600, 3), "best": best}, open(P("done.json"), "w"), indent=1)
    print(f"done: {step} steps, {seconds / 3600:.2f} h; best val_loss {best['val_loss']:.4f} at step {best['step']}; "
          f"models in {P('best')} and {P('final')}", flush=True)


if __name__ == "__main__":
    main()
