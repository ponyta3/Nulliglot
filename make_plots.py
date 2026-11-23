"""Draw the figures for the bul_100m instruction-tuning experiments.

    .venv/bin/python make_plots.py

Reads the train_log.jsonl of each run (and the Hopper test evaluation, if it has
been copied to adapters/hopper_bul_instruct/test_eval.json) and writes PNGs to
plots/. Numbers that came from one-off tests are written in below with a note
of where they were measured.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
OUT = ROOT / "plots"
OUT.mkdir(exist_ok=True)

# One colour per system, the same in every figure.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BASE, LORA, MARKERS, FULLFT = "#77766f", "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10.5,
    "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 0, "ytick.major.size": 0,
    "lines.linewidth": 2, "legend.frameon": False, "legend.fontsize": 9.5,
})


def read_log(run):
    recs = [json.loads(l) for l in (ROOT / "adapters" / run / "train_log.jsonl").read_text().splitlines() if l]
    val = [(r["step"], r["val_loss"]) for r in recs if "val_loss" in r]
    train = [(r["step"], r["train_loss"]) for r in recs if "train_loss" in r]
    return val, train


def end_label(ax, x, y, text, color, dy=0.0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points",
                va="center", fontsize=9.5, color=INK2)
    ax.plot([x], [y], "o", ms=5, color=color, mec=SURFACE, mew=1.5, zorder=5)


def save(fig, name, note, note_y=-0.035):
    fig.text(0.01, note_y, note, fontsize=8.5, color=INK2, va="top")
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


# ---------------------------------------------------------------- figure 1
def fig_validation():
    fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.8), gridspec_kw={"wspace": 0.32})

    runs = [("bul_instruct", "LoRA + stop token", LORA, 9),
            ("bul_instruct_markers", "LoRA + stop token + marker embeddings", MARKERS, -9),
            ("bul_instruct_fullft", "Full fine-tuning (all weights)", FULLFT, 0)]
    for run, label, color, dy in runs:
        val, _ = read_log(run)
        xs, ys = zip(*val)
        a.plot(xs, ys, color=color, label=label)
        end_label(a, xs[-1], ys[-1], f"{ys[-1]:.3f}", color, dy)
    base = read_log("bul_instruct")[0][0][1]
    a.axhline(base, color=BASE, lw=1.4, ls=(0, (4, 3)))
    a.text(4632, base + 0.012, f"base model, untrained: {base:.3f}", color=INK2, fontsize=9.5, ha="right", va="bottom")
    a.set_title("Cleaned data: 24,701 training rows, laptop")
    a.set_xlabel("training step (3 epochs = 4,632 steps)")
    a.set_ylabel("validation loss per answer token")
    a.set_xlim(0, 5300); a.set_ylim(2.35, 3.45)
    a.legend(loc="upper right", bbox_to_anchor=(1.0, 0.9))

    for run, label, ls, dy in [("hopper_bul_instruct", "Hopper (A100 GPU slice)", "-", -9),
                               ("bul_instruct_raw", "Laptop (same data, settings, seed)", (0, (4, 2)), 9)]:
        val, _ = read_log(run)
        xs, ys = zip(*val)
        b.plot(xs, ys, color=LORA, ls=ls, label="LoRA + stop token, " + label)
        end_label(b, xs[-1], ys[-1], f"{ys[-1]:.3f}", LORA, dy)
    base = read_log("hopper_bul_instruct")[0][0][1]
    b.axhline(base, color=BASE, lw=1.4, ls=(0, (4, 3)))
    b.text(7191, base + 0.012, f"base model, untrained: {base:.3f}", color=INK2, fontsize=9.5, ha="right", va="bottom")
    b.set_title("Raw data: 38,344 training rows, Hopper and laptop")
    b.set_xlabel("training step (3 epochs = 7,191 steps)")
    b.set_ylabel("validation loss per answer token")
    b.set_xlim(0, 8200); b.set_ylim(2.85, 3.8)
    b.legend(loc="upper right", bbox_to_anchor=(1.0, 0.9))

    fig.suptitle("Validation loss falls in every run and flattens by the third epoch",
                 x=0.01, ha="left", fontsize=13.5, fontweight="bold", y=1.0)
    save(fig, "1_validation_loss.png",
         "The two panels use different validation sets (300 rows each), so compare curves within a panel, not across panels.")


# ---------------------------------------------------------------- figure 2
def fig_train_vs_val():
    val, train = read_log("hopper_bul_instruct")
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    tx, ty = zip(*train)
    # 10-point running mean of the per-20-step training loss
    k = 10
    sm = [sum(ty[max(0, i - k + 1):i + 1]) / len(ty[max(0, i - k + 1):i + 1]) for i in range(len(ty))]
    ax.plot(tx, ty, color=MARKERS, lw=0.8, alpha=0.35)
    ax.plot(tx, sm, color=MARKERS, label="training loss (running mean of 200 steps; faint line = every 20 steps)")
    vx, vy = zip(*val)
    ax.plot(vx, vy, color=LORA, label="validation loss (300 held-out rows, every 200 steps)")
    end_label(ax, vx[-1], vy[-1], f"validation {vy[-1]:.3f}", LORA, 8)
    end_label(ax, tx[-1], sm[-1], f"training {sm[-1]:.3f}", MARKERS, -8)
    ax.set_xlim(0, 8600); ax.set_ylim(2.8, 4.3)
    ax.set_xlabel("training step (3 epochs = 7,191 steps)")
    ax.set_ylabel("loss per answer token")
    ax.set_title("Hopper full run: training and validation loss stay together (no overfitting)", fontsize=12.5)
    ax.legend(loc="upper right")
    save(fig, "2_hopper_train_vs_validation.png",
         "Hopper job 3725405, LoRA rank 16 + stop token on frozen bul_100m, raw data, 38,344 rows, batch size 16, learning rate 2e-4.")


# ---------------------------------------------------------------- figure 3
def fig_context():
    # Measured on the laptop, 600 texts from the cleaned training answers:
    # loss on a 64-token target after a 128-token context of which only the
    # last k tokens are real (the rest comes from another text).
    keep = [0, 8, 16, 32, 64, 128]
    base = [3.409, 3.207, 3.197, 3.189, 3.182, 3.179]
    full = [2.769, 2.521, 2.505, 2.492, 2.479, 2.465]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    pos = range(len(keep))
    ax.plot(pos, base, color=BASE, marker="o", ms=6, mec=SURFACE, mew=1.5, label="Base model bul_100m")
    ax.plot(pos, full, color=FULLFT, marker="o", ms=6, mec=SURFACE, mew=1.5, label="Full fine-tuning (all weights)")
    for ys in (base, full):
        ax.annotate(f"{ys[0]:.2f}", (0, ys[0]), xytext=(8, 4), textcoords="offset points", fontsize=9.5, color=INK2)
        ax.annotate(f"{ys[1]:.2f}", (1, ys[1]), xytext=(0, 9), textcoords="offset points", fontsize=9.5, color=INK2, ha="center")
        ax.annotate(f"{ys[-1]:.2f}", (5, ys[-1]), xytext=(0, 9), textcoords="offset points", fontsize=9.5, color=INK2, ha="center")
    ax.annotate("almost all the benefit comes\nfrom the last 8 tokens", (1, base[1]), xytext=(1.5, 3.36),
                fontsize=9.5, color=INK2, arrowprops={"arrowstyle": "-", "color": INK2, "lw": 0.8})
    ax.set_xticks(list(pos)); ax.set_xticklabels([str(k) for k in keep])
    ax.set_xlabel("real context tokens kept directly before the target (of 128)")
    ax.set_ylabel("loss per target token")
    ax.set_ylim(2.35, 3.55)
    ax.set_title("The model uses only the last few tokens of context", fontsize=12.5)
    ax.legend(loc="center right")
    save(fig, "3_context_use.png",
         "Lower is better. A model that used earlier text would keep improving from 8 to 128 tokens; these lines are nearly flat.")


# ---------------------------------------------------------------- figure 4
def fig_test_eval():
    path = ROOT / "adapters" / "hopper_bul_instruct" / "test_eval.json"
    if not path.exists():
        print("skipping test-evaluation figure:", path, "not found")
        return
    s = json.loads(path.read_text(encoding="utf-8"))["summary"]
    b, a = s["base"], s["adapter"]
    panels = [
        ("Stops with the\nstop token", [b["stopped"] * 100, a["stopped"] * 100], "{:.0f}%", (0, 115), ["Base", "Adapter"]),
        ("Answer has no\nLatin letters", [b["cyrillic_only"] * 100, a["cyrillic_only"] * 100], "{:.0f}%", (0, 115), ["Base", "Adapter"]),
        ("Average answer\nlength (tokens)", [b["avg_tokens"], a["avg_tokens"]], "{:.0f}", (0, 230), ["Base", "Adapter"]),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(16, 4.6), gridspec_kw={"wspace": 0.6, "width_ratios": [1, 1, 1, 1.5, 1.5]})
    for ax, (title, vals, fmt, ylim, names) in zip(axes, panels):
        bars = ax.bar(names, vals, width=0.5, color=[BASE, LORA])
        for r, v in zip(bars, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + ylim[1] * 0.02, fmt.format(v), ha="center", fontsize=10.5, color=INK)
        ax.set_title(title, fontsize=10.5); ax.set_ylim(*ylim); ax.grid(axis="x", visible=False)

    def pair(ax, title, labels, bv, av, fmt, ylim):
        x = [0, 1, 2.6, 3.6]
        bars = ax.bar(x, bv + av, width=0.8, color=[BASE, BASE, LORA, LORA])
        bars[1].set_alpha(0.55); bars[3].set_alpha(0.55)
        for r, v in zip(bars, bv + av):
            ax.text(r.get_x() + r.get_width() / 2, v + (ylim[1] - ylim[0]) * 0.02, fmt.format(v), ha="center", fontsize=10, color=INK)
        ax.set_xticks(x); ax.set_xticklabels(labels * 2, fontsize=9.5)
        tr = ax.get_xaxis_transform()
        ax.text(0.5, -0.17, "Base", ha="center", va="top", fontsize=10.5, color=INK, transform=tr)
        ax.text(3.1, -0.17, "Adapter", ha="center", va="top", fontsize=10.5, color=INK, transform=tr)
        ax.set_title(title, fontsize=10.5); ax.set_ylim(*ylim); ax.grid(axis="x", visible=False)

    pair(axes[3], "Loss on the reference answer,\ngiven its own or a wrong prompt", ["own", "wrong"],
         [b["loss_true"], b["loss_wrong"]], [a["loss_true"], a["loss_wrong"]], "{:.3f}", (0, 4.6))
    pair(axes[4], "Word overlap of the answer with\nits own or a wrong reference", ["own", "wrong"],
         [b["overlap_true"], b["overlap_wrong"]], [a["overlap_true"], a["overlap_wrong"]], "{:.3f}",
         (0, max(0.2, 1.4 * max(b["overlap_true"], a["overlap_true"], b["overlap_wrong"], a["overlap_wrong"]))))
    fig.suptitle(f"Test evaluation on {b['n']} held-out rows: the adapter learns to stop, but barely uses the instruction",
                 x=0.01, ha="left", fontsize=13.5, fontweight="bold", y=1.1)
    save(fig, "4_test_evaluation.png",
         "If the model used the instruction, 'wrong prompt' loss would be much higher than 'own prompt', and overlap with its own reference much higher than with a wrong one.", note_y=-0.12)


if __name__ == "__main__":
    fig_validation()
    fig_train_vs_val()
    fig_context()
    fig_test_eval()
