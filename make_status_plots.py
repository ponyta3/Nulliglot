"""Draw the current-status figures for the bul_100m curriculum (Stage B to Phase 3).

    .venv/bin/python make_status_plots.py

Reads the train_log.jsonl and the evaluation JSON files of each run and writes
PNGs to plots_current_status/. make_plots.py covers the earlier runs (12-14 Nov).
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).parent
AD = ROOT / "adapters"
OUT = ROOT / "plots_current_status"
OUT.mkdir(exist_ok=True)

# Same colours as make_plots.py.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
GREY, BLUE, ORANGE, GREEN = "#77766f", "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10.5,
    "text.color": INK, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 0, "ytick.major.size": 0,
    "lines.linewidth": 2, "legend.frameon": False, "legend.fontsize": 9.5,
})


def load(path):
    return json.loads((AD / path).read_text(encoding="utf-8"))


def read_log(run):
    recs = [json.loads(l) for l in (AD / run / "train_log.jsonl").read_text().splitlines() if l]
    val = [(r["step"], r["val_loss"]) for r in recs if "val_loss" in r]
    train = [(r["step"], r["train_loss"]) for r in recs if "train_loss" in r]
    return val, train


def smooth(ys, k=25):
    out, s = [], 0.0
    for i, y in enumerate(ys):
        s += y
        if i >= k:
            s -= ys[i - k]
        out.append(s / min(i + 1, k))
    return out


def dot(ax, x, y, text, color, dx=6, dy=0, ha="left"):
    ax.plot([x], [y], "o", ms=6, color=color, mec=SURFACE, mew=1.5, zorder=5)
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                va="center", ha=ha, fontsize=9.5, color=INK2)


def save(fig, name, note, note_y=-0.03):
    fig.text(0.01, note_y, note, fontsize=8.5, color=INK2, va="top")
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def loss_figure(stages, name, title, note, ylim, xlabel, figsize):
    """Training and validation loss over consecutive runs on one step axis."""
    fig, ax = plt.subplots(figsize=figsize)
    offset, first = 0, True
    for run, label, val_note in stages:
        val, train = read_log(run)
        tx, ty = zip(*train)
        vx, vy = zip(*val)
        tx = [x + offset for x in tx]
        vx = [x + offset for x in vx]
        ax.plot(tx, smooth(list(ty)), color=ORANGE, lw=1.6,
                label="training loss (running mean of 500 steps)" if first else None)
        ax.plot(vx, vy, color=BLUE, label="validation loss (300 held-out rows)" if first else None)
        end = offset + val[-1][0]
        if offset:
            ax.axvline(offset, color=INK2, lw=0.8, ls=(0, (2, 3)))
        tr = ax.get_xaxis_transform()
        ax.text((offset + end) / 2, 1.115, label, ha="center", va="bottom", fontsize=10, color=INK, transform=tr)
        ax.text((offset + end) / 2, 1.10, val_note, ha="center", va="top", fontsize=8.5, color=INK2, transform=tr)
        if first:
            dot(ax, vx[0], vy[0], f"{vy[0]:.2f}", BLUE, dx=8)
        dot(ax, vx[-1], vy[-1], f"{vy[-1]:.2f}", BLUE, dx=-6, dy=11, ha="right")
        offset, first = end, False
    ax.set_xlim(0, offset * 1.02)
    ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("loss per answer token")
    ax.xaxis.set_major_formatter(lambda x, _: f"{int(x):,}")
    ax.set_title(title, pad=58)
    ax.legend(loc="upper right")
    save(fig, name, note)


# ---------------------------------------------------------------- figure 1
def fig_reading_loss():
    loss_figure(
        [("stage_b_lora", "Stage B", "copy + extract\n128,000 rows"),
         ("phase1_lora", "Phase 1", "copy + extract, 640,000 rows"),
         ("phase2_lora", "Phase 2", "20 instruction\ntasks"),
         ("phase2b_lora", "Phase 2b", "same 20 tasks, continued")],
        "1_reading_curriculum_loss.png",
        "Reading curriculum: loss falls in every stage, training and validation together",
        "Frozen bul_100m, LoRA rank 256. Phase 2 uses a new validation set (20 tasks), so its curve starts at a different level than Phase 1 ends.\n"
        "Training and validation loss stay together in every stage: no overfitting.",
        (0, 4.6), "training step, all reading stages in order", (12.5, 5.2))


# ---------------------------------------------------------------- figure 2
def fig_question_loss():
    loss_figure(
        [("pilot_plain", "Pilot A", "10,395 steps"),
         ("phase3_lora", "Phase 3 (final adapter)", "31,185 more steps")],
        "2_question_answering_loss.png",
        "Question answering: validation loss flattens near 2.67",
        "Alpaca questions mixed with 25% reading tasks, started from the Phase 2b adapter. Validation loss is on the 300 held-out rows of this mix.\n"
        "Training loss keeps falling below validation, but validation never rises.",
        (2.0, 5.4), "training step, question-answering stage", (10, 5.2))


# ---------------------------------------------------------------- figure 3
def fig_copy_skill():
    stages = [("Base model", None), ("Stage B", "stage_b_lora"), ("Phase 1", "phase1_lora"),
              ("Phase 2", "phase2_lora"), ("Phase 2b", "phase2b_lora"), ("Phase 3\n(questions)", "phase3_lora")]
    base = load("phase1_lora/test_copy_eval.json")["summary"]["base"]
    own, wrong = [base["overlap_true"]], [base["overlap_wrong"]]
    for _, run in stages[1:]:
        a = load(f"{run}/test_copy_eval.json")["summary"]["adapter"]
        own.append(a["overlap_true"]); wrong.append(a["overlap_wrong"])
    x = list(range(len(stages)))
    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.axhline(0.5, color=INK2, lw=0.9, ls=(0, (4, 3)))
    ax.text(0.02, 0.515, "gate set for Phase 1: 0.5", fontsize=9, color=INK2)
    ax.plot(x, wrong, color=GREY, marker="o", ms=7, mec=SURFACE, mew=1.5,
            label="overlap with a wrong passage (what guessing gives)")
    ax.plot(x, own, color=BLUE, marker="o", ms=7, mec=SURFACE, mew=1.5,
            label="overlap with the passage it was asked to copy")
    for i, v in enumerate(own):
        ax.annotate(f"{v:.2f}", (i, v), xytext=(-12, 8) if i == 1 else (0, 10), textcoords="offset points",
                    ha="right" if i == 1 else "center", fontsize=9.5, color=INK2)
    ax.annotate(f"{wrong[-1]:.2f}", (x[-1], wrong[-1]), xytext=(0, -15), textcoords="offset points",
                ha="center", fontsize=9.5, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels([s for s, _ in stages])
    ax.set_xlim(-0.3, len(stages) - 0.7); ax.set_ylim(0, 1.05)
    ax.set_ylabel("word-overlap F1 on unseen passages")
    ax.set_title("Copying a text from the prompt: learned in Phase 1, partly lost in Phase 3", fontsize=12)
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.3))
    save(fig, "3_copy_skill_by_stage.png",
         "1.0 = copied exactly. Test passages are never trained on (200 rows for the base model, Stage B and Phase 1; 100 rows from Phase 2 on).")


# ---------------------------------------------------------------- figure 4
def fig_tasks():
    b = load("phase2b_lora/test_tasks_eval.json")["summary"]
    c = load("phase3_lora/test_tasks_eval.json")["summary"]
    tasks = sorted([t for t in b if t != "ALL"], key=lambda t: (b[t]["exact"], c[t]["exact"]))
    fig, ax = plt.subplots(figsize=(10.5, 8.6))
    h = 0.36
    for i, t in enumerate(tasks):
        for off, s, color in ((h / 2 + 0.02, b, BLUE), (-h / 2 - 0.02, c, ORANGE)):
            v = s[t]["exact"] * 100
            ax.barh(i + off, v, height=h, color=color)
            ax.text(v + 1, i + off, f"{v:.0f}", va="center", fontsize=8.5, color=INK2)
    from matplotlib.patches import Patch
    handles = [Patch(color=BLUE, label=f"Phase 2b adapter (reading): {b['ALL']['exact'] * 100:.1f}% over all tasks"),
               Patch(color=ORANGE, label=f"Phase 3 adapter (questions): {c['ALL']['exact'] * 100:.1f}% over all tasks")]
    ax.set_yticks(range(len(tasks))); ax.set_yticklabels([t.replace("_", " ") for t in tasks])
    ax.set_xlim(0, 100); ax.set_ylim(-0.7, len(tasks) - 0.3)
    ax.set_xlabel("answers exactly right (%), 30 unseen rows per task")
    ax.grid(axis="y", visible=False)
    ax.set_title("Reading tasks: taking the start of a text works, looking up a named word does not", fontsize=11.5)
    ax.legend(handles=handles, loc="lower right")
    save(fig, "4_reading_tasks_exact.png",
         "With 30 rows per task, one row is 3.3 points, so small differences between the two adapters are noise.", note_y=0.02)


# ---------------------------------------------------------------- figure 5
def fig_open_questions():
    ordinary = load("hopper_bul_instruct/test_eval.json")["summary"]["adapter"]
    pilot = load("pilot_plain/test_eval.json")["summary"]
    p3 = load("phase3_lora/test_eval.json")["summary"]
    dec = load("phase3_lora/decode_eval.json")["summary"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"wspace": 0.28, "width_ratios": [1, 1.5]})

    names = ["Ordinary\ntuning", "Pilot A", "Phase 3"]
    gaps = [ordinary["loss_gap"], pilot["loss_gap"], p3["loss_gap"]]
    bars = a.bar(names, gaps, width=0.5, color=BLUE)
    for r, v in zip(bars, gaps):
        a.text(r.get_x() + r.get_width() / 2, v + 0.004, f"{v:.2f}", ha="center", fontsize=10, color=INK)
    a.set_ylim(0, 0.22); a.grid(axis="x", visible=False)
    a.set_ylabel("loss with a wrong question minus loss with its own")
    a.set_title("Loss gap (higher = the question matters more)", fontsize=11.5)

    names = ["Ordinary\ntuning", "Pilot A", "Phase 3", "Phase 3 +\ndecoding rules"]
    own = [ordinary["overlap_true"], pilot["plain"]["overlap_true"], dec["baseline"]["overlap_true"], dec["boost1"]["overlap_true"]]
    wrong = [ordinary["overlap_wrong"], pilot["plain"]["overlap_wrong"], dec["baseline"]["overlap_wrong"], dec["boost1"]["overlap_wrong"]]
    w = 0.36
    for i, (o, x) in enumerate(zip(own, wrong)):
        b.bar(i - w / 2 - 0.02, o, width=w, color=BLUE, label="its own reference answer" if i == 0 else None)
        b.bar(i + w / 2 + 0.02, x, width=w, color=GREY, label="a wrong reference answer" if i == 0 else None)
        b.text(i - w / 2 - 0.02, o + 0.004, f"{o:.3f}", ha="center", fontsize=9.5, color=INK)
        b.text(i + w / 2 + 0.02, x + 0.004, f"{x:.3f}", ha="center", fontsize=9.5, color=INK)
    b.set_xticks(range(4)); b.set_xticklabels(names)
    b.set_ylim(0, 0.22); b.grid(axis="x", visible=False)
    b.set_ylabel("word-overlap F1 of the generated answer")
    b.set_title("Word overlap of the answer (1.0 = identical)", fontsize=11.5)
    b.legend(loc="upper left", title="Overlap with", title_fontsize=9.5, alignment="left")
    fig.suptitle("Open questions: answers follow the question's words, but are far from the reference",
                 x=0.01, ha="left", fontsize=13.5, fontweight="bold", y=1.02)
    save(fig, "5_open_question_relevance.png",
         "200 held-out Alpaca test rows. Ordinary tuning = LoRA rank 16 on Alpaca only (Hopper run). "
         "Decoding rules = no repeated 4-token runs, no Latin tokens, mild boost for the question's words.", note_y=-0.06)


# ---------------------------------------------------------------- figure 6
def fig_benchmarks():
    r1 = load("benchmark_results.json")
    r3 = load("benchmark_results_phase3.json")
    benches = [("mmlu", "MMLU"), ("arc_easy", "ARC Easy"), ("arc_challenge", "ARC Challenge")]
    systems = [("Base model", r1["results"]["base"], GREY),
               ("Ordinary tuning", r1["results"]["lora"], BLUE),
               ("After the curriculum (Phase 3)", r3["results"]["phase3"], GREEN)]
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    w = 0.25
    for j, (label, res, color) in enumerate(systems):
        for i, (key, _) in enumerate(benches):
            v = res[key]["template"]["mean"] * 100
            x = i + (j - 1) * (w + 0.02)
            ax.bar(x, v, width=w, color=color, label=label if i == 0 else None)
            ax.text(x, v + 0.7, f"{v:.1f}", ha="center", va="bottom", fontsize=9.5, color=INK, zorder=6,
                    bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1.2})
    ax.axhline(25, color=INK, lw=1.1, ls=(0, (4, 3)), zorder=4)
    ax.text(3.0, 25.5, "chance: 25%", ha="right", va="bottom", fontsize=9.5, color=INK2)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{name}\n{r1['n'][key]:,} questions" for key, name in benches])
    ax.set_ylim(0, 40); ax.set_xlim(-0.55, 3.02); ax.grid(axis="x", visible=False)
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Benchmarks: every system is at chance")
    ax.legend(loc="upper left", ncol=3)
    save(fig, "6_benchmarks.png",
         "Bulgarian MMLU and ARC, four choices each, scored by the mean likelihood of each choice with the instruction template.")


if __name__ == "__main__":
    fig_reading_loss()
    fig_question_loss()
    fig_copy_skill()
    fig_tasks()
    fig_open_questions()
    fig_benchmarks()
