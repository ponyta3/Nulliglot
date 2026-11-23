"""Convert the downloaded Bulgarian benchmark test sets to one plain JSON format.

    python prepare_benchmarks.py

Reads the parquet files under data/benchmarks/raw/ (downloaded from
INSAIT-Institute/MMLU-bgeval, arc-easy-bgeval and arc-challenge-bgeval) and
writes data/benchmarks/{mmlu,arc_easy,arc_challenge}.json, each a list of

    {"question": str, "choices": [str, ...], "answer": int, "subject": str}

where "answer" is the index of the correct choice. Plain JSON means the
scoring script needs no parquet reader.
"""

import json
from pathlib import Path

import pandas as pd

RAW = Path("data/benchmarks/raw")
OUT = Path("data/benchmarks")
SOURCES = {
    "mmlu": "MMLU-bgeval",
    "arc_easy": "arc-easy-bgeval",
    "arc_challenge": "arc-challenge-bgeval",
}


def convert(name, repo):
    df = pd.read_parquet(RAW / repo / "data" / "test-00000-of-00001.parquet")
    rows, skipped = [], 0
    for r in df.to_dict("records"):
        if name == "mmlu":
            choices, answer, subject = list(r["choices"]), int(r["answer"]), r["subject"]
        else:
            choices = list(r["choices"]["text"])
            labels = [str(x) for x in r["choices"]["label"]]
            if str(r["answerKey"]) not in labels:
                skipped += 1
                continue
            answer, subject = labels.index(str(r["answerKey"])), name
        choices = [str(c).strip() for c in choices]
        if len(choices) < 2 or not all(choices) or not 0 <= answer < len(choices):
            skipped += 1
            continue
        rows.append({"question": str(r["question"]).strip(), "choices": choices,
                     "answer": answer, "subject": subject})
    (OUT / f"{name}.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    sizes = sorted({len(r["choices"]) for r in rows})
    print(f"{name}: {len(rows)} questions ({skipped} skipped), choices per question {sizes}")


if __name__ == "__main__":
    for name, repo in SOURCES.items():
        convert(name, repo)
