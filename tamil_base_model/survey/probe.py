"""Sample a Hugging Face dataset through the public datasets-server API and
measure how suitable it is as Tamil-only pretraining text. No login, no download.

    python3 probe.py <dataset> [--config C] [--split S] [--field F] [--n 300]

Prints one JSON object. Run with no --config to list the configs/splits first.
"""
import argparse, json, statistics, sys, time, unicodedata, urllib.parse, urllib.request

API = "https://datasets-server.huggingface.co"

def get(path, **q):
    url = f"{API}/{path}?{urllib.parse.urlencode(q)}"
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "dataset-probe"})
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", "replace")[:300]
            if e.code in (429, 500, 502, 503) and attempt < 3:
                time.sleep(5 * (attempt + 1)); continue
            return {"error": f"HTTP {e.code}: {body}"}
        except Exception as e:
            if attempt < 3: time.sleep(5); continue
            return {"error": repr(e)}

def script_counts(text):
    ta = la = other = 0
    for ch in text:
        if "஀" <= ch <= "௿": ta += 1
        elif ch.isalpha():
            if ("a" <= ch.lower() <= "z") or "LATIN" in unicodedata.name(ch, ""): la += 1
            else: other += 1
    return ta, la, other

ap = argparse.ArgumentParser()
ap.add_argument("dataset"); ap.add_argument("--config"); ap.add_argument("--split")
ap.add_argument("--field"); ap.add_argument("--n", type=int, default=300)
a = ap.parse_args()

if not a.config:
    s = get("splits", dataset=a.dataset)
    print(json.dumps(s if "error" in s else [(x["config"], x["split"]) for x in s["splits"]], ensure_ascii=False, indent=1)[:6000])
    sys.exit()

split = a.split or "train"
size = get("size", dataset=a.dataset, config=a.config)
num_rows = None
if "error" not in size:
    for sp in size["size"]["splits"]:
        if sp["split"] == split: num_rows = sp["num_rows"]; nbytes = sp.get("num_bytes_memory")
out = {"dataset": a.dataset, "config": a.config, "split": split, "num_rows": num_rows}
if num_rows is None:
    out["size_error"] = size.get("error", "split not found"); num_rows = a.n
texts, field, truncated = [], a.field, 0
pages = max(1, a.n // 100)
for i in range(pages):
    off = int(i * max(0, num_rows - 100) / max(1, pages - 1)) if pages > 1 else 0
    r = get("rows", dataset=a.dataset, config=a.config, split=split, offset=off, length=100)
    if "error" in r: out["rows_error"] = r["error"]; break
    if field is None:
        keys = [k for k, v in r["rows"][0]["row"].items() if isinstance(v, str)] if r["rows"] else []
        named = [k for k in ("text", "content", "sentence", "document", "article", "body") if k in keys]
        field = named[0] if named else (max(keys, key=lambda k: sum(len(x["row"].get(k) or "") for x in r["rows"])) if keys else None)
    for row in r["rows"]:
        truncated += field in row.get("truncated_cells", [])
        v = row["row"].get(field)
        if isinstance(v, str): texts.append(v)
if not texts:
    print(json.dumps(out, ensure_ascii=False)); sys.exit()
words = [len(t.split()) for t in texts]
tot = [script_counts(t) for t in texts]
T, L, O = (sum(x[i] for x in tot) for i in range(3))
letters = max(1, T + L + O)
def frac(cond): return round(sum(1 for x in tot if cond(x)) / len(tot), 3)
out.update({
    "text_field": field, "sampled": len(texts),
    "cells_truncated_by_api": truncated,  # if > 0, words_mean and est_total_words are underestimates
    "words_mean": round(statistics.mean(words), 1), "words_median": statistics.median(words),
    "est_total_words_millions": round(statistics.mean(words) * out["num_rows"] / 1e6, 1) if out["num_rows"] else None,
    "tamil_letter_share": round(T / letters, 4), "latin_letter_share": round(L / letters, 4), "other_script_share": round(O / letters, 4),
    "docs_ge95pct_tamil": frac(lambda x: x[0] >= 0.95 * max(1, sum(x))),
    "docs_ge80pct_tamil": frac(lambda x: x[0] >= 0.80 * max(1, sum(x))),
    "docs_with_no_tamil": frac(lambda x: x[0] == 0),
    "exact_duplicates_in_sample": round(1 - len(set(texts)) / len(texts), 3),
    "example": texts[len(texts) // 2][:300],
})
print(json.dumps(out, ensure_ascii=False, indent=1))
