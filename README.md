# Nulliglot

Instruction-tuning strictly monolingual language models. The first model is Bulgarian
(`iamshnoo/bul_100m`); German (`iamshnoo/deu_100m`) is planned next. Base model, adapters and
training data stay in the single target language.

A separate project, a Tamil base model trained from scratch, is in `tamil_base_model/`; see its
own README.

## Status (18 November 2026)

- The tuned model stops after its answer, writes only Bulgarian, and can copy a text from its
  prompt (word overlap 0.89, from 0.06).
- Its answers to open questions follow the question's vocabulary but are not correct, and
  Bulgarian MMLU and ARC are at chance (25%).
- Next step under discussion: continued Bulgarian-only pretraining of the base model.

## Layout

| Path | Content |
|---|---|
| `train_lora.py` | training (LoRA or full fine-tune, stop token, template embeddings) |
| `instruct_prompt.py`, `stop_token.py` | prompt template and stop-token handling shared by all scripts |
| `generate.py` | inference with an adapter and the decoding rules |
| `eval_*.py` | test-set, reading-task, decoding and MMLU/ARC evaluations |
| `make_*.py`, `split_data.py`, `prepare_benchmarks.py` | build the data sets for each stage |
| `make_plots.py`, `make_status_plots.py` | figures in `plots/` (first runs) and `plots_current_status/` (curriculum) |
| `agents.py`, `dialogue_sim.py`, `ui.py`, `chat_bul_100m.py` | earlier base-model dialogue experiments |
| `data/` | training, validation and test sets for every stage; `data/HANDOFF.md` describes the cleaning |
| `adapters/<run>/` | per run: settings, training log, samples, evaluation results; weights are release assets |
| `logs/` | full training and evaluation logs from the laptop |
| `hopper.sh`, `hopper_jobs/`, `hopper_activity.log` | Slurm jobs and the record of every command run on the Hopper cluster |
| `hopper/` | files copied back from the Hopper cluster: Slurm job logs, results of the runs made there, and the scripts as they ran |

## Weights and large data files

Files over GitHub's size limit are attached to the release `artifacts-2026-11-23` instead of
being stored in git: every `*.safetensors` file and the seven largest data files (gzipped).
`artifacts/MANIFEST.tsv` lists each one with its path, size and SHA-256.

```
./fetch_artifacts.sh              # restore everything, about 12 GB
./fetch_artifacts.sh phase3_lora  # restore one run
```

The two adapters in use are `adapters/phase2b_lora` (tasks on a given text) and
`adapters/phase3_lora` (open questions). The base model is not stored here; it downloads from
Hugging Face as `iamshnoo/bul_100m`.

## Setup

```
python3 -m venv .venv
.venv/bin/pip install torch transformers peft datasets matplotlib
```

Package versions are not pinned in this repository. The Hopper environment used
torch 2.15.0, transformers 5.22.2, peft 0.21.2 and accelerate 1.15.0.

## Data source and licence

The instruction data is `pinzhenchen/alpaca-cleaned-bg`, a machine translation of
`yahma/alpaca-cleaned`, licensed CC BY-NC 4.0 (non-commercial). Keep that in mind before
making this repository public.
