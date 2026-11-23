"""Small trainable extras beside the LoRA adapter.

--train-stop-token         output row for <|end_of_text|>      (StopTokenRow)
--train-marker-embeddings  input embeddings of template tokens (MarkerEmbeddings)

Stop token
----------

The base models never saw <|end_of_text|> in pretraining, so its lm_head row is
still at initialisation and LoRA alone can only bend hidden states towards that
random vector. This module learns a correction `delta` (one vector of
hidden_size floats) so the stop-token logit becomes h @ (w_eos + delta). The
base weights on disk are never modified.

Marker embeddings
-----------------
Template tokens such as "###" and "\n\n" were also never seen in pretraining,
so their input embeddings are untrained noise. MarkerEmbeddings learns a
correction per template token id, added to the embedding output wherever that
token occurs.

Inference:  load_instruct_model(model_id, dir) -> base + LoRA + whichever
            extras the run saved. Plain PeftModel.from_pretrained skips them.
"""

from pathlib import Path

import torch
from torch import nn

FILENAME = "stop_token.pt"
MARKER_FILENAME = "marker_embeddings.pt"


class StopTokenRow(nn.Module):
    def __init__(self, model, eos_id):
        super().__init__()
        head = model.get_output_embeddings()
        self.eos_id = eos_id
        self.delta = nn.Parameter(torch.zeros(head.in_features))
        self._hook = head.register_forward_hook(self._add)

    def _add(self, module, inputs, logits):
        extra = inputs[0].to(self.delta.dtype) @ self.delta
        logits[..., self.eos_id] = logits[..., self.eos_id] + extra.to(logits.dtype)
        return logits

    def save(self, directory):
        torch.save({"eos_id": self.eos_id, "delta": self.delta.detach().cpu()},
                   Path(directory) / FILENAME)


class MarkerEmbeddings(nn.Module):
    def __init__(self, model, token_ids):
        super().__init__()
        emb = model.get_input_embeddings()
        self.token_ids = sorted(set(token_ids))
        # slot[token] = 1 + row in delta for marker tokens, 0 for everything else
        slot = torch.zeros(emb.num_embeddings, dtype=torch.long)
        slot[torch.tensor(self.token_ids)] = torch.arange(1, len(self.token_ids) + 1)
        self.register_buffer("slot", slot, persistent=False)
        self.delta = nn.Parameter(torch.zeros(len(self.token_ids), emb.embedding_dim))
        self._hook = emb.register_forward_hook(self._add)

    def _add(self, module, inputs, out):
        table = torch.cat([self.delta.new_zeros(1, self.delta.shape[1]), self.delta])
        return out + table[self.slot[inputs[0]]].to(out.dtype)

    def save(self, directory):
        torch.save({"token_ids": self.token_ids, "delta": self.delta.detach().cpu()},
                   Path(directory) / MARKER_FILENAME)


def merge_marker_embeddings(model, directory):
    """Add saved marker deltas into the in-memory embedding rows."""
    path = Path(directory) / MARKER_FILENAME
    if not path.exists():
        return False
    saved = torch.load(path, map_location="cpu")
    weight = model.get_input_embeddings().weight
    with torch.no_grad():
        weight[torch.tensor(saved["token_ids"])] += saved["delta"].to(weight.device, weight.dtype)
    return True


def merge_stop_token(model, directory):
    """Add a saved delta into the in-memory lm_head row. Returns True if found."""
    path = Path(directory) / FILENAME
    if not path.exists():
        return False
    saved = torch.load(path, map_location="cpu")
    weight = model.get_output_embeddings().weight
    with torch.no_grad():
        weight[saved["eos_id"]] += saved["delta"].to(weight.device, weight.dtype)
    return True


def load_instruct_model(model_id, adapter_dir, dtype=torch.float32):
    """Load the result of a train_lora.py run.

    LoRA run: base model + adapter, plus any extras the run saved.
    --full-finetune run: adapter_dir holds a complete model; model_id is unused.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    if not (Path(adapter_dir) / "adapter_config.json").exists():
        return AutoModelForCausalLM.from_pretrained(adapter_dir, dtype=dtype)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    merge_stop_token(model, adapter_dir)
    merge_marker_embeddings(model, adapter_dir)
    return PeftModel.from_pretrained(model, adapter_dir)
