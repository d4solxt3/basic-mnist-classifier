"""Fitting corpus: WikiText-103 (raw) records of >= 600 characters, as in the
reference implementation (``jlens.examples.load_wikitext_prompts``), tokenised
and truncated to 128 tokens."""

from __future__ import annotations

import json
import os

import torch

from jspace.model import LensModel, Recorder


def wikitext_records(n: int, *, min_chars: int = 600, offset: int = 0, cache: str | None = None) -> list[str]:
    if cache and os.path.exists(cache):
        with open(cache) as f:
            recs = json.load(f)
        if len(recs) >= offset + n or os.environ.get("JSPACE_CORPUS"):
            return recs[offset:offset + n]
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    recs = []
    for r in ds:
        t = r["text"]
        if len(t.strip()) >= min_chars:
            recs.append(t)
            if len(recs) >= offset + n:
                break
    if cache:
        with open(cache, "w") as f:
            json.dump(recs, f)
    return recs[offset:offset + n]


def tokenize(model: LensModel, texts: list[str], max_len: int = 128, min_len: int = 64) -> list[torch.Tensor]:
    out = []
    for t in texts:
        ids = model.encode(t, max_length=max_len)
        if ids.shape[1] >= min_len:
            out.append(ids)
    return out


@torch.no_grad()
def collect_acts(model: LensModel, prompts: list[torch.Tensor], layers: list[int], *, skip_first: int = 16,
                 per_prompt: int = 32, seed: int = 0) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    """Residuals at ``per_prompt`` random positions (>= skip_first) of each
    prompt, at ``layers`` and at the final layer."""
    g = torch.Generator().manual_seed(seed)
    final = model.n_layers - 1
    rec_layers = sorted(set(layers) | {final})
    acts = {l: [] for l in rec_layers}
    for ids in prompts:
        T = ids.shape[1]
        pos = skip_first + torch.randperm(T - 1 - skip_first, generator=g)[:per_prompt]
        with Recorder(model, rec_layers) as rec:
            model.forward(ids)
        for l in rec_layers:
            acts[l].append(rec.acts[l][0, pos].float())
    acts = {l: torch.cat(v) for l, v in acts.items()}
    target = acts.pop(final) if final not in layers else acts[final]
    return acts, target
