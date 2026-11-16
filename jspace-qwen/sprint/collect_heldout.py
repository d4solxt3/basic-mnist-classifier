"""Held-out WikiText activations for the transport-alignment metric (M3) and the generic
false-positive control (M2b). Records 7000-7127 were never used for fitting lenses, routers,
the regression lens or any earlier analysis (those used records 0-511 and 5000-5255).

python sprint/collect_heldout.py
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jspace import paths as P  # noqa: E402
from jspace.corpus import collect_acts, tokenize, wikitext_records  # noqa: E402
from jspace.model import load_model  # noqa: E402

torch.set_num_threads(2)
OFFSET, N, PER = 7000, 128, 32
LAYERS = [4, 7, 11, 14, 18, 21, 24]
model = load_model("Qwen/Qwen3-0.6B")
texts = wikitext_records(max(8000, OFFSET + N + 64), cache=P.CORPUS)[OFFSET:OFFSET + N + 64]
prompts = tokenize(model, texts)[:N]
acts, target = collect_acts(model, prompts, LAYERS, per_prompt=PER, seed=7000)
torch.save({"acts": {l: v.half() for l, v in acts.items()}, "target": target.half(), "records": [OFFSET, OFFSET + N],
            "per_prompt": PER, "prompt_of_row": torch.arange(len(prompts)).repeat_interleave(PER)[:target.shape[0]]},
           os.path.join(P.LENSES, "heldout_wiki_acts.pt"))
print({l: tuple(v.shape) for l, v in acts.items()}, tuple(target.shape))
