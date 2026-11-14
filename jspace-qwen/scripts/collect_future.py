"""Unlabelled training signal for self-supervised Jacobian filtering (ours).

For held-out corpus positions p (>= 16, with >= W tokens of future), record the
residual at the J++ layers and the set of *semantic* future token ids
t_{p+1..p+W} (tokens with a letter or digit). A good workspace lens should rank
the tokens the model will actually go on to say: that is the J-lens's own
definition (sensitivity summed over all future target positions).

python scripts/collect_future.py --n 200 --offset 5300 --per-prompt 4 --window 16
"""

import argparse
import os
import sys

import torch

ROOT = __file__.rsplit("/scripts/", 1)[0]
sys.path.insert(0, ROOT)
from jspace import evals as E  # noqa: E402
from jspace import paths as P  # noqa: E402
from jspace.corpus import tokenize, wikitext_records  # noqa: E402
from jspace.model import Recorder, load_model  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--offset", type=int, default=5300)
ap.add_argument("--per-prompt", type=int, default=4)
ap.add_argument("--window", type=int, default=16)
ap.add_argument("--layers", type=int, nargs="+", default=[4, 7, 11, 14, 18, 21, 24])
a = ap.parse_args()
m = load_model(a.model)
nonsem = E.non_semantic_ids(m.tokenizer, m.vocab_size)
texts = wikitext_records(6000, cache=P.CORPUS)[a.offset:a.offset + a.n + 32]
prompts = tokenize(m, texts)[:a.n]
g = torch.Generator().manual_seed(0)
H = {l: [] for l in a.layers}
fut, pos_all = [], []
for ids in prompts:
    T = ids.shape[1]
    hi = T - a.window
    if hi <= 16:
        continue
    pos = 16 + torch.randperm(hi - 16, generator=g)[:a.per_prompt]
    with torch.no_grad(), Recorder(m, a.layers) as rec:
        m.forward(ids)
    for p in pos.tolist():
        f = sorted({int(t) for t in ids[0, p + 1:p + 1 + a.window] if not nonsem[int(t)]})
        if not f:
            continue
        fut.append(f)
        pos_all.append(p)
        for l in a.layers:
            H[l].append(rec.acts[l][0, p].float())
out = {"H": {l: torch.stack(v) for l, v in H.items()}, "future": fut, "pos": pos_all, "window": a.window}
torch.save(out, os.path.join(P.LENSES, "future_targets.pt"))
print(f"{len(fut)} positions, mean {sum(map(len, fut)) / len(fut):.1f} semantic future tokens each")
