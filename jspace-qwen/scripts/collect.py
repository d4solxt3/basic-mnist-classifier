"""Collect held-out corpus residuals; fit the regression-lens baseline.

python scripts/collect.py --n 256 --offset 5000
"""

import argparse
import os
import sys

import torch

ROOT = __file__.rsplit("/scripts/", 1)[0]
sys.path.insert(0, ROOT)
from jspace.corpus import collect_acts, tokenize, wikitext_records  # noqa: E402
from jspace.lens import fit_regression_lens  # noqa: E402
from jspace.model import load_model  # noqa: E402
from jspace import paths as P  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
ap.add_argument("--n", type=int, default=256)
ap.add_argument("--offset", type=int, default=5000, help="held out from fitting prompts")
ap.add_argument("--per-prompt", type=int, default=16)
ap.add_argument("--out", default=P.LENSES)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
model = load_model(a.model)
cache = P.CORPUS
texts = wikitext_records(max(6000, a.offset + a.n + 64), cache=cache)[a.offset:a.offset + a.n + 64]
prompts = tokenize(model, texts)[:a.n]
layers = list(range(model.n_layers - 1))
acts, target = collect_acts(model, prompts, layers, per_prompt=a.per_prompt)
print({l: tuple(v.shape) for l, v in list(acts.items())[:2]}, tuple(target.shape))
torch.save({"acts": {l: v.half() for l, v in acts.items()}, "target": target.half()},
           os.path.join(a.out, "corpus_acts.pt"))
# Regression lens on the first 3/4 of positions; the rest stays available as a check.
n_fit = int(0.75 * target.shape[0])
reg = fit_regression_lens({l: v[:n_fit] for l, v in acts.items()}, target[:n_fit])
reg.save(os.path.join(a.out, "regress.pt"))
for l in sorted(acts)[:: max(1, len(acts) // 5)]:
    X, Y = acts[l][n_fit:], target[n_fit:]
    pred = reg.transport(X, l)
    r2 = 1 - ((pred - Y).pow(2).sum() / (Y - Y.mean(0)).pow(2).sum())
    print(f"regression lens layer {l}: held-out R^2 = {float(r2):.3f}")
