"""Fit a Jacobian lens (exact reference estimator, or a sketch estimator).

Examples
  python scripts/fit.py --method exact  --n 64  --snapshots 1 2 4 8 16 32 64 --tag exact
  python scripts/fit.py --method sketch --K 16 --cv online --n 2048 --tag sk16 \
         --snapshots 32 64 128 256 512 1024 2048
  python scripts/fit.py --method sketch --K 16 --cv online --shaped --n 2048 --tag sk16w
"""

import argparse
import json
import os
import sys

import torch

ROOT = __file__.rsplit("/scripts/", 1)[0]
sys.path.insert(0, ROOT)
from jspace.corpus import tokenize, wikitext_records  # noqa: E402
from jspace.fitting import fit, readout_shaping, save_lens_state  # noqa: E402
from jspace.model import load_model  # noqa: E402
from jspace import paths as P  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
ap.add_argument("--method", choices=["exact", "sketch"], default="exact")
ap.add_argument("--n", type=int, default=64)
ap.add_argument("--offset", type=int, default=0, help="first corpus record to use")
ap.add_argument("--K", type=int, default=16)
ap.add_argument("--cv", default="online", choices=["none", "identity", "online"])
ap.add_argument("--shaped", action="store_true", help="readout-aware probe shaping W=(A^T A)^(1/4)")
ap.add_argument("--dim-batch", type=int, default=16)
ap.add_argument("--seq", type=int, default=128)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--snapshots", type=int, nargs="*", default=[])
ap.add_argument("--per-prompt-layers", type=int, nargs="*", default=[])
ap.add_argument("--tag", required=True)
ap.add_argument("--out", default=P.LENSES)
ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()

torch.set_num_threads(a.threads)
os.makedirs(a.out, exist_ok=True)
model = load_model(a.model)
cache = P.CORPUS
texts = wikitext_records(max(6000, a.offset + a.n + 64), cache=cache)[a.offset:a.offset + a.n + 64]
prompts = tokenize(model, texts, max_len=a.seq)[:a.n]
print(f"{len(prompts)} prompts, mean len {sum(p.shape[1] for p in prompts) / len(prompts):.1f}")

W = Winv = None
if a.shaped:
    W, Winv = readout_shaping(model.unembed_matrix())
st = fit(model, prompts, method=a.method, K=a.K, cv=a.cv, W=W, Winv=Winv, dim_batch=a.dim_batch,
         seed=a.seed, snapshots=a.snapshots, snapshot_dir=a.out, tag=a.tag,
         save_per_prompt_layers=a.per_prompt_layers,
         per_prompt_path=os.path.join(a.out, f"{a.tag}_per_prompt.pt") if a.per_prompt_layers else None)
save_lens_state(st, os.path.join(a.out, f"{a.tag}_final.pt"))
with open(os.path.join(a.out, f"{a.tag}_log.json"), "w") as f:
    json.dump({"args": vars(a), "seconds": st.seconds, "vjp_rows": st.vjp_rows, "tokens_bwd": st.tokens_bwd,
               "log": st.log}, f)
print(f"done: {st.n} prompts, {st.vjp_rows} VJP rows, {st.seconds / 60:.1f} min")
