"""Learning curves: lens quality vs. compute (VJP rows) for the exact and
sketch estimators.

Primary metric: item recall@10 (macro over the five evals, best rank over the
seven J++ layers, raw readout), on (a) correctness-filtered items and (b) all
items. Secondary: relative error of the J-space dictionary A J at layers
7/14/21 against an *independent* reference: the exact lens on prompts 16..31,
J_ref = (32 J_32 - 16 J_16) / 16; estimators are compared on their own
prompts 0..15 equivalents where nested (exact n<=16).

python scripts/curves.py --tags exact sk16 sk16w sk16id sk16none
"""

import argparse
import glob
import json
import os
import re
import sys

import torch

ROOT = __file__.rsplit("/scripts/", 1)[0]
sys.path.insert(0, ROOT)
from jspace import evals as E  # noqa: E402
from jspace import paths as P  # noqa: E402
from jspace.fitting import load_lens_state  # noqa: E402
from jspace.lens import Lens  # noqa: E402
from jspace.model import load_model  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
ap.add_argument("--tags", nargs="+", default=["exact", "sk16", "sk16w", "sk16id", "sk16none"])
ap.add_argument("--layers", type=int, nargs="+", default=[4, 7, 11, 14, 18, 21, 24])
ap.add_argument("--err-layers", type=int, nargs="+", default=[7, 14, 21])
ap.add_argument("--out", default="curves")
a = ap.parse_args()

model = load_model(a.model)
A = model.unembed_matrix()
prep = torch.load(os.path.join(P.LENSES, "readout_prep.pt"), weights_only=False)
prep_all = torch.load(os.path.join(P.LENSES, "readout_prep.pt"), weights_only=False)
prep_all.correct = [None] * len(prep_all.correct)
cols = list(range(len(a.layers)))

ref = None
p16, p32 = os.path.join(P.LENSES, "exact_n16.pt"), os.path.join(P.LENSES, "exact_n32.pt")
if os.path.exists(p16) and os.path.exists(p32):
    J16, J32 = load_lens_state(p16)["J"], load_lens_state(p32)["J"]
    ref = {l: (32 * J32[l] - 16 * J16[l]) / 16 for l in a.err_layers}

out_path = os.path.join(P.RESULTS, f"{a.out}.json")
res = json.load(open(out_path)) if os.path.exists(out_path) else {}
for tag in a.tags:
    for path in sorted(glob.glob(os.path.join(P.LENSES, f"{tag}_n*.pt")),
                       key=lambda p: int(re.search(r"_n(\d+)\.pt$", p).group(1))):
        key = os.path.basename(path)[:-3]
        if key in res:
            continue
        s = load_lens_state(path)
        lens = Lens("jlens", {l: s["J"][l] for l in a.layers})
        r = E.lens_ranks(model, prep, lens, a.layers)
        row = {"tag": tag, "n": s["n"], "vjp_rows": s["vjp_rows"], "seconds": s["seconds"],
               "recall10_filtered": E.recall_at_k(prep, r, cols, 10, n_boot=500)["macro"],
               "recall10_all": E.recall_at_k(prep_all, r, cols, 10, n_boot=500)["macro"]}
        if ref is not None and not (tag == "exact" and s["n"] > 16):
            row["dict_err"] = {l: float((A @ s["J"][l] - A @ ref[l]).norm() / (A @ ref[l]).norm())
                               for l in a.err_layers}
        res[key] = row
        print(f"{key:<16} rows={s['vjp_rows']:>7} recall@10 filt={row['recall10_filtered']['mean']:.3f} "
              f"all={row['recall10_all']['mean']:.3f} err={row.get('dict_err', {})}", flush=True)
        with open(out_path, "w") as f:
            json.dump(res, f, indent=1)
