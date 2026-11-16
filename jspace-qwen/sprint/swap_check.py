"""M4: causal swap check for a set of lenses at a fixed set of layers.

Flexible-generalization trials ("f(A)" prompts, arg A -> arg B; success = f(B) at word rank 1),
clean-pass-correct trials only, clamp swap in the plane of the two lens directions (A J_l rows,
unit-normalised) at every edited layer and every position but 0 (jspace.interventions.swap).
Matched null: the same edit toward an unrelated word (success should stay ~0). The logit lens
(no Jacobian transport) is always included as the direction control.

python sprint/swap_check.py --maps sprint/results/maps_baseline.pt --layers 4 7 11 14 --tag base_e+m
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from jspace import evals as E  # noqa: E402
from jspace.interventions import _min_rank, clean_pass, swap  # noqa: E402
from jspace.lens import Lens  # noqa: E402
from jspace.model import load_model  # noqa: E402
from swaps import boot_ci, flexgen_trials  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--maps", required=True, help=".pt: {lens_name: {layer: [d,d] map}}")
ap.add_argument("--layers", type=int, nargs="+", required=True)
ap.add_argument("--scales", type=float, nargs="+", default=[1.0, 2.0])
ap.add_argument("--tag", required=True)
ap.add_argument("--threads", type=int, default=4)
a = ap.parse_args()
torch.set_num_threads(a.threads)

model = load_model("Qwen/Qwen3-0.6B")
tok = model.tokenizer
maps = torch.load(a.maps, weights_only=True)
lenses = {name: Lens("jlens", {l: m[l].float() for l in a.layers}, name=name) for name, m in maps.items()}
lenses["logit"] = Lens("logit")
word_mask = E.non_semantic_ids(tok, model.vocab_size)
rows = []
clean_cache = {}
for tr in flexgen_trials(tok):
    if not tr["succ"] or not tr["base"]:
        continue
    s_id, t_id = E.resolve_token(tok, tr["src"]), E.resolve_token(tok, tr["tgt"])
    if s_id is None or t_id is None:
        continue
    ids = model.encode(tr["prompt"])
    if tr["prompt"] not in clean_cache:
        clean_cache[tr["prompt"]] = clean_pass(model, ids, a.layers)
    cl = clean_cache[tr["prompt"]]
    if _min_rank(cl[0], tr["base"], word_mask) != 1:
        continue
    for lname, lens in lenses.items():
        for sc in a.scales:
            r = swap(model, ids, lens, a.layers, s_id, t_id, tr["succ"], tr["base"], scale=sc, clean=cl,
                     word_mask=word_mask)
            rows.append(dict(name=tr["name"], variant=tr["variant"], lens=lname, scale=sc,
                             success=int(r.success_rank == 1), lost=int(r.baseline_rank > 1),
                             success_rank=r.success_rank, edit_norm=r.edit_norm_ratio, top=r.top))
summary = {"layers": a.layers, "scales": a.scales, "results": {}}
for variant in ("main", "random_null"):
    for lname in lenses:
        for sc in a.scales:
            sel = [r for r in rows if r["variant"] == variant and r["lens"] == lname and r["scale"] == sc]
            if not sel:
                continue
            summary["results"][f"{variant}|{lname}|{sc}"] = {
                "n": len(sel), "success": boot_ci([r["success"] for r in sel]),
                "baseline_lost": boot_ci([r["lost"] for r in sel]),
                "median_edit_norm": float(np.median([r["edit_norm"] for r in sel]))}
# best scale per lens (main), with its null at the same scale
best = {}
for lname in lenses:
    cands = [(summary["results"][f"main|{lname}|{sc}"]["success"][0], sc) for sc in a.scales
             if f"main|{lname}|{sc}" in summary["results"]]
    if cands:
        s, sc = max(cands)
        best[lname] = {"scale": sc, "success": summary["results"][f"main|{lname}|{sc}"]["success"],
                       "null": summary["results"].get(f"random_null|{lname}|{sc}", {}).get("success")}
summary["best_scale"] = best
for k, v in summary["results"].items():
    print(f"{k:<32} n={v['n']:>3} success={v['success'][0]:.3f} [{v['success'][1]:.3f},{v['success'][2]:.3f}] "
          f"edit={v['median_edit_norm']:.2f}")
out = os.path.join(ROOT, "sprint", "results")
os.makedirs(out, exist_ok=True)
with open(os.path.join(out, f"swap_{a.tag}.json"), "w") as f:
    json.dump(summary, f, indent=1)
with open(os.path.join(out, f"swap_{a.tag}_rows.json"), "w") as f:
    json.dump(rows, f)
