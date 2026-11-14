"""Readout evaluation: recall@k of latent intermediates on the six lens evals.

python scripts/readout_eval.py --lens jlens=lenses/exact_final.pt --calib --tag main
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = __file__.rsplit("/scripts/", 1)[0]
sys.path.insert(0, ROOT)
from jspace import evals as E  # noqa: E402
from jspace.lens import Calibration, Lens, fit_calibration  # noqa: E402
from jspace.model import load_model  # noqa: E402
from jspace import paths as P  # noqa: E402


def get_prep(model, path):
    if os.path.exists(path):
        return torch.load(path, weights_only=False)
    items = E.load_items(os.path.join(ROOT, "data", "evaluations"))
    prep = E.prepare(model, items)
    torch.save(prep, path)
    return prep


def get_calib(model, lens, name, out_dir, acts=None):
    p = os.path.join(out_dir, f"calib_{name}.pt")
    if os.path.exists(p):
        return Calibration.load(p)
    if acts is None:
        s = torch.load(os.path.join(out_dir, "corpus_acts.pt"), weights_only=True)
        acts = {int(l): v.float() for l, v in s["acts"].items()}
    layers = [l for l in acts if lens.has_layer(l)]
    cal = fit_calibration(model, lens, {l: acts[l] for l in layers})
    cal.save(p)
    return cal


def summarize(prep, ranks, layers, band, ks=(1, 10)):
    cols_all = list(range(len(layers)))
    cols_band = [layers.index(l) for l in band if l in layers]
    out = {}
    for k in ks:
        out[f"all@{k}"] = E.recall_at_k(prep, ranks, cols_all, k)
        out[f"band@{k}"] = E.recall_at_k(prep, ranks, cols_band, k)
        out[f"orderops_all@{k}"] = E.recall_at_k(prep, ranks, cols_all, k, evals=("order-ops",))
        out[f"per_layer@{k}"] = {int(l): E.recall_at_k(prep, ranks, [j], k, n_boot=200)["macro"]["mean"]
                                 for j, l in enumerate(layers)}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--lens", nargs="*", default=[], help="name=path (jlens snapshot or Lens file)")
    ap.add_argument("--no-logit", action="store_true")
    ap.add_argument("--regress", action="store_true")
    ap.add_argument("--calib", action="store_true")
    ap.add_argument("--filter", action="store_true", help="also score with J++ Readout Filtering")
    ap.add_argument("--band", type=int, nargs=2, default=None)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    model = load_model(a.model)
    L = model.n_layers
    lens_dir = P.LENSES
    res_dir = P.RESULTS
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(lens_dir, exist_ok=True)
    prep = get_prep(model, os.path.join(lens_dir, "readout_prep.pt"))
    n_correct = {ev: [int(sum(1 for it, c in zip(prep.items, prep.correct) if it.eval == ev and c)),
                      int(sum(1 for it in prep.items if it.eval == ev))] for ev in E.SPECS}
    print("items (correct/total):", n_correct)

    lenses = {}
    if not a.no_logit:
        lenses["logit"] = Lens("logit")
    if a.regress:
        lenses["regress"] = Lens.load(os.path.join(lens_dir, "regress.pt"))
    for spec in a.lens:
        name, path = spec.split("=", 1)
        lenses[name] = Lens.load(P.resolve(path))

    layers = list(range(L - 1))
    band = list(range(a.band[0], a.band[1] + 1)) if a.band else list(range(L // 4, 3 * L // 4))
    exclude = E.non_semantic_ids(model.tokenizer, model.vocab_size) if a.filter else None
    acts = None
    if a.calib:
        s = torch.load(os.path.join(lens_dir, "corpus_acts.pt"), weights_only=True)
        acts = {int(l): v.float() for l, v in s["acts"].items()}

    summary = {"n_items": n_correct, "band": band, "layers": layers, "results": {}}
    all_ranks = {}
    for name, lens in lenses.items():
        cal = get_calib(model, lens, f"{a.tag}_{name}", lens_dir, acts) if a.calib else None
        variants = [("raw", None)] + ([("z", None)] if a.calib else [])
        if a.filter:
            variants += [("raw", "filter")] + ([("z", "filter")] if a.calib else [])
        for mode, filt in variants:
            key = f"{name}|{mode}" + ("|filter" if filt else "")
            r = E.lens_ranks(model, prep, lens, layers, calib=cal, mode=mode,
                             exclude=exclude if filt else None)
            all_ranks[key] = r
            summary["results"][key] = summarize(prep, r, layers, band)
            m = summary["results"][key]
            print(f"{key:<28} all@1={m['all@1']['macro']['mean']:.3f} all@10={m['all@10']['macro']['mean']:.3f} "
                  f"[{m['all@10']['macro']['lo']:.3f},{m['all@10']['macro']['hi']:.3f}] "
                  f"band@10={m['band@10']['macro']['mean']:.3f}", flush=True)
    np.savez_compressed(os.path.join(res_dir, f"readout_ranks_{a.tag}.npz"), **{k.replace("|", "__"): v
                                                                                 for k, v in all_ranks.items()})
    with open(os.path.join(res_dir, f"readout_{a.tag}.json"), "w") as f:
        json.dump(summary, f, indent=1)
