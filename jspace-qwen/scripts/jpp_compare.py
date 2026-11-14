"""Head-to-head against the J++ Lens protocol on Qwen3-0.6B.

Protocol (J++, Ayonrinde & Lindsey 2026): five macro readout evals, items the
model answers correctly, stratified 20% fitting / 80% held-out split (seed 0);
expert weights fitted on the fitting split only; recall@k = best rank over
seven evenly spaced layers, scored on the held-out split; Readout Filtering
(tokens without a letter/digit removed) for J++.

Lenses (all fitted on the same 32 WikiText-103 prompts):
  logit, regress            baselines
  J-lens                    exact reference estimator (plain gradients)
  R-lens                    LRP backward, single Jacobian
  J-filt                    plain-gradient experts + learned weights
  J++                       LRP experts + learned weights          <- the bar
  Routed / Routed-LRP       ours: hard-routed experts (unsupervised)
Readouts: raw or z (per-token calibration on held-out corpus positions, ours),
each with or without Readout Filtering.

python scripts/jpp_compare.py --tag main
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
from jspace import paths as P  # noqa: E402
from jspace.jpp import RoutedLens, combine, expert_logits, fit_expert_weights, split_items  # noqa: E402
from jspace.lens import Calibration, Lens, fit_calibration  # noqa: E402
from jspace.model import load_model  # noqa: E402
from jspace.routing import Router  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
ap.add_argument("--plain", default="exact")
ap.add_argument("--lrp", default="lrp")
ap.add_argument("--layers", type=int, nargs="+", default=[4, 7, 11, 14, 18, 21, 24])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tag", required=True)
ap.add_argument("--all-items", action="store_true", help="secondary analysis: no correctness filter")
ap.add_argument("--ss-positions", type=int, default=200)
ap.add_argument("--ss-steps", type=int, default=100)
ap.add_argument("--calib", action="store_true", help="also centred / z-scored readouts")
a = ap.parse_args()

torch.manual_seed(0)
model = load_model(a.model)
prep = torch.load(os.path.join(P.LENSES, "readout_prep.pt"), weights_only=False)
if a.all_items:
    prep.correct = [None] * len(prep.correct)
keep = E.scored_item_mask(prep)
fit_idx, held_idx = split_items(prep.items, keep, seed=a.seed)
held = set(held_idx)
print(f"items: {int(keep.sum())} scored, {len(fit_idx)} fitting, {len(held_idx)} held-out")
rows = E.row_index(prep)
exclude = E.non_semantic_ids(model.tokenizer, model.vocab_size)
router = Router.load(os.path.join(P.LENSES, "router_E8.pt"))
acts_s = torch.load(os.path.join(P.LENSES, "corpus_acts.pt"), weights_only=True)
acts = {int(l): v.float() for l, v in acts_s["acts"].items() if int(l) in a.layers}
layers = a.layers
fp = os.path.join(P.LENSES, "future_targets.pt")
fut = torch.load(fp, weights_only=False) if os.path.exists(fp) else None

lenses: dict[str, Lens] = {"logit": Lens("logit"), "regress": Lens.load(os.path.join(P.LENSES, "regress.pt"))}
weights_log = {}
for fam, tag in (("plain", a.plain), ("lrp", a.lrp)):
    jp = os.path.join(P.LENSES, f"{tag}_final.pt")
    ep = os.path.join(P.LENSES, f"{tag}_experts_final.pt")
    if not os.path.exists(jp) and "_n" in tag:  # a snapshot, e.g. exact_n8
        jp = os.path.join(P.LENSES, f"{tag}.pt")
        ep = os.path.join(P.LENSES, tag.replace("_n", "_experts_n") + ".pt")
    if not os.path.exists(jp):
        print(f"[skip] {fam}: {jp} not found")
        continue
    single = Lens.load(jp)
    lenses["J-lens" if fam == "plain" else "R-lens"] = single
    if not os.path.exists(ep):
        continue
    ex = torch.load(ep, weights_only=True)
    experts = {int(l): v.float() for l, v in ex["experts"].items() if int(l) in layers}
    # J++ expert weights, fitted on the fitting split only (with Readout Filtering).
    fit_rows = [(r, i, w) for r, (i, w) in enumerate(rows) if i in set(fit_idx)]
    items_fit = sorted({i for _, i, _ in fit_rows})
    pos_of = {i: j for j, i in enumerate(items_fit)}
    combined = {}
    for l in layers:
        Z = expert_logits(model, experts[l], prep.resid[items_fit, l])
        w, sc = fit_expert_weights(Z, [pos_of[i] for _, i, _ in fit_rows],
                                   [prep.cands[i][wd] for _, i, wd in fit_rows],
                                   [prep.items[i].eval for _, i, _ in fit_rows], exclude=exclude)
        combined[l] = combine(experts[l], w)
        weights_log[f"{fam}|{l}"] = {"w": w.tolist(), "train_macro": sc}
        print(f"  {fam} layer {l}: weights {np.round(w.numpy(), 2).tolist()} train macro@10 {sc:.3f}", flush=True)
        del Z
    lenses["J-filt" if fam == "plain" else "J++"] = Lens("jlens", combined, name="weighted")
    # Self-supervised Jacobian filtering (ours): same objective and search, but the
    # "items" are unlabelled corpus positions and the candidates are the semantic
    # tokens the text actually goes on to produce in the next 16 positions.
    if fut is not None:
        ss = {}
        npos = min(a.ss_positions, len(fut["future"]))
        for l in layers:
            Z = expert_logits(model, experts[l], fut["H"][l][:npos])
            w, sc = fit_expert_weights(Z, list(range(npos)), fut["future"][:npos], ["corpus"] * npos,
                                       exclude=exclude, steps=a.ss_steps)
            ss[l] = combine(experts[l], w)
            weights_log[f"ss-{fam}|{l}"] = {"w": w.tolist(), "train_future_recall": sc}
            print(f"  SS {fam} layer {l}: weights {np.round(w.numpy(), 2).tolist()} future recall@10 {sc:.3f}",
                  flush=True)
            del Z
        lenses["SS-J-filt" if fam == "plain" else "SS-J++"] = Lens("jlens", ss, name="ss-weighted")
    pooled = {l: (experts[l] * ex["counts"][l][:, None, None] / ex["counts"][l].sum()).sum(0) for l in layers}
    base = "Routed" if fam == "plain" else "Routed-LRP"
    lenses[base] = RoutedLens(experts, router)
    # Shrinkage toward the pooled Jacobian (lambda selected on the fitting split).
    lenses[base + "-s0.5"] = RoutedLens({l: 0.5 * experts[l] + 0.5 * pooled[l][None] for l in layers}, router)
    lenses["Pooled" if fam == "plain" else "Pooled-LRP"] = Lens("jlens", pooled)

results, ranks_all = {}, {}
cols = list(range(len(layers)))
for name, lens in lenses.items():
    cal = fit_calibration(model, lens, acts) if a.calib else None
    for mode in (("raw", "center", "z") if a.calib else ("raw",)):
        for filt in (False, True):
            key = f"{name}|{mode}" + ("|filter" if filt else "")
            r = E.lens_ranks(model, prep, lens, layers, calib=cal, mode=mode, exclude=exclude if filt else None)
            ranks_all[key] = r
            res = {}
            for k in (1, 10):
                for wt in ("item", "pair"):
                    res[f"{wt}@{k}"] = E.recall_at_k(prep, r, cols, k, items_subset=held, weighting=wt)
            res["fit_item@10"] = E.recall_at_k(prep, r, cols, 10, items_subset=set(fit_idx), n_boot=10)["macro"]["mean"]
            results[key] = res
            m = res["item@10"]["macro"]
            print(f"{key:<26} item@10={m['mean']:.3f} [{m['lo']:.3f},{m['hi']:.3f}] "
                  f"pair@10={res['pair@10']['macro']['mean']:.3f} item@1={res['item@1']['macro']['mean']:.3f}",
                  flush=True)

# Model selection on the FITTING split only: for each lens, the readout variant
# (raw / center / z, +- filter) with the best fitting-split item recall@10.
selected = {}
families = {}
for name in lenses:
    families.setdefault(name.split("-s")[0], []).append(name)
for fam_name, members in families.items():
    name = fam_name
    cands = [k for k in results if k.split("|")[0] in members]
    best = max(cands, key=lambda k: (results[k]["fit_item@10"], k.endswith("raw|filter")))
    selected[name] = best
    m = results[best]["item@10"]["macro"]
    print(f"[selected on fit split] {name:<12} -> {best:<26} held-out item@10={m['mean']:.3f} [{m['lo']:.3f},{m['hi']:.3f}]")
ref = "J++|raw|filter" if "J++|raw|filter" in ranks_all else None
diffs = {}
if ref:
    for key, r in ranks_all.items():
        diffs[key] = E.paired_bootstrap_diff(prep, r, ranks_all[ref], cols, cols, 10, items_subset=held)
os.makedirs(P.RESULTS, exist_ok=True)
with open(os.path.join(P.RESULTS, f"jpp_compare_{a.tag}.json"), "w") as f:
    json.dump({"layers": layers, "n_fit": len(fit_idx), "n_held": len(held_idx), "results": results,
               "diff_vs_jpp": diffs, "weights": weights_log, "selected": selected}, f, indent=1)
np.savez_compressed(os.path.join(P.RESULTS, f"jpp_compare_ranks_{a.tag}.npz"),
                    **{k.replace("|", "__"): v for k, v in ranks_all.items()})
if ref:
    print("\npaired difference in held-out item recall@10 vs J++ (raw|filter):")
    for key, d in sorted(diffs.items(), key=lambda kv: -kv[1]["diff"]):
        print(f"  {key:<26} {d['diff']:+.3f} [{d['lo']:+.3f},{d['hi']:+.3f}]  P(<=0)={d['p_le_0']:.3f}")
