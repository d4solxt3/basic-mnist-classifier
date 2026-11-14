"""Round 2 of the J++ comparison: pre-registered candidates, evaluated over
several fitting/held-out splits (see paper/prereg_round2.md).

Round 1 (scripts/jpp_compare.py, split seed 0) has been looked at, so its
held-out numbers are no longer an unbiased test for anything designed after
it. Round 2 therefore re-runs the whole J++ protocol on fresh stratified splits
(seeds 1..5; seed 0 is reported but not used for the confirmatory summary),
refitting every label-dependent component (J++ weights and all candidates) on
each split's fitting items and scoring on that split's held-out items.

Unsupervised lenses (logit, J-lens, R-lens, B-lens, routed LRP experts) do not
depend on the split; their ranks are computed once.

All weight searches use ``fit_expert_weights_v2``, the faithful port of J++'s
search (masking after combining, full-vocabulary spread, (exact, surrogate)
tie-break); round 1 used an earlier port with three small deviations.

Candidates (all with Readout Filtering, best rank over the seven J++ layers):
  JppB            two-stage: J++ map, then per-layer signed weights over
                  [J++, B-lens] (normalised), fitted like J++ on the same items
  RB-mix          per-layer signed weights over [R-lens, B-lens]
  Routed-J++-a200 routed LRP experts shrunk toward the J++ map,
                  lambda_e = n_e / (n_e + 200) (kappa fixed in round 1)
  Bagged-J++      J++ weights averaged over bootstrap resamples (stratified by
                  eval) of the fitting items, then renormalised
Round-1 lenses re-evaluated: J++, R-lens, J-lens, logit, JB-mix,
Routed-LRP-a200, Routed-LRP-s0.5, B-lens.

python scripts/jpp_splits.py --seeds 0 1 2 3 4 5 --bags 8 --tag round2

Round 3 (paper/prereg_round3.md): 20 new splits, the two round-2 winners and
RLB-mix over [R-lens, LRP broadcast lens], plus out-of-pool scoring on the
items that fail the correctness filter:
python scripts/jpp_splits.py --seeds $(seq 6 25) --bags 8 --candidates Bagged-J++ RB-mix RLB-mix \
    --lrp-b lenses/lrp_later_final.pt --oop --tag round3
"""

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = __file__.rsplit("/scripts/", 1)[0]
sys.path.insert(0, ROOT)
from jspace import evals as E  # noqa: E402
from jspace import paths as P  # noqa: E402
from jspace.jpp import (RoutedLens, adaptive_shrink, combine, expert_logits, finalize_experts,  # noqa: E402
                        fit_expert_weights_v2, split_items)  # noqa: E402
from jspace.lens import Lens  # noqa: E402
from jspace.model import load_model  # noqa: E402
from jspace.routing import Router  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
ap.add_argument("--layers", type=int, nargs="+", default=[4, 7, 11, 14, 18, 21, 24])
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
ap.add_argument("--bags", type=int, default=8)
ap.add_argument("--steps", type=int, default=200, help="J++ Adam steps (reference: 200)")
ap.add_argument("--b-lens", default="lenses/later_final.pt")
CANDS = ["JppB", "RB-mix", "JB-mix", "Bagged-J++", "Routed-J++-a200", "RLB-mix"]
ap.add_argument("--candidates", nargs="+", choices=CANDS, default=CANDS[:5],
                help="label-dependent lenses to fit besides J++ (default: the round-2 set)")
ap.add_argument("--lrp-b", default=None, help="LRP broadcast lens (adds LRP-B-lens; needed for RLB-mix)")
ap.add_argument("--oop", action="store_true",
                help="also score the items that fail the correctness filter (out-of-pool, secondary)")
ap.add_argument("--min-win-frac", type=float, default=0.8, help="win criterion: wins >= frac * n splits")
ap.add_argument("--tag", required=True)
a = ap.parse_args()
assert "RLB-mix" not in a.candidates or a.lrp_b, "RLB-mix needs --lrp-b"

T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


torch.manual_seed(0)
model = load_model(a.model)
prep = torch.load(os.path.join(P.LENSES, "readout_prep.pt"), weights_only=False)
keep = E.scored_item_mask(prep)
rows = E.row_index(prep)
exclude = E.non_semantic_ids(model.tokenizer, model.vocab_size)
router = Router.load(os.path.join(P.LENSES, "router_E8.pt"))
layers = a.layers
cols = list(range(len(layers)))

J = Lens.load(os.path.join(P.LENSES, "exact_final.pt"))
Rl = Lens.load(os.path.join(P.LENSES, "lrp_final.pt"))
B = Lens.load(P.resolve(a.b_lens))
LB = Lens.load(P.resolve(a.lrp_b)) if a.lrp_b else None
if a.oop:
    prep_all = copy.copy(prep)
    prep_all.correct = [None] * len(prep.correct)  # score every item; subset = correctness failures
    oop = {i for i in range(len(prep.items)) if not keep[i] and E.scored_item_mask(prep_all)[i]}
ex = torch.load(os.path.join(P.LENSES, "lrp_experts_final.pt"), weights_only=True)
experts, pooled, counts = {}, {}, {}
for l, v in ex["experts"].items():
    if int(l) in layers:
        counts[int(l)] = ex["counts"][l]
        experts[int(l)], pooled[int(l)], _ = finalize_experts(v, ex["counts"][l])
del ex


def unit(m):
    return m.float() / m.float().norm()


def ranks(lens):
    return E.lens_ranks(model, prep, lens, layers, exclude=exclude)


# ---------------------------------------------------------------- split-independent lenses
fixed = {
    "logit": Lens("logit"),
    "J-lens": J,
    "R-lens": Rl,
    "B-lens": B,
    "Routed-LRP-a200": RoutedLens({l: adaptive_shrink(experts[l], pooled[l], counts[l], 200.0) for l in layers},
                                  router),
    "Routed-LRP-s0.5": RoutedLens({l: 0.5 * experts[l] + 0.5 * pooled[l][None] for l in layers}, router),
}
if LB is not None:
    fixed["LRP-B-lens"] = LB
R_fixed = {}
for name, lens in fixed.items():
    R_fixed[name] = ranks(lens)
    log(f"ranks {name}")


def fit_rows_of(fit_idx):
    fs = set(fit_idx)
    fr = [(r, i, w) for r, (i, w) in enumerate(rows) if i in fs]
    items_fit = sorted({i for _, i, _ in fr})
    return fr, items_fit, {i: j for j, i in enumerate(items_fit)}


def fit_weights(maps_l, fr, items_fit, pos_of, l, sample=None):
    """J++ weight search over the given stacked maps at layer l on the fitting
    rows. ``sample``: optional list of item ids (with repetition) = a bootstrap
    resample; its rows are used with multiplicity."""
    Z = expert_logits(model, maps_l, prep.resid[items_fit, l])
    if sample is None:
        use = fr
    else:
        by_item = {}
        for t in fr:
            by_item.setdefault(t[1], []).append(t)
        use = [t for i in sample for t in by_item[i]]
    w, sc = fit_expert_weights_v2(Z, [pos_of[i] for _, i, _ in use], [prep.cands[i][wd] for _, i, wd in use],
                               [prep.items[i].eval for _, i, _ in use], exclude=exclude, steps=a.steps)
    del Z
    return w, sc


def stratified_bootstrap(items_fit, rng):
    by_ev = {}
    for i in items_fit:
        by_ev.setdefault(prep.items[i].eval, []).append(i)
    out = []
    for ev in sorted(by_ev):
        ids = by_ev[ev]
        out += [ids[j] for j in rng.integers(0, len(ids), len(ids))]
    return out


out = {"layers": layers, "seeds": a.seeds, "bags": a.bags, "steps": a.steps, "candidates": a.candidates,
       "lrp_b": a.lrp_b, "per_seed": {}}
if a.oop:
    out["n_oop"] = len(oop)
    log(f"out-of-pool items (fail the correctness filter): {len(oop)}")
for seed in a.seeds:
    fit_idx, held_idx = split_items(prep.items, keep, seed=seed)
    held, fit_set = set(held_idx), set(fit_idx)
    fr, items_fit, pos_of = fit_rows_of(fit_idx)
    log(f"seed {seed}: {len(fit_idx)} fitting, {len(held_idx)} held-out items")
    jpp, jppb, rb, jb, rlb, bag, wlog = {}, {}, {}, {}, {}, {}, {}
    rng = np.random.default_rng(1000 + seed)
    samples = [stratified_bootstrap(items_fit, rng) for _ in range(a.bags)]
    for l in layers:
        w, sc = fit_weights(experts[l], fr, items_fit, pos_of, l)
        jpp[l] = combine(experts[l], w)
        wlog[f"J++|{l}"] = {"w": w.tolist(), "train": sc}
        for name, first, second, store in (("JppB", jpp, B, jppb), ("RB-mix", Rl.maps, B, rb),
                                           ("JB-mix", J.maps, B, jb), ("RLB-mix", Rl.maps, LB, rlb)):
            if name not in a.candidates:
                continue
            two = torch.stack([unit(first[l]), unit(second.maps[l])])
            w2, sc2 = fit_weights(two, fr, items_fit, pos_of, l)
            store[l] = combine(two, w2)
            wlog[f"{name}|{l}"] = {"w": w2.tolist(), "train": sc2}
        if "Bagged-J++" in a.candidates:
            ws = []
            for smp in samples:
                wb, _ = fit_weights(experts[l], fr, items_fit, pos_of, l, sample=smp)
                ws.append(wb)
            wbar = torch.stack(ws).mean(0)
            wbar = wbar / wbar.norm()
            bag[l] = combine(experts[l], wbar)
            wlog[f"Bagged-J++|{l}"] = {"w": wbar.tolist(), "w_bags": [x.tolist() for x in ws]}
        log(f"seed {seed} layer {l}: J++ train {sc:.3f}; " +
            "; ".join(f"{n} w={np.round(wlog[f'{n}|{l}']['w'], 2).tolist()}" for n in ("JppB", "RB-mix", "RLB-mix")
                      if f"{n}|{l}" in wlog))
    # Routed hybrid: experts shrunk toward the (rescaled) J++ map.
    hyb = {}
    for l in layers:
        target = jpp[l] * (pooled[l].norm() / jpp[l].norm())
        lam = counts[l].float() / (counts[l].float() + 200.0)
        hyb[l] = lam[:, None, None] * experts[l] + (1 - lam)[:, None, None] * target[None]
    sup = {"J++": Lens("jlens", jpp, name="J++"), "JppB": Lens("jlens", jppb, name="JppB"),
           "RB-mix": Lens("jlens", rb, name="RB-mix"), "JB-mix": Lens("jlens", jb, name="JB-mix"),
           "RLB-mix": Lens("jlens", rlb, name="RLB-mix"), "Bagged-J++": Lens("jlens", bag, name="Bagged-J++"),
           "Routed-J++-a200": RoutedLens(hyb, router)}
    sup = {n: v for n, v in sup.items() if n == "J++" or n in a.candidates}
    R_seed = dict(R_fixed)
    for name, lens in sup.items():
        R_seed[name] = ranks(lens)
    log(f"seed {seed}: ranks done")
    res = {}
    for name, r in R_seed.items():
        h = E.recall_at_k(prep, r, cols, 10, items_subset=held)
        f = E.recall_at_k(prep, r, cols, 10, items_subset=fit_set, n_boot=10)
        d = E.paired_bootstrap_diff(prep, r, R_seed["J++"], cols, cols, 10, items_subset=held)
        res[name] = {"held@10": h["macro"]["mean"], "held_ci": [h["macro"]["lo"], h["macro"]["hi"]],
                     "fit@10": f["macro"]["mean"], "diff_vs_jpp": d,
                     "per_eval": {ev: h[ev]["mean"] for ev in E.MACRO_EVALS if ev in h}}
        extra = ""
        if a.oop:
            o = E.recall_at_k(prep_all, r, cols, 10, items_subset=oop, n_boot=200)
            od = E.paired_bootstrap_diff(prep_all, r, R_seed["J++"], cols, cols, 10, items_subset=oop, n_boot=200)
            res[name]["oop@10"] = o["macro"]["mean"]
            res[name]["oop_diff_vs_jpp"] = od
            extra = f"  | oop {o['macro']['mean']:.3f} diff {od['diff']:+.3f}"
        log(f"seed {seed} {name:<16} held {res[name]['held@10']:.3f} fit {res[name]['fit@10']:.3f} "
            f"diff {d['diff']:+.3f} [{d['lo']:+.3f},{d['hi']:+.3f}]" + extra)
    out["per_seed"][str(seed)] = {"n_fit": len(fit_idx), "n_held": len(held_idx), "results": res, "weights": wlog}
    os.makedirs(P.RESULTS, exist_ok=True)
    with open(os.path.join(P.RESULTS, f"jpp_splits_{a.tag}.json"), "w") as fh:
        json.dump(out, fh, indent=1)

conf = [s for s in a.seeds if s != 0]
if conf:
    print("\nconfirmatory summary over seeds", conf)
    names = list(out["per_seed"][str(conf[0])]["results"])
    summ = {}
    for n in names:
        d = [out["per_seed"][str(s)]["results"][n]["diff_vs_jpp"]["diff"] for s in conf]
        h = [out["per_seed"][str(s)]["results"][n]["held@10"] for s in conf]
        summ[n] = {"mean_held": float(np.mean(h)), "mean_diff": float(np.mean(d)), "min_diff": float(np.min(d)),
                   "max_diff": float(np.max(d)), "wins": int(sum(x > 0 for x in d)),
                   "ties": int(sum(x == 0 for x in d)), "losses": int(sum(x < 0 for x in d)), "n": len(d)}
        summ[n]["meets_criterion"] = bool(n in a.candidates and summ[n]["mean_diff"] > 0
                                          and summ[n]["wins"] >= a.min_win_frac * len(d))
        extra = ""
        if a.oop:
            od = [out["per_seed"][str(s)]["results"][n]["oop_diff_vs_jpp"]["diff"] for s in conf]
            oh = [out["per_seed"][str(s)]["results"][n]["oop@10"] for s in conf]
            summ[n].update({"mean_oop": float(np.mean(oh)), "mean_oop_diff": float(np.mean(od)),
                            "oop_wins": int(sum(x > 0 for x in od))})
            extra = f"  | oop {np.mean(oh):.3f} diff {np.mean(od):+.3f} wins {summ[n]['oop_wins']}"
        print(f"  {n:<16} mean held {np.mean(h):.3f}  mean diff vs J++ {np.mean(d):+.3f} "
              f"[{np.min(d):+.3f},{np.max(d):+.3f}]  W/T/L {summ[n]['wins']}/{summ[n]['ties']}/{summ[n]['losses']}"
              f"{'  MEETS CRITERION' if summ[n]['meets_criterion'] else ''}" + extra)
    out["confirmatory"] = summ
    with open(os.path.join(P.RESULTS, f"jpp_splits_{a.tag}.json"), "w") as fh:
        json.dump(out, fh, indent=1)
log("done")
