"""E0.2: baseline metrics (metric_before) on the tuning splits, for the logit lens, J-lens, R-lens
and J++ (refitted on each split's fitting items with the J++ search), with all of M1-M3.
M4 (swaps) for the baseline lenses runs separately (sprint/swap_check.py) on split 0's J++ maps.

python sprint/baseline.py --seeds 0 1 2 3 4 5
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import LAYERS, STRATA, Ctx, LensSpec, evaluate, runlog_append, save_json  # noqa: E402
from sprintfit import fit_jpp  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
ap.add_argument("--tag", default="tuning")
a = ap.parse_args()
T0 = time.time()
ctx = Ctx()
fixed = ctx.standard_lenses()
reg = LensSpec("regress", maps={l: ctx.regress.maps[l] for l in LAYERS}, bias={l: ctx.regress.bias[l] for l in LAYERS})
tr_ref = {name: ctx.transport_metrics(lens) for name, lens in list(fixed.items()) + [("regress", reg)]}
print(f"[{time.time() - T0:.0f}s] transport metrics of fixed lenses done", flush=True)
out = {"seeds": a.seeds, "transport_fixed": tr_ref, "per_seed": {}}
maps_seed0 = None
for seed in a.seeds:
    fit_idx, held_idx = ctx.split(seed)
    jpp_maps, wlog = fit_jpp(ctx, fit_idx)
    if maps_seed0 is None:
        maps_seed0 = jpp_maps
    print(f"[{time.time() - T0:.0f}s] seed {seed}: J++ fitted", flush=True)
    lenses = dict(fixed)
    lenses["J++"] = LensSpec("J++", maps=jpp_maps)
    res = {}
    for name, lens in lenses.items():
        summ, _ = evaluate(ctx, lens, held_idx, seed, wiki=True, transport=(name == "J++"))
        if name != "J++":
            for l in LAYERS:
                summ["per_layer"][l].update({f"tr_{k}": v for k, v in tr_ref[name][l].items()})
            for s, ls in STRATA.items():
                for k in ("cos", "nmse", "kl"):
                    summ[s][f"tr_{k}"] = float(np.mean([tr_ref[name][l][k] for l in ls]))
        res[name] = summ
    fit_summ, _ = evaluate(ctx, lenses["J++"], fit_idx, seed, wiki=False, transport=False)
    res["J++ (fit split, in-sample)"] = {"best7_recall10": fit_summ["best7_recall10"],
                                          **{s: {"recall10": fit_summ[s]["recall10"]} for s in STRATA}}
    out["per_seed"][str(seed)] = {"results": res, "jpp_weights": wlog, "n_fit": len(fit_idx), "n_held": len(held_idx)}
    print(f"[{time.time() - T0:.0f}s] seed {seed}: " + "; ".join(
        f"{n} early {r['early']['recall10']:.3f} mid {r['mid']['recall10']:.3f} late {r['late']['recall10']:.3f} "
        f"best7 {r['best7_recall10']:.3f}" for n, r in res.items() if "fit split" not in n), flush=True)
    save_json(out, f"baseline_{a.tag}.json")
torch.save({"J++": {l: m.half() for l, m in maps_seed0.items()},
            "J-lens": {l: ctx.jlens.maps[l].half() for l in LAYERS},
            "R-lens": {l: ctx.rlens.maps[l].half() for l in LAYERS}},
           os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", f"maps_baseline_{a.tag}_seed{a.seeds[0]}.pt"))

# aggregate over seeds
names = [n for n in out["per_seed"][str(a.seeds[0])]["results"] if "fit split" not in n]
agg = {}
for n in names:
    agg[n] = {}
    for s in list(STRATA) + ["best7"]:
        keys = ["recall10"] if s == "best7" else ["recall10", "mrr", "fp10_matched", "mass_matched", "auc_matched",
                                                  "fp10_wiki", "mass_wiki", "own_mass", "tr_cos", "tr_nmse", "tr_kl"]
        for k in keys:
            v = [out["per_seed"][str(sd)]["results"][n]["best7_recall10"] if s == "best7"
                 else out["per_seed"][str(sd)]["results"][n][s].get(k, float("nan")) for sd in a.seeds]
            agg[n][f"{s}.{k}"] = {"mean": float(np.mean(v)), "min": float(np.min(v)), "max": float(np.max(v))}
fitg = {s: float(np.mean([out["per_seed"][str(sd)]["results"]["J++ (fit split, in-sample)"][s]["recall10"]
                          for sd in a.seeds])) for s in STRATA}
out["aggregate"] = agg
out["jpp_fit_split_recall10"] = fitg
save_json(out, f"baseline_{a.tag}.json")
for n in names:
    print(n, {k: round(v["mean"], 4) for k, v in agg[n].items()})
j = agg["J++"]
runlog_append("E0.2-baseline", "metric_before for J++ and reference lenses (tuning splits " + ",".join(map(str, a.seeds)) + ")",
              f"J++ held-out recall@10 early/mid/late/best7 = {j['early.recall10']['mean']:.3f}/{j['mid.recall10']['mean']:.3f}/"
              f"{j['late.recall10']['mean']:.3f}/{j['best7.recall10']['mean']:.3f}; MRR early {j['early.mrr']['mean']:.3f}",
              f"FP@10 matched early {j['early.fp10_matched']['mean']:.3f}, wiki early {j['early.fp10_wiki']['mean']:.4f}; "
              f"transport cos early {j['early.tr_cos']['mean']:.3f}, NMSE {j['early.tr_nmse']['mean']:.3f}",
              f"R-lens early/best7 {agg['R-lens']['early.recall10']['mean']:.3f}/{agg['R-lens']['best7.recall10']['mean']:.3f}; "
              f"J++ in-sample early {fitg['early']:.3f}",
              "baseline only; see sprint/results/baseline_" + a.tag + ".json")
print(f"done in {time.time() - T0:.0f}s")
