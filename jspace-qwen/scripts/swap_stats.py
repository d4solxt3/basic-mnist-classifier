"""Paired statistics for the clamp-swap experiments (results/swaps_<tag>_rows.json).

For each experiment / variant / scale: success rate per lens, and the paired
difference in success between the J-lens and each control lens on the same
trials (item-bootstrap 95% CI and discordant-pair counts). Only trials whose
clean pass is correct are scored, as in swaps.py.

python scripts/swap_stats.py --tags main midband
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = __file__.rsplit("/scripts/", 1)[0]
sys.path.insert(0, ROOT)
from jspace import paths as P  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tags", nargs="+", default=["main"])
a = ap.parse_args()


def paired(x, y, n=5000, seed=0):
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = x - y
    rng = np.random.default_rng(seed)
    b = d[rng.integers(0, len(d), size=(n, len(d)))].mean(1)
    return {"diff": float(d.mean()), "lo": float(np.quantile(b, 0.025)), "hi": float(np.quantile(b, 0.975)),
            "j_only": int(((x == 1) & (y == 0)).sum()), "other_only": int(((x == 0) & (y == 1)).sum())}


out = {}
for tag in a.tags:
    p = os.path.join(P.RESULTS, f"swaps_{tag}_rows.json")
    if not os.path.exists(p):
        continue
    rows = [r for r in json.load(open(p)) if r["clean_base_rank"] == 1]
    by = defaultdict(dict)  # (exp, variant, scale, name) -> lens -> row
    for r in rows:
        by[(r["exp"], r["variant"], r["scale"], r["name"])][r["lens"]] = r
    res = {}
    keys = sorted({k[:3] for k in by})
    for exp, var, sc in keys:
        trials = [v for k, v in by.items() if k[:3] == (exp, var, sc)]
        lenses = sorted({ln for t in trials for ln in t})
        succ = {ln: [int(t[ln]["success_rank"] == 1) for t in trials if ln in t] for ln in lenses}
        norm = {ln: float(np.median([t[ln]["edit_norm_ratio"] for t in trials if ln in t])) for ln in lenses}
        entry = {"n": len(trials), "success": {ln: float(np.mean(v)) for ln, v in succ.items()},
                 "median_edit_norm": norm}
        if "jlens" in lenses:
            for ln in lenses:
                if ln == "jlens":
                    continue
                both = [t for t in trials if "jlens" in t and ln in t]
                entry[f"jlens_minus_{ln}"] = paired([int(t["jlens"]["success_rank"] == 1) for t in both],
                                                    [int(t[ln]["success_rank"] == 1) for t in both])
        res[f"{exp}|{var}|{sc}"] = entry
        line = "  ".join(f"{ln} {entry['success'][ln]:.3f} (edit {norm[ln]:.2f})" for ln in lenses)
        extra = "  ".join(f"J-{k.split('_')[-1]} {entry[k]['diff']:+.3f} [{entry[k]['lo']:+.3f},{entry[k]['hi']:+.3f}]"
                          f" ({entry[k]['j_only']}/{entry[k]['other_only']})" for k in entry if k.startswith("jlens_minus"))
        print(f"{tag:<8} {exp:<10} {var:<12} x{sc:<4} n={len(trials):>3}  {line}  |  {extra}")
    out[tag] = res
with open(os.path.join(P.RESULTS, "swap_stats.json"), "w") as f:
    json.dump(out, f, indent=1)
