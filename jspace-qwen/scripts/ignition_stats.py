"""Statistics for the ignition experiment (results/ignition_<tag>_<lens>.npy).

Per trial and layer, the 10%->90% transition width (in alpha) of the
reciprocal-rank share curve, computed exactly as in ignition.py. Intervals are
cluster bootstraps over concept pairs (every pair is run in several carrier
sentences, so trials of one pair are not independent).

If results/ignition_<tag>_linear_<lens>.npy exists (ignition.py --linear-control),
the same statistic is computed for the control in which the residual at each
layer is the straight-line interpolation between the alpha = 0 and alpha = 1
residuals. That control has the readout's own non-linearity but none of the
network's, so  width(actual) - width(linear)  isolates sharpening done by the
network rather than by the rank-share statistic.

python scripts/ignition_stats.py --tag main
"""

import argparse
import json
import os
import sys

import numpy as np

ROOT = __file__.rsplit("/scripts/", 1)[0]
sys.path.insert(0, ROOT)
from jspace import paths as P  # noqa: E402
from scripts.ignition import crossing  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="main")
ap.add_argument("--n-boot", type=int, default=2000)
a = ap.parse_args()

meta = json.load(open(os.path.join(P.RESULTS, f"ignition_{a.tag}.json")))
layers, alphas = meta["layers"], np.array(meta["alphas"])
trials = meta["trials"]
kinds = np.array([t["kind"] for t in trials])
pair_id = np.array([f"{t['kind']}|{t['A']}|{t['B']}" for t in trials])


def widths(S):
    """[trials, layers] transition widths (nan where the curve never switches)."""
    W = np.full((S.shape[0], S.shape[2]), np.nan)
    for t in range(S.shape[0]):
        for j in range(S.shape[2]):
            W[t, j] = crossing(alphas, S[t, :, j], 0.9) - crossing(alphas, S[t, :, j], 0.1)
    return W


def cluster_boot(values, clusters, stat=np.nanmedian, n=2000, seed=0):
    """values [trials] -> (stat, lo, hi) with a bootstrap over clusters."""
    rng = np.random.default_rng(seed)
    ids = np.unique(clusters)
    groups = [np.where(clusters == c)[0] for c in ids]
    est = float(stat(values))
    bs = []
    for _ in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[i] for i in pick])
        bs.append(stat(values[idx]))
    return [est, float(np.nanquantile(bs, 0.025)), float(np.nanquantile(bs, 0.975))]


for lens in ("jlens", "logit"):  # the control must agree with the actual run at alpha = 0 and 1
    pa = os.path.join(P.RESULTS, f"ignition_{a.tag}_{lens}.npy")
    pl = os.path.join(P.RESULTS, f"ignition_{a.tag}_linear_{lens}.npy")
    if os.path.exists(pa) and os.path.exists(pl):
        Sa, Sl = np.load(pa), np.load(pl)
        ends = np.abs(Sa[:, [0, -1], :] - Sl[:, [0, -1], :])
        print(f"{lens}: endpoint |actual - linear| share: max {ends.max():.3g}, mean {ends.mean():.3g}")

W = {}
for lens in ("jlens", "logit"):
    for cond in ("", "linear_"):
        p = os.path.join(P.RESULTS, f"ignition_{a.tag}_{cond}{lens}.npy")
        if os.path.exists(p):
            W[f"{cond}{lens}"] = widths(np.load(p))
            print("loaded", os.path.basename(p))

out = {"layers": layers, "per_layer": {}, "contrasts": {}}
for kind in sorted(set(kinds)):
    m = kinds == kind
    for name, Wm in W.items():
        rows = [cluster_boot(Wm[m, j], pair_id[m], n=a.n_boot) for j in range(len(layers))]
        out["per_layer"][f"{name}|{kind}"] = rows
        print(f"{name:<13} {kind:<9} " + " ".join(f"{r[0]:.2f}" for r in rows))
    # paired contrasts (same trials), median of per-trial differences
    def contrast(label, x):
        rows = [cluster_boot(x[m, j], pair_id[m], n=a.n_boot) for j in range(len(layers))]
        out["contrasts"][f"{label}|{kind}"] = rows
        print(f"  {label:<26} {kind:<9} " + " ".join(f"{r[0]:+.2f}" for r in rows))
    if "jlens" in W and "logit" in W:
        contrast("jlens-logit", W["jlens"] - W["logit"])
    for lens in ("jlens", "logit"):
        if lens in W and f"linear_{lens}" in W:
            contrast(f"{lens}: actual-linear", W[lens] - W[f"linear_{lens}"])
with open(os.path.join(P.RESULTS, f"ignition_{a.tag}_stats.json"), "w") as f:
    json.dump(out, f, indent=1)
print("layers:", layers)
