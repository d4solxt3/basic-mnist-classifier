"""Pre-declared workspace-band rule, applied to the J-lens per-layer readout curve
before any causal experiment:

  band = the contiguous run of layers around argmax_l recall@10(l) on which
         recall@10(l) >= 0.5 * max, excluding the last three (output-adjacent)
         layers.

python scripts/choose_band.py results/readout_main.json "J-lens|raw"
"""

import json
import os
import sys

ROOT = __file__.rsplit("/scripts/", 1)[0]
path, key = sys.argv[1], sys.argv[2]
res = json.load(open(path if os.path.isabs(path) else os.path.join(ROOT, path)))
curve = {int(l): v for l, v in res["results"][key]["per_layer@10"].items()}
L = max(curve) + 2  # curve covers layers 0..L-2
allowed = [l for l in sorted(curve) if l <= L - 4]
peak = max(allowed, key=lambda l: curve[l])
thr = 0.5 * curve[peak]
lo = hi = peak
while lo - 1 in curve and lo - 1 in allowed and curve[lo - 1] >= thr:
    lo -= 1
while hi + 1 in curve and hi + 1 in allowed and curve[hi + 1] >= thr:
    hi += 1
out = {"source": path, "key": key, "peak": peak, "peak_recall": curve[peak], "threshold": thr, "band": [lo, hi],
       "curve": curve}
with open(os.path.join(ROOT, "results", "band.json"), "w") as f:
    json.dump(out, f, indent=1)
print(f"peak layer {peak} (recall@10={curve[peak]:.3f}); band = [{lo}, {hi}]")
