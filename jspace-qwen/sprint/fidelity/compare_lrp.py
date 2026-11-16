"""Fidelity check, our side: the same prompt's per-cluster LRP Jacobian row sums from our
port (jspace.fitting.fit, exact estimator, lrp=True, same router, same token ids), compared
with jpp_lens's (sprint/fidelity/jpp_side.py output).

python sprint/fidelity/compare_lrp.py JPP_OUT.pt
"""

import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from jspace.fitting import fit  # noqa: E402
from jspace.model import load_model  # noqa: E402
from jspace.routing import Router  # noqa: E402

torch.set_num_threads(4)
ref = torch.load(sys.argv[1], weights_only=True)
layers = sorted(ref["row_sums"])
model = load_model("Qwen/Qwen3-0.6B")
router = Router.load(os.path.join(ROOT, "lenses", "router_E8.pt"))
ids = ref["input_ids"][None].long()
st = fit(model, [ids], source_layers=layers, method="exact", dim_batch=8, router=router, lrp=True,
         verbose=False, targets="all")
out = {"seq_len": int(ids.shape[1]), "target_layer_jpp": int(ref["target_layer"]),
       "target_layer_ours": int(st.target_layer), "layers": {}}
for l in layers:
    ours, theirs = st.cl_sum[l].float(), ref["row_sums"][l].float()
    c_ours, c_theirs = st.cl_count[l].long(), ref["counts"][l].long()
    rel = float((ours - theirs).norm() / theirs.norm())
    per_e = []
    for e in range(theirs.shape[0]):
        if c_theirs[e] == 0:
            per_e.append(None)
            continue
        a, b = ours[e].flatten(), theirs[e].flatten()
        per_e.append({"rel_err": float((a - b).norm() / b.norm()),
                      "cos": float(a @ b / (a.norm() * b.norm()))})
    pooled_o, pooled_t = ours.sum(0), theirs.sum(0)
    out["layers"][str(l)] = {"counts_equal": bool(torch.equal(c_ours, c_theirs)),
                             "counts_ours": c_ours.tolist(), "counts_jpp": c_theirs.tolist(),
                             "rel_err_all_experts": rel,
                             "rel_err_pooled": float((pooled_o - pooled_t).norm() / pooled_t.norm()),
                             "per_expert": per_e}
print(json.dumps(out, indent=1))
os.makedirs(os.path.join(ROOT, "sprint", "results"), exist_ok=True)
with open(os.path.join(ROOT, "sprint", "results", "fidelity_lrp.json"), "w") as f:
    json.dump(out, f, indent=1)
