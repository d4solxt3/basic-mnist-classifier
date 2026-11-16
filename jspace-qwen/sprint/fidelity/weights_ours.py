"""Fidelity check of the J++ weight search. Stage 1 (our venv): build the fitting split's
expert logits for two layers, run our port (fit_expert_weights_v2), save inputs and result.
Stage 2 (sprint/fidelity/weights_jpp.py, jpp_lens venv) runs jpp_lens's fit_expert_weights
on the identical tensors; stage 3 (this script with --compare) compares.

python sprint/fidelity/weights_ours.py OUT_DIR
python sprint/fidelity/weights_ours.py OUT_DIR --compare
"""

import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
OUT = sys.argv[1]
LAYERS = [4, 21]
SEED = 0  # a tuning split (round-2 seed 0, already looked at)

if "--compare" in sys.argv:
    res = {}
    for l in LAYERS:
        o = torch.load(os.path.join(OUT, f"ours_{l}.pt"), weights_only=True)
        j = torch.load(os.path.join(OUT, f"jpp_{l}.pt"), weights_only=True)
        res[str(l)] = {"cos_weights": float(o["w"] @ j["w"] / (o["w"].norm() * j["w"].norm())),
                       "train_macro_ours": o["score"], "train_macro_jpp": j["score"],
                       "w_ours": [round(x, 4) for x in o["w"].tolist()],
                       "w_jpp": [round(x, 4) for x in j["w"].tolist()], "jpp_init": j["init"], "jpp_step": j["step"]}
    print(json.dumps(res, indent=1))
    with open(os.path.join(ROOT, "sprint", "results", "fidelity_weights.json"), "w") as f:
        json.dump(res, f, indent=1)
    sys.exit(0)

from jspace import evals as E  # noqa: E402
from jspace.jpp import expert_logits, finalize_experts, fit_expert_weights_v2, split_items  # noqa: E402
from jspace.model import load_model  # noqa: E402

torch.set_num_threads(4)
os.makedirs(OUT, exist_ok=True)
model = load_model("Qwen/Qwen3-0.6B")
prep = torch.load(os.path.join(ROOT, "lenses", "readout_prep.pt"), weights_only=False)
keep = E.scored_item_mask(prep)
rows = E.row_index(prep)
exclude = E.non_semantic_ids(model.tokenizer, model.vocab_size)
fit_idx, _ = split_items(prep.items, keep, seed=SEED)
fs = set(fit_idx)
items_fit = sorted({i for i, _ in rows if i in fs})
pos_of = {i: j for j, i in enumerate(items_fit)}
fr = [(i, w) for i, w in rows if i in fs]
ex = torch.load(os.path.join(ROOT, "lenses", "lrp_experts_final.pt"), weights_only=True)
for l in LAYERS:
    experts, _, _ = finalize_experts(ex["experts"][l], ex["counts"][l])
    Z = expert_logits(model, experts, prep.resid[items_fit, l])           # [n, E, V]
    row_item = [pos_of[i] for i, _ in fr]
    row_cands = [prep.cands[i][w] for i, w in fr]
    row_eval = [prep.items[i].eval for i, _ in fr]
    w, sc = fit_expert_weights_v2(Z, row_item, row_cands, row_eval, exclude=exclude)
    torch.save({"Z": Z, "row_item": row_item, "row_cands": row_cands, "row_eval": row_eval,
                "exclude": exclude, "n_items": len(items_fit)}, os.path.join(OUT, f"inputs_{l}.pt"))
    torch.save({"w": w, "score": sc}, os.path.join(OUT, f"ours_{l}.pt"))
    print(f"layer {l}: ours train macro {sc:.4f} w={[round(x, 3) for x in w.tolist()]}")
