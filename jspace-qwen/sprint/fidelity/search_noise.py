"""How much does the J++ weight search's outcome depend on float-level details? Run the reference
search (jpp_lens fit_expert_weights, autograd) and our algorithmically identical fast search
(sprint/jppsearch.py, analytic gradient) on the same inputs for tuning splits 0-1 x 7 layers,
and score both weightings on the split's held-out items (per-layer pair-level macro pass@10, the
J++ objective's own metric, Readout Filtering on).

python sprint/fidelity/search_noise.py
"""
import json
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sprint"))
sys.path.insert(0, "/home/user/safety-research/jpp_lens/src")
from jppsearch import Pairs, fit_weights, macro  # noqa: E402
from workspace_lens.fitting.condense_experts.fitter import (DEFAULT_CONDENSE_CONFIG,  # noqa: E402
                                                            build_search_starting_points, fit_expert_weights)
from workspace_lens.fitting.condense_experts.rank_objective import PairTable, expert_pair_pass_scores  # noqa: E402

from jspace import evals as E  # noqa: E402
from jspace.jpp import expert_logits, finalize_experts, split_items  # noqa: E402
from jspace.model import load_model  # noqa: E402

torch.set_num_threads(4)
model = load_model("Qwen/Qwen3-0.6B")
prep = torch.load(os.path.join(ROOT, "lenses", "readout_prep.pt"), weights_only=False)
keep = E.scored_item_mask(prep)
rows = E.row_index(prep)
exclude = E.non_semantic_ids(model.tokenizer, model.vocab_size)
ex = torch.load(os.path.join(ROOT, "lenses", "lrp_experts_final.pt"), weights_only=True)
LAYERS = [4, 7, 11, 14, 18, 21, 24]


def pair_data(item_ids):
    s = set(item_ids)
    its = sorted({i for i, _ in rows if i in s})
    pos = {i: j for j, i in enumerate(its)}
    fr = [(i, w) for i, w in rows if i in s and prep.items[i].eval in E.MACRO_EVALS + ("order-ops",)]
    pairs = Pairs.build([pos[i] for i, _ in fr], [prep.cands[i][w] for i, w in fr], [prep.items[i].eval for i, _ in fr])
    excl = exclude.clone()
    for i, w in fr:
        excl[prep.cands[i][w]] = False
    return its, pairs, excl, fr


def heldout_pass(Lh, pairs, excl, w):
    z = torch.einsum("e,pev->pv", w.float(), Lh).masked_fill(excl[None], float("-inf"))
    best = z[pairs.item].gather(1, pairs.cands).amax(1)
    kth = z.topk(10, dim=1).values[:, -1][pairs.item]
    return macro((best - kth >= 0).float(), pairs)


out = []
for seed in (0, 1):
    fit_idx, held_idx = split_items(prep.items, keep, seed=seed)
    its_f, pairs_f, excl_f, _ = pair_data(fit_idx)
    its_h, pairs_h, excl_h, _ = pair_data(held_idx)
    for l in LAYERS:
        experts, _, _ = finalize_experts(ex["experts"][l], ex["counts"][l])
        Lf = expert_logits(model, experts, prep.resid[its_f, l])
        t0 = time.time()
        mine = fit_weights(Lf, pairs_f, excl_f)
        t_mine = time.time() - t0
        ev = tuple(sorted({prep.items[its_f[int(i)]].eval for i in pairs_f.item}))
        pt = PairTable(pairs_f.item, pairs_f.cands, pairs_f.ev, ev)
        t0 = time.time()
        sc = expert_pair_pass_scores(Lf, pt, excluded_mask_Bool_V=excl_f, k=10)
        ref = fit_expert_weights(Lf, pt, excluded_mask_Bool_V=excl_f,
                                 initial_weightings=build_search_starting_points(sc, top_n=3),
                                 condense_config=DEFAULT_CONDENSE_CONFIG)
        t_ref = time.time() - t0
        del Lf
        Lh = expert_logits(model, experts, prep.resid[its_h, l])
        r = {"seed": seed, "layer": l, "train_mine": mine.train_exact, "train_ref": ref.train_macro_pass,
             "held_mine": heldout_pass(Lh, pairs_h, excl_h, mine.w), "held_ref": heldout_pass(Lh, pairs_h, excl_h, ref.weights_E),
             "cos_w": float(mine.w @ ref.weights_E), "sec_mine": round(t_mine, 1), "sec_ref": round(t_ref, 1)}
        del Lh
        out.append(r)
        print(json.dumps(r), flush=True)
        os.makedirs(os.path.join(ROOT, "sprint", "results"), exist_ok=True)
        with open(os.path.join(ROOT, "sprint", "results", "search_noise.json"), "w") as f:
            json.dump(out, f, indent=1)
