"""Fidelity check stage 2 (jpp_lens venv): jpp_lens's own fit_expert_weights on the identical
expert logits, pair table and exclusion mask that our port saw (weights_ours.py stage 1).

/home/user/safety-research/jpp_lens/.venv/bin/python sprint/fidelity/weights_jpp.py OUT_DIR
"""
import os
import sys

import torch as t

from workspace_lens.fitting.condense_experts.fitter import (DEFAULT_CONDENSE_CONFIG, build_search_starting_points,
                                                            fit_expert_weights)
from workspace_lens.fitting.condense_experts.rank_objective import PairTable, expert_pair_pass_scores

OUT = sys.argv[1]
t.set_num_threads(4)
for l in (4, 21):
    d = t.load(os.path.join(OUT, f"inputs_{l}.pt"), weights_only=False)
    Z = d["Z"].float()
    ev_names = tuple(sorted(set(d["row_eval"])))
    maxc = max(len(c) for c in d["row_cands"])
    pairs = PairTable(item_index_Int_Q=t.tensor(d["row_item"], dtype=t.long),
                      candidate_ids_Int_QC=t.tensor([list(c) + [c[0]] * (maxc - len(c)) for c in d["row_cands"]],
                                                    dtype=t.long),
                      eval_index_Int_Q=t.tensor([ev_names.index(e) for e in d["row_eval"]], dtype=t.long),
                      group_eval_slugs=ev_names)
    excl = d["exclude"].clone()
    for c in d["row_cands"]:
        excl[list(c)] = False          # candidates are never excluded (jpp's rule)
    scores = expert_pair_pass_scores(Z, pairs, excluded_mask_Bool_V=excl, k=10)
    lw = fit_expert_weights(Z, pairs, excluded_mask_Bool_V=excl,
                            initial_weightings=build_search_starting_points(scores, top_n=3),
                            condense_config=DEFAULT_CONDENSE_CONFIG)
    t.save({"w": lw.weights_E, "score": lw.train_macro_pass, "init": lw.init_name, "step": lw.step},
           os.path.join(OUT, f"jpp_{l}.pt"))
    print(f"layer {l}: jpp train macro {lw.train_macro_pass:.4f} init {lw.init_name} step {lw.step}")
