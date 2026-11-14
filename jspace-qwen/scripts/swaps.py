"""Causal clamp-swap experiments with matched controls.

probe-swap (90 two-hop prompts):
  main         intermediate -> swap_to      success = swap_answer at rank 1
  answer_swap  answer -> swap_answer        (ceiling: is the edit strong enough?)
  random_null  intermediate -> unrelated    success = swap_answer (should be ~0)
flexible-generalization (4 categories x 4 functions x 12 ordered arg pairs):
  arg A -> arg B in "f(A)" prompts          success = f(B) at rank 1

Every trial is run with J-lens, logit-lens (and optionally regression-lens)
directions, unit-normalised, at several scales. Only trials whose clean pass
is correct (baseline answer at word-rank 1) are scored.

python scripts/swaps.py --jlens lenses/exact_final.pt --band 8 18 --tag main
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch

ROOT = __file__.rsplit("/scripts/", 1)[0]
sys.path.insert(0, ROOT)
from jspace import evals as E  # noqa: E402
from jspace.interventions import _min_rank, clean_pass, swap  # noqa: E402
from jspace.lens import Lens  # noqa: E402
from jspace.model import load_model  # noqa: E402
from jspace import paths as P  # noqa: E402

RANDOM_POOL = ("table", "window", "pencil", "carpet", "engine", "ladder", "button", "pillow", "basket", "mirror")


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return [float("nan")] * 3
    rng = np.random.default_rng(seed)
    b = rng.choice(x, size=(n, len(x)), replace=True).mean(1)
    return [float(x.mean()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]


def probe_swap_trials(tok):
    items = json.load(open(os.path.join(ROOT, "data", "experiments", "probe-swap.json")))["items"]
    rng = random.Random(0)
    trials = []
    for it in items:
        succ = E.single_token_ids(tok, E.surface_forms(it["swap_answer"]))
        base = E.single_token_ids(tok, E.surface_forms(it["answer"]))
        rnd = rng.choice(RANDOM_POOL)
        for variant, s, t in (("main", it["intermediate"], it["swap_to"]),
                              ("answer_swap", it["answer"], it["swap_answer"]),
                              ("random_null", it["intermediate"], rnd)):
            trials.append(dict(exp="probe-swap", name=it["name"], category=it["category"], variant=variant,
                               prompt=it["prompt"], src=s, tgt=t, succ=succ, base=base))
    return trials


def flexgen_trials(tok):
    cats = json.load(open(os.path.join(ROOT, "data", "experiments", "flexible-generalization.json")))["categories"]
    trials = []
    for c in cats:
        for f in c["funcs"]:
            for a_ in c["args"]:
                for b_ in c["args"]:
                    if a_ == b_:
                        continue
                    succ = E.single_token_ids(tok, E.surface_forms(f["answers"][b_]))
                    base = E.single_token_ids(tok, E.surface_forms(f["answers"][a_]))
                    trials.append(dict(exp="flexgen", name=f"{c['name']}/{f['name']}/{a_}->{b_}",
                                       category=c["name"], variant="main",
                                       prompt=f["template"].format(arg=a_), src=a_, tgt=b_, succ=succ, base=base))
                    rnd = RANDOM_POOL[hash((c["name"], f["name"], a_, b_)) % len(RANDOM_POOL)]
                    trials.append(dict(exp="flexgen", name=f"{c['name']}/{f['name']}/{a_}->{rnd}",
                                       category=c["name"], variant="random_null",
                                       prompt=f["template"].format(arg=a_), src=a_, tgt=rnd, succ=succ, base=base))
    return trials


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--jlens", required=True)
    ap.add_argument("--regress", action="store_true")
    ap.add_argument("--band", type=int, nargs=2, required=True)
    ap.add_argument("--scales", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--exps", nargs="+", default=["probe-swap", "flexgen"])
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    model = load_model(a.model)
    tok = model.tokenizer
    band = list(range(a.band[0], a.band[1] + 1))
    lenses = {"jlens": Lens.load(P.resolve(a.jlens)),
              "logit": Lens("logit")}
    if a.regress:
        lenses["regress"] = Lens.load(os.path.join(P.LENSES, "regress.pt"))
    word_mask = E.non_semantic_ids(tok, model.vocab_size)
    trials = []
    if "probe-swap" in a.exps:
        trials += probe_swap_trials(tok)
    if "flexgen" in a.exps:
        trials += flexgen_trials(tok)

    rows = []
    clean_cache = {}
    for i, tr in enumerate(trials):
        if not tr["succ"] or not tr["base"]:
            continue
        s_id, t_id = E.resolve_token(tok, tr["src"]), E.resolve_token(tok, tr["tgt"])
        if s_id is None or t_id is None:
            continue
        ids = model.encode(tr["prompt"])
        if tr["prompt"] not in clean_cache:
            clean_cache[tr["prompt"]] = clean_pass(model, ids, band)
        cl = clean_cache[tr["prompt"]]
        base_rank0 = _min_rank(cl[0], tr["base"], word_mask)
        succ_rank0 = _min_rank(cl[0], tr["succ"], word_mask)
        for lname, lens in lenses.items():
            for sc in a.scales:
                r = swap(model, ids, lens, band, s_id, t_id, tr["succ"], tr["base"], scale=sc, clean=cl,
                         word_mask=word_mask)
                rows.append(dict(exp=tr["exp"], name=tr["name"], category=tr["category"], variant=tr["variant"],
                                 lens=lname, scale=sc, clean_base_rank=base_rank0, clean_succ_rank=succ_rank0,
                                 success_rank=r.success_rank, baseline_rank=r.baseline_rank,
                                 success_logprob=r.success_logprob, baseline_logprob=r.baseline_logprob,
                                 top=r.top, edit_norm_ratio=r.edit_norm_ratio, n_layers=r.n_layers_edited))
        if i % 25 == 0:
            print(f"trial {i}/{len(trials)}", flush=True)

    res_dir = P.RESULTS
    os.makedirs(res_dir, exist_ok=True)
    with open(os.path.join(res_dir, f"swaps_{a.tag}_rows.json"), "w") as f:
        json.dump(rows, f)
    summary = {"band": band, "scales": a.scales, "results": {}}
    for exp in ("probe-swap", "flexgen"):
        for variant in ("main", "answer_swap", "random_null"):
            for lname in lenses:
                for sc in a.scales:
                    sel = [r for r in rows if r["exp"] == exp and r["variant"] == variant and r["lens"] == lname
                           and r["scale"] == sc and r["clean_base_rank"] == 1]
                    if not sel:
                        continue
                    succ = [r["success_rank"] == 1 for r in sel]
                    flip = [r["baseline_rank"] > 1 for r in sel]
                    key = f"{exp}|{variant}|{lname}|{sc}"
                    summary["results"][key] = {"n": len(sel), "success": boot_ci(succ), "baseline_lost": boot_ci(flip),
                                               "median_edit_norm": float(np.median([r["edit_norm_ratio"] for r in sel]))}
                    m = summary["results"][key]
                    print(f"{key:<40} n={m['n']:>3} success={m['success'][0]:.3f} "
                          f"[{m['success'][1]:.3f},{m['success'][2]:.3f}] lost={m['baseline_lost'][0]:.3f} "
                          f"edit={m['median_edit_norm']:.2f}")
    with open(os.path.join(res_dir, f"swaps_{a.tag}.json"), "w") as f:
        json.dump(summary, f, indent=1)
