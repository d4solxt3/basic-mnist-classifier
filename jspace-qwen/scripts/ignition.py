"""Ignition: does the lens readout switch abruptly as the input is morphed?

For a concept pair (A, B) and a carrier sentence with slot {W}, the slot
token's input embedding is replaced by  alpha*emb(A) + (1-alpha)*emb(B), alpha
swept 0 -> 1. At each layer we read out A's reciprocal-rank share
    s = (1/rank_A) / (1/rank_A + 1/rank_B)
at the slot position. Per (trial, layer) we fit the 10%->90% transition width
in alpha. A global-workspace-like "ignition" predicts sharp (narrow)
transitions in the workspace band. Two controls: (i) the same statistic on the
*logit lens*, and (ii) the linear input mixture itself (width 0.8 by
construction for a linear readout of a linear mix).

python scripts/ignition.py --jlens lenses/exact_final.pt --carriers 20 --tag main
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np
import torch

ROOT = __file__.rsplit("/scripts/", 1)[0]
sys.path.insert(0, ROOT)
from jspace import evals as E  # noqa: E402
from jspace.lens import Lens  # noqa: E402
from jspace.model import Recorder, load_model  # noqa: E402
from jspace import paths as P  # noqa: E402


def build_ids(tok, template, word_id):
    pre, post = template.split("{W}")
    pre_ids = tok(pre.rstrip(), add_special_tokens=False).input_ids
    post_ids = tok(post, add_special_tokens=False).input_ids if post else []
    ids = pre_ids + [word_id] + post_ids
    return torch.tensor([ids]), len(pre_ids)


def crossing(alphas, s, level):
    """First alpha at which the (monotone-smoothed) share curve reaches level."""
    s = np.maximum.accumulate(s)
    idx = np.where(s >= level)[0]
    if len(idx) == 0:
        return np.nan
    j = idx[0]
    if j == 0:
        return alphas[0]
    a0, a1, s0, s1 = alphas[j - 1], alphas[j], s[j - 1], s[j]
    return a0 + (level - s0) * (a1 - a0) / max(s1 - s0, 1e-9)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--jlens", required=True)
    ap.add_argument("--carriers", type=int, default=20)
    ap.add_argument("--alphas", type=int, default=21)
    ap.add_argument("--layer-stride", type=int, default=2)
    ap.add_argument("--controls", action="store_true", help="also run idiom/scrambled pairs")
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()

    model = load_model(a.model)
    tok = model.tokenizer
    data = json.load(open(os.path.join(ROOT, "data", "experiments", "ignition.json")))
    jl = Lens.load(P.resolve(a.jlens))
    lenses = {"jlens": jl, "logit": Lens("logit")}
    L = model.n_layers
    layers = list(range(0, L - 1, a.layer_stride)) + [L - 1]
    alphas = np.linspace(0, 1, a.alphas)
    pairs = [("country", x, y) for x, y in itertools.combinations(data["countries_12"], 2)]
    if a.controls:
        pairs += [("idiom", x, y) for x, y in data["idiom_pairs"]]
        pairs += [("scrambled", x, y) for x, y in data["scrambled_pairs"]]
    carriers = data["ctx_templates"][:a.carriers]
    noun_carriers = data["noun_ctx_templates"][:max(1, a.carriers // 2)]
    emb = model.embed.weight

    out = {"layers": layers, "alphas": alphas.tolist(), "trials": []}
    shares = {k: [] for k in lenses}
    for pi, (kind, A, B) in enumerate(pairs):
        ia, ib = E.resolve_token(tok, A), E.resolve_token(tok, B)
        if ia is None or ib is None:
            continue
        for tmpl in (carriers if kind == "country" else noun_carriers):
            ids, pos = build_ids(tok, tmpl, ia)
            with torch.no_grad():
                x = emb[ids].repeat(len(alphas), 1, 1).clone()
                al = torch.tensor(alphas, dtype=x.dtype)[:, None]
                x[:, pos] = al * emb[ia] + (1 - al) * emb[ib]
                with Recorder(model, layers) as rec:
                    model.text(inputs_embeds=x, use_cache=False)
                for lname, lens in lenses.items():
                    S = np.zeros((len(alphas), len(layers)))
                    for j, l in enumerate(layers):
                        h = rec.acts[l][:, pos].float()
                        z = lens.logits(model, h, l) if (lname == "logit" or l < L - 1) else model.unembed_f32(h)
                        ra = (z > z[:, ia:ia + 1]).sum(1).float() + 1
                        rb = (z > z[:, ib:ib + 1]).sum(1).float() + 1
                        S[:, j] = ((1 / ra) / (1 / ra + 1 / rb)).numpy()
                    shares[lname].append(S)
            out["trials"].append({"kind": kind, "A": A, "B": B, "carrier": tmpl})
        print(f"pair {pi + 1}/{len(pairs)} {A}/{B}", flush=True)

    res = {"layers": layers, "alphas": alphas.tolist(), "trials": out["trials"], "summary": {}}
    for lname, S_list in shares.items():
        S = np.stack(S_list)  # [trials, alphas, layers]
        np.save(os.path.join(P.RESULTS, f"ignition_{a.tag}_{lname}.npy"), S)
        kinds = np.array([t["kind"] for t in out["trials"]])
        for kind in sorted(set(kinds)):
            Sk = S[kinds == kind]
            widths, thr = [], []
            for j in range(len(layers)):
                w = [crossing(alphas, Sk[t, :, j], 0.9) - crossing(alphas, Sk[t, :, j], 0.1) for t in range(len(Sk))]
                th = [crossing(alphas, Sk[t, :, j], 0.5) for t in range(len(Sk))]
                widths.append([float(np.nanmedian(w)), float(np.mean(np.isfinite(w)))])
                thr.append(float(np.nanmedian(th)))
            res["summary"][f"{lname}|{kind}"] = {"median_width": [x[0] for x in widths],
                                                 "frac_switching": [x[1] for x in widths],
                                                 "median_threshold": thr,
                                                 "mean_share": Sk.mean(0).tolist()}
            print(lname, kind, "width by layer:", [f"{x[0]:.2f}" for x in widths])
    with open(os.path.join(P.RESULTS, f"ignition_{a.tag}.json"), "w") as f:
        json.dump(res, f)
