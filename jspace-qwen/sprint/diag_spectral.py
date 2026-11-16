"""Label-free diagnostics behind Experiment 1 (no readout labels are touched).

Premise of Exp 1: within each LRP expert, some output directions carry reproducible
expert-specific structure and others carry estimation noise, and the pooled map's singular bands
separate the two. We test the premise directly with a split-half reliability analysis:

* pooled LRP map  Jbar_l = sum_e n_e J_e / sum_e n_e,  SVD  Jbar = U S V^T;
* output bands on U: [0,16), [16,64), [64,256), [256,1024);
* expert deviations D_e = J_e - Jbar from prompts 0-15 (snapshot n16) and from prompts 16-31
  (derived from the n32 and n16 sums), projected on each band: P_b D_e = U_b U_b^T D_e;
* reliability r_b = <P_b D^(1), P_b D^(2)> / (|P_b D^(1)| |P_b D^(2)|), count-weighted over experts;
  signal-to-noise per band from the half-sums;
* coordinate alignment of the leading singular vectors (max |u_k[i]|; 1 = a single coordinate),
  for the "coordinate-aligned outlier directions" part of the hypothesis;
* the same reliability in the readout metric (A P_b D, A = W_U diag(g)), which is what ranks see.

python sprint/diag_spectral.py
"""
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from jspace import paths as P  # noqa: E402
from jspace.model import load_model  # noqa: E402

torch.set_num_threads(2)
LAYERS = [4, 7, 11, 14, 18, 21, 24]
EDGES = [0, 16, 64, 256, 1024]
MIN_HALF = 25

full = torch.load(os.path.join(P.LENSES, "lrp_experts_final.pt"), weights_only=True)
half = torch.load(os.path.join(P.LENSES, "lrp_experts_n16.pt"), weights_only=True)
model = load_model("Qwen/Qwen3-0.6B")
A = (model.lm_head.weight.float() * model.final_norm.weight.float()[None, :])
G = A.T @ A                                                          # readout metric, [d, d]
del model, A
out = {"edges": EDGES, "layers": {}}
for l in LAYERS:
    m32, c32 = full["experts"][l].float(), full["counts"][l].float()
    m16, c16 = half["experts"][l].float(), half["counts"][l].float()
    s32, s16 = m32 * c32[:, None, None], m16 * c16[:, None, None]
    c2 = c32 - c16
    h1 = m16
    h2 = (s32 - s16) / c2.clamp_min(1)[:, None, None]
    pooled = s32.sum(0) / c32.sum()
    pooled1, pooled2 = s16.sum(0) / c16.sum(), (s32 - s16).sum(0) / c2.sum()
    U, S, Vh = torch.linalg.svd(pooled)
    keep = (c16 >= MIN_HALF) & (c2 >= MIN_HALF)
    w = torch.minimum(c16, c2) * keep
    bands = []
    for b in range(len(EDGES) - 1):
        Ub = U[:, EDGES[b]:EDGES[b + 1]]
        num = den1 = den2 = numA = den1A = den2A = 0.0
        e_sig, e_noise, e_total = 0.0, 0.0, 0.0
        for e in range(m32.shape[0]):
            if not keep[e]:
                continue
            D1 = Ub @ (Ub.T @ (h1[e] - pooled1))
            D2 = Ub @ (Ub.T @ (h2[e] - pooled2))
            Dfull = Ub @ (Ub.T @ (m32[e] - pooled))
            num += float(w[e] * (D1 * D2).sum())
            den1 += float(w[e] * (D1 * D1).sum())
            den2 += float(w[e] * (D2 * D2).sum())
            GD1, GD2 = G @ D1, G @ D2
            numA += float(w[e] * (D1 * GD2).sum())
            den1A += float(w[e] * (D1 * GD1).sum())
            den2A += float(w[e] * (D2 * GD2).sum())
            e_sig += float(w[e] * (D1 * D2).sum())
            e_noise += float(w[e] * ((D1 - D2) ** 2).sum()) / 4
            e_total += float(w[e] * (Dfull * Dfull).sum())
        bands.append({"band": f"{EDGES[b]}-{EDGES[b + 1]}",
                      "reliability_frob": num / (den1 * den2) ** 0.5 if den1 > 0 and den2 > 0 else None,
                      "reliability_readout": numA / (den1A * den2A) ** 0.5 if den1A > 0 and den2A > 0 else None,
                      "signal_over_noise_frob": e_sig / e_noise if e_noise > 0 else None,
                      "deviation_energy": e_total,
                      "pooled_energy": float((S[EDGES[b]:EDGES[b + 1]] ** 2).sum())})
    tot_dev = sum(b["deviation_energy"] for b in bands)
    for b in bands:
        b["deviation_energy_frac"] = b["deviation_energy"] / tot_dev
        b["pooled_energy_frac"] = b["pooled_energy"] / float((S ** 2).sum())
    align_u = U[:, :10].abs().max(0).values.tolist()
    align_v = Vh[:10].abs().max(1).values.tolist()
    out["layers"][str(l)] = {"experts_kept_for_split_half": int(keep.sum()), "bands": bands,
                             "top10_left_sv_max_coord": [round(x, 3) for x in align_u],
                             "top10_right_sv_max_coord": [round(x, 3) for x in align_v],
                             "singular_values_top5": [round(float(x), 2) for x in S[:5]],
                             "effective_rank": float(S.sum() ** 2 / (S ** 2).sum())}
    print(f"layer {l}: kept {int(keep.sum())}/8  eff.rank {out['layers'][str(l)]['effective_rank']:.0f}  "
          f"max|u_k| top10 {[round(x, 2) for x in align_u]}  max|v_k| {[round(x, 2) for x in align_v]}")
    for b in bands:
        print(f"   band {b['band']:>9}: reliability frob {b['reliability_frob']:.3f} readout {b['reliability_readout']:.3f}"
              f"  S/N {b['signal_over_noise_frob']:.2f}  share of expert-deviation energy {b['deviation_energy_frac']:.3f}"
              f"  share of pooled energy {b['pooled_energy_frac']:.3f}")
os.makedirs(os.path.join(ROOT, "sprint", "results"), exist_ok=True)
with open(os.path.join(ROOT, "sprint", "results", "diag_spectral.json"), "w") as f:
    json.dump(out, f, indent=1)
