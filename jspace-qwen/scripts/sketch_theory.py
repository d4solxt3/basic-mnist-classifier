"""Sketch estimators: replay check, variance theory and Monte Carlo, from the
per-prompt exact Jacobians (prompts 0..31, layers 7/14/21; lenses/exact_per_prompt.pt).

Every sketch estimate is linear in the per-prompt Jacobian J_n, because
G = (WQ)^T J_n. Given J_n and the probe frames we can therefore rebuild any
sketch estimator offline, without running the model.

1. Replay. Rebuild the online-control-variate snapshots sk16_n16/n32 and
   sk16w_n16/n32 from the stored J_n and the same Haar frames (seed 1). Compare
   them with the snapshots that fit.py saved. Agreement means the divergence is
   a property of the estimator and not a bug in the VJP code.
2. Theory (W = I, Frobenius norm; Proposition 3 of the paper). Let v_n =
   E||Jbar_n - J||^2 be the running-mean error and c = d/K - 1. Then
       v_1     = (c+1) s2 + c ||J - I||^2
       v_{n+1} = ((n^2 + c) v_n + (c+1) s2) / (n+1)^2,
   where s2 is the per-prompt variance. v_n grows while n < (c-1)/2.
3. Monte Carlo. Prompts are resampled from the 32 and the estimators are run
   over several seeds. The truth is the 32-prompt mean. Recorded per estimator:
   readout relative error ||A (Jhat - J)|| / ||A J|| and Frobenius error, at
   matched VJP rows. The exact estimator (d rows per prompt) is included.

python scripts/sketch_theory.py --seeds 16 --steps 512
"""

import argparse
import json
import math
import os
import sys

import torch

ROOT = __file__.rsplit("/scripts/", 1)[0]
sys.path.insert(0, ROOT)
from jspace import paths as P  # noqa: E402
from jspace.fitting import haar_frame, load_lens_state, readout_shaping  # noqa: E402
from jspace.model import load_model  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=16)
ap.add_argument("--steps", type=int, default=512)
ap.add_argument("--K", type=int, default=16)
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--skip-replay", action="store_true")
a = ap.parse_args()
torch.set_num_threads(a.threads)

model = load_model("Qwen/Qwen3-0.6B")
A = model.unembed_matrix()
W, Winv = readout_shaping(A)
AtA = (A.double().T @ A.double()).float()
del A, model

PP = torch.load(os.path.join(P.LENSES, "exact_per_prompt.pt"), weights_only=False)
layers = sorted(PP)
Nprom, d = PP[layers[0]].shape[0], PP[layers[0]].shape[1]
K = a.K
c = d / K - 1
eye = torch.eye(d)
Jpop = {l: PP[l].float().mean(0) for l in layers}


def rnorm2(X):
    """||A X||_F^2 via A^T A."""
    return float((X * (AtA @ X)).sum())


# ---------------------------------------------------------------- statistics of J_n
stats = {}
for l in layers:
    J = Jpop[l]
    s2 = float(sum(((PP[l][n].float() - J) ** 2).sum() for n in range(Nprom)) / (Nprom - 1))
    s2A = sum(rnorm2(PP[l][n].float() - J) for n in range(Nprom)) / (Nprom - 1)
    S = sum((PP[l][n].float() - eye) @ (PP[l][n].float() - eye).T for n in range(Nprom)) / Nprom
    Z = W @ W
    Zinv = Winv @ Winv
    rho_data = float(torch.trace(Zinv @ AtA) * torch.trace(Z @ S) / (torch.trace(AtA) * torch.trace(S)))
    stats[l] = {
        "sigma2": s2, "sigma2_readout": s2A,
        "J_minus_I2": float(((J - eye) ** 2).sum()), "J2": float((J ** 2).sum()),
        "J_minus_I2_readout": rnorm2(J - eye), "J2_readout": rnorm2(J),
        "rho_data_identity_cv": rho_data,
        # equal-VJP-cost variance ratio sketch/exact (Frobenius): 1 + (1 - K/d) ||J - J0||^2 / sigma^2
        "equal_cost_ratio_identity": 1 + (1 - K / d) * float(((J - eye) ** 2).sum()) / s2,
        "equal_cost_ratio_none": 1 + (1 - K / d) * float((J ** 2).sum()) / s2,
        "critical_n": (c - 1) / 2,
    }
    print(f"layer {l}: " + " ".join(f"{k}={v:.4g}" for k, v in stats[l].items()), flush=True)


def theory_online(l, N):
    s = stats[l]
    v = (c + 1) * s["sigma2"] + c * s["J_minus_I2"]
    out = [v]
    for n in range(1, N):
        v = ((n * n + c) * v + (c + 1) * s["sigma2"]) / (n + 1) ** 2
        out.append(v)
    return out          # out[n-1] = v_n


def theory_fixed(l, N, J0):
    s = stats[l]
    key = {"identity": "J_minus_I2", "none": "J2"}[J0]
    per = s["sigma2"] + c * (s[key] + s["sigma2"])
    return [per / n for n in range(1, N + 1)]


# ---------------------------------------------------------------- estimator replay
def run(seq, cv, shaped, seed, snaps, burnin=0, kappa=None):
    """Run one sketch estimator over the prompt index sequence; return snapshots {n: {l: J}}."""
    gen = torch.Generator().manual_seed(seed)
    sums = {l: torch.zeros(d, d) for l in layers}
    out, n = {}, 0
    for idx in seq:
        Q = haar_frame(d, K, gen)
        C = Q if not shaped else W @ Q
        B = (d / K) * (Q if not shaped else Winv @ Q)      # Winv (d/K) Q, applied to the residual
        for l in layers:
            Jn = PP[l][idx].float()
            if cv == "none":
                J0 = None
            elif cv == "identity" or n == 0 or n < burnin:
                J0 = eye
            elif cv == "online":
                J0 = sums[l] / n
            elif cv == "shrink":            # J0 = I + lambda_n (Jbar_n - I), lambda_n = n / (n + kappa)
                lam = n / (n + kappa)
                J0 = eye + lam * (sums[l] / n - eye)
            else:
                raise ValueError(cv)
            G = C.T @ Jn
            resid = G if J0 is None else G - C.T @ J0
            upd = B @ resid
            sums[l] += upd if J0 is None else J0 + upd
        n += 1
        if n in snaps:
            out[n] = {l: sums[l] / n for l in layers}
    return out


res = {"d": d, "K": K, "n_prompts_population": Nprom, "stats": {str(l): v for l, v in stats.items()}}

# 1. replay check against the saved snapshots (prompts 0..31 in order, seed 1)
RUNS = {"sk16": dict(cv="online", shaped=False), "sk16w": dict(cv="online", shaped=True),
        "sk16id": dict(cv="identity", shaped=False), "sk16none": dict(cv="none", shaped=False),
        "sk16idw": dict(cv="identity", shaped=True), "sk16sh": dict(cv="shrink", shaped=False, kappa=d / K)}
if not a.skip_replay:
    rep = {}
    for tag, cfg in RUNS.items():
        if not os.path.exists(os.path.join(P.LENSES, f"{tag}_n32.pt")):
            continue
        snaps = run(list(range(32)), seed=1, snaps={16, 32}, **cfg)
        for n in (16, 32):
            saved = load_lens_state(os.path.join(P.LENSES, f"{tag}_n{n}.pt"))["J"]
            rep[f"{tag}_n{n}"] = {
                str(l): {"rel_diff_readout": math.sqrt(rnorm2(snaps[n][l] - saved[l]) / rnorm2(saved[l])),
                         "replay_err_vs_pop": math.sqrt(rnorm2(snaps[n][l] - Jpop[l]) / rnorm2(Jpop[l])),
                         "saved_err_vs_pop": math.sqrt(rnorm2(saved[l] - Jpop[l]) / rnorm2(Jpop[l]))}
                for l in layers}
            print(tag, n, {l: round(v["rel_diff_readout"], 4) for l, v in rep[f"{tag}_n{n}"].items()}, flush=True)
    res["replay"] = rep

# 4. Real-run sketch noise against its exact expectation. For a fixed J0 and the actual prompts
#    0..n-1, the sketch noise Jhat - mean_i J_i = (1/n) sum_i (a W^-1 P_i W - I) M_i, M_i = J_i - J0,
#    a = d/K, P_i a Haar rank-K projection, has (exactly, with E[P S P] = c1 S + c2 tr(S) I)
#      E||A noise||^2 = (1/n^2) sum_i [a^2 c1 ||A M_i||^2 + a^2 c2 tr(Z^-1 A^T A) ||W M_i||^2 - ||A M_i||^2],
#    c2 = K(d-K) / (d(d-1)(d+2)),  c1 = K(K+2) / (d(d+2)) - c2,  Z = W^2 (W = I unshaped).
c2 = K * (d - K) / (d * (d - 1) * (d + 2))
c1 = K * (K + 2) / (d * (d + 2)) - c2
a_ = d / K
trG = float(torch.trace(AtA))
trGZinv = float(torch.trace(AtA @ (Winv @ Winv)))
noise = {}
for tag, cfg in RUNS.items():
    if cfg["cv"] not in ("identity", "none"):
        continue
    for n in (16, 32):
        path = os.path.join(P.LENSES, f"{tag}_n{n}.pt")
        if not os.path.exists(path):
            continue
        saved = load_lens_state(path)["J"]
        row = {}
        for l in layers:
            Jbar = PP[l][:n].float().mean(0)
            pred = 0.0
            predF = 0.0
            for i in range(n):
                M = PP[l][i].float() - (eye if cfg["cv"] == "identity" else 0)
                AM = rnorm2(M)
                WM = float(((W @ M) ** 2).sum()) if cfg["shaped"] else float((M ** 2).sum())
                pred += a_ ** 2 * c1 * AM + a_ ** 2 * c2 * (trGZinv if cfg["shaped"] else trG) * WM - AM
                predF += (a_ - 1) * float((M ** 2).sum())
            pred /= n * n
            predF /= n * n
            base = rnorm2(Jbar)
            meas = rnorm2(saved[l] - Jbar)
            row[str(l)] = {"pred_readout_rel": math.sqrt(pred / base), "meas_readout_rel": math.sqrt(meas / base),
                           "pred_frob": predF, "meas_frob": float(((saved[l] - Jbar) ** 2).sum())}
        noise[f"{tag}_n{n}"] = row
        print("noise", tag, n, {l: (round(v["pred_readout_rel"], 3), round(v["meas_readout_rel"], 3))
                                for l, v in row.items()}, flush=True)
res["noise_check"] = noise

# 5. Are consecutive corpus records correlated? Mean squared distance between per-prompt Jacobians
#    for adjacent records (|i-j| = 1) vs records at least 8 apart; and the half-split error of the
#    exact lens (records 0..15 vs 16..31) against its iid prediction sqrt(2 sigma^2 / 16) / ||J||.
clus = {}
for l in layers:
    X = PP[l].float().reshape(Nprom, -1)
    sq = (X * X).sum(1)
    D = (sq[:, None] + sq[None, :] - 2 * X @ X.T).clamp_min(0)
    idx = torch.arange(Nprom)
    gap = (idx[:, None] - idx[None, :]).abs()
    adj = float(D[gap == 1].mean())
    far = float(D[gap >= 8].mean())
    half = float((PP[l][:16].float().mean(0) - PP[l][16:].float().mean(0)).norm() / Jpop[l].norm())
    iid = math.sqrt(2 * stats[l]["sigma2"] / 16) / float(Jpop[l].norm())
    clus[str(l)] = {"adjacent_over_far_sqdist": adj / far, "half_split_rel": half, "half_split_iid_pred": iid}
    print("cluster", l, clus[str(l)], flush=True)
del X, D
res["clustering"] = clus

# 2-3. Monte Carlo over resampled prompts
snaps = [2 ** i for i in range(0, int(math.log2(a.steps)) + 1)]
configs = {
    "none": dict(cv="none", shaped=False),
    "identity": dict(cv="identity", shaped=False),
    "identity_shaped": dict(cv="identity", shaped=True),
    "online": dict(cv="online", shaped=False),
    "online_shaped": dict(cv="online", shaped=True),
    "online_burnin64": dict(cv="online", shaped=False, burnin=64),
    "online_burnin128": dict(cv="online", shaped=False, burnin=128),
    "shrink64": dict(cv="shrink", shaped=False, kappa=64),
}
mc = {name: {str(n): {"readout": [], "frob2": []} for n in snaps} for name in configs}
mc["exact"] = {str(n): {"readout": [], "frob2": []} for n in [1, 2, 4, 8, 16, 32]}
for seed in range(a.seeds):   # --seeds 0 runs only parts 1, 4 and 5
    g = torch.Generator().manual_seed(1000 + seed)
    seq = torch.randint(0, Nprom, (a.steps,), generator=g).tolist()
    for name, cfg in configs.items():
        out = run(seq, seed=100 + seed, snaps=set(snaps), **cfg)
        for n, Js in out.items():
            mc[name][str(n)]["readout"].append(
                {str(l): math.sqrt(rnorm2(Js[l] - Jpop[l]) / rnorm2(Jpop[l])) for l in layers})
            mc[name][str(n)]["frob2"].append({str(l): float(((Js[l] - Jpop[l]) ** 2).sum()) for l in layers})
    for n in [1, 2, 4, 8, 16, 32]:
        Js = {l: sum(PP[l][i].float() for i in seq[:n]) / n for l in layers}
        mc["exact"][str(n)]["readout"].append(
            {str(l): math.sqrt(rnorm2(Js[l] - Jpop[l]) / rnorm2(Jpop[l])) for l in layers})
        mc["exact"][str(n)]["frob2"].append({str(l): float(((Js[l] - Jpop[l]) ** 2).sum()) for l in layers})
    print(f"seed {seed} done", flush=True)


def summarise(rows, key):
    """Median over seeds of the readout error; mean over seeds of the Frobenius error^2."""
    out = {}
    for l in layers:
        v = sorted(r[str(l)] for r in rows[key])
        out[str(l)] = v[len(v) // 2] if key == "readout" else sum(v) / len(v)
    return out


summary = {}
for name, by_n in (mc.items() if a.seeds > 0 else []):
    rows_per = d if name == "exact" else K
    summary[name] = {n: {"vjp_rows": int(n) * rows_per,
                         "readout_median": summarise(r, "readout"),
                         "frob2_mean": summarise(r, "frob2")} for n, r in by_n.items()}
if a.seeds > 0:
    res["mc"] = summary
res["theory"] = {str(l): {"online_v": {str(n): theory_online(l, a.steps)[n - 1] for n in snaps},
                          "identity_v": {str(n): theory_fixed(l, a.steps, "identity")[n - 1] for n in snaps},
                          "none_v": {str(n): theory_fixed(l, a.steps, "none")[n - 1] for n in snaps}}
                 for l in layers}
res["mc_seeds"] = a.seeds
os.makedirs(P.RESULTS, exist_ok=True)
if a.seeds == 0:
    with open(os.path.join(P.RESULTS, "sketch_theory_partial.json"), "w") as f:
        json.dump(res, f, indent=1)
    sys.exit(0)
with open(os.path.join(P.RESULTS, "sketch_theory.json"), "w") as f:
    json.dump(res, f, indent=1)

# compact printout: theory vs Monte Carlo (Frobenius), online control variate
for l in layers:
    print(f"\nlayer {l}  n: theory_v  mc_frob2 (online) | identity theory/mc")
    for n in snaps:
        print(f"  {n:4d}: {res['theory'][str(l)]['online_v'][str(n)]:.4g}  "
              f"{summary['online'][str(n)]['frob2_mean'][str(l)]:.4g} | "
              f"{res['theory'][str(l)]['identity_v'][str(n)]:.4g}  "
              f"{summary['identity'][str(n)]['frob2_mean'][str(l)]:.4g}")
