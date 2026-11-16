"""Experiment 1 (pre-registered in sprint/PLAN.md): spectral / subspace expert condensation.

Arms, per layer, fitted on each split's fitting items with the J++ search (sprint/jppsearch.py):
  A0  J++: 8 expert weights, reference starts.
  A1  SB-J++ (primary): 32 weights over atoms P_b J_e, P_b = U_b U_b^T on the pooled LRP map's
      left singular bands [0,16) [16,64) [64,256) [256,1024); ridge lambda * |u - u_ref|^2 toward
      the split's own J++ solution u_ref = (w_J++ (x) 1_4) / 2; starts = u_ref + reference starts.
  A2  random-basis control: A1 with U replaced by a Haar-random orthogonal matrix (seed 1000 + l),
      same bands, lambda, starts. Fitted at the early layers only.
  A3  (secondary) 4 weights over pooled-band atoms P_b Jbar (spectral reweighting of the R-lens).
  A5  (secondary) label-free spectral denoising of each expert from the two prompt halves, then the
      plain J++ search.

Stages:
  --stage tune     seeds 0-5, early layers: A0, and A1 at each lambda in {0, 0.03, 0.1, 0.3, 1};
                   lambda* = best mean held-out early recall@10 (largest lambda within 0.2 points).
  --stage confirm  seeds 100-119: all arms (A3, A5 on the first 10 seeds), full M1-M3.
  --stage loto     leave-one-task-out (secondary): 47 fitting items stratified over the other four
                   macro evals + order-ops, scored on all items of the left-out eval; early layers.
  --stage verdict  pre-registered criteria -> verdict, RUNLOG row.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import LAYERS, MACRO, RESULTS, STRATA, Ctx, LensSpec, evaluate, runlog_append, save_json  # noqa: E402
from jppsearch import fit_weights, reference_starts, single_atom_scores  # noqa: E402
from sprintfit import combine, fit_rows  # noqa: E402

from jspace import paths as P  # noqa: E402

EDGES = [0, 16, 64, 256, 1024]
LAMBDAS = [0.0, 0.03, 0.1, 0.3, 1.0]
TUNE_SEEDS = [0, 1, 2, 3, 4, 5]
CONF_SEEDS = list(range(100, 120))
SECONDARY_SEEDS = CONF_SEEDS[:10]
EARLY = STRATA["early"]
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


def haar(d: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    Q, R = torch.linalg.qr(torch.randn(d, d, generator=g, dtype=torch.float64))
    return (Q * torch.sign(torch.diagonal(R))[None, :]).float()


class Bands:
    """Output-side band projectors P_b = U_b U_b^T of an orthogonal basis U (columns)."""

    def __init__(self, U: torch.Tensor, edges=EDGES):
        self.U, self.edges, self.B = U.float(), edges, len(edges) - 1

    def project(self, X: torch.Tensor) -> torch.Tensor:
        """X [..., d] -> [..., B, d] (P_b x for every band)."""
        outs = []
        for b in range(self.B):
            Ub = self.U[:, self.edges[b]:self.edges[b + 1]]
            outs.append((X @ Ub) @ Ub.T)
        return torch.stack(outs, dim=-2)

    def atom_transports(self, experts: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """[n, E*B, d]: P_b J_e h_n, expert-major order (atom a = e*B + b)."""
        X = torch.einsum("edk,nk->ned", experts.float(), H.float())        # [n, E, d]
        n, E, d = X.shape
        return self.project(X).reshape(n, E * self.B, d)

    def combine(self, experts: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        E = experts.shape[0]
        W = w.float().view(E, self.B)
        M = torch.zeros(experts.shape[1], experts.shape[2])
        for b in range(self.B):
            Ub = self.U[:, self.edges[b]:self.edges[b + 1]]
            Mb = torch.einsum("e,edk->dk", W[:, b], experts.float())
            M += Ub @ (Ub.T @ Mb)
        return M


def pooled_bands(ctx: Ctx, l: int) -> Bands:
    U, _, _ = torch.linalg.svd(ctx.pooled[l].float())
    return Bands(U)


def logits_of(ctx: Ctx, X: torch.Tensor) -> torch.Tensor:
    """Rank-equivalent pre-norm logits A x for transports X [n, A, d] -> [n, A, V]."""
    return X @ ctx.A.T


def fit_A0(ctx: Ctx, fit_idx, l: int):
    its, pairs = fit_rows(ctx, fit_idx)
    L = logits_of(ctx, torch.einsum("edk,nk->ned", ctx.experts[l].float(), ctx.prep.resid[its, l].float()))
    r = fit_weights(L, pairs, ctx.excl)
    del L
    return r, combine(ctx.experts[l], r.w)


class BandFitter:
    """A1/A2 fitting inputs for one (split, layer, basis), reusable across lambda values."""

    def __init__(self, ctx: Ctx, fit_idx, l: int, bands: Bands, w_jpp: torch.Tensor):
        self.ctx, self.l, self.bands = ctx, l, bands
        its, self.pairs = fit_rows(ctx, fit_idx)
        self.L = logits_of(ctx, bands.atom_transports(ctx.experts[l], ctx.prep.resid[its, l]))
        E = ctx.experts[l].shape[0]
        self.u_ref = (w_jpp.float()[:, None].expand(E, bands.B).reshape(-1)) / math.sqrt(bands.B)
        self.starts = {"jpp": self.u_ref, **reference_starts(single_atom_scores(self.L, self.pairs, ctx.excl))}

    def fit(self, lam: float):
        r = fit_weights(self.L, self.pairs, self.ctx.excl, starts=self.starts, lam=lam, w_ref=self.u_ref)
        return r, self.bands.combine(self.ctx.experts[self.l], r.w)

    def free(self):
        del self.L


def fit_A3(ctx: Ctx, fit_idx, l: int, bands: Bands):
    its, pairs = fit_rows(ctx, fit_idx)
    X = bands.project(ctx.prep.resid[its, l].float() @ ctx.pooled[l].float().T)       # [n, B, d]
    L = logits_of(ctx, X)
    r = fit_weights(L, pairs, ctx.excl)
    W = r.w.float()
    M = torch.zeros_like(ctx.pooled[l].float())
    for b in range(bands.B):
        Ub = bands.U[:, bands.edges[b]:bands.edges[b + 1]]
        M += W[b] * (Ub @ (Ub.T @ ctx.pooled[l].float()))
    return r, M


def denoised_experts(ctx: Ctx, l: int, bands: Bands, min_half: int = 25):
    """Label-free: J'_e = Jbar + sum_b kappa_{e,b} P_b (J_e - Jbar), kappa = max(0, 1 - noise/total), with
    noise = |P_b (D1 - D2)|^2 / 4 from the prompt halves (0-15 vs 16-31) and total = |P_b D|^2."""
    full = torch.load(os.path.join(P.LENSES, "lrp_experts_final.pt"), weights_only=True, mmap=True)
    half = torch.load(os.path.join(P.LENSES, "lrp_experts_n16.pt"), weights_only=True, mmap=True)
    m32, c32 = full["experts"][l].float(), full["counts"][l].float()
    m16, c16 = half["experts"][l].float(), half["counts"][l].float()
    s32, s16, c2 = m32 * c32[:, None, None], m16 * c16[:, None, None], c32 - c16
    h2 = (s32 - s16) / c2.clamp_min(1)[:, None, None]
    p1, p2 = s16.sum(0) / c16.sum(), (s32 - s16).sum(0) / c2.sum()
    Jbar = ctx.pooled[l].float()
    out, kap = [], []
    for e in range(m32.shape[0]):
        D = ctx.experts[l][e].float() - Jbar
        if c16[e] < min_half or c2[e] < min_half or float(D.norm()) == 0.0:
            out.append(ctx.experts[l][e].float())
            kap.append([1.0] * bands.B)
            continue
        N = m16[e] - p1 - (h2[e] - p2)                       # D1 - D2 (half-sample difference)
        ks = []
        Je = Jbar.clone()
        for b in range(bands.B):
            Ub = bands.U[:, bands.edges[b]:bands.edges[b + 1]]
            PDb = Ub @ (Ub.T @ D)                               # output-side band projection of D
            PNb = Ub @ (Ub.T @ N)
            total = float((PDb ** 2).sum())
            noise = float((PNb ** 2).sum()) / 4
            k = max(0.0, 1.0 - noise / total) if total > 0 else 1.0
            ks.append(k)
            Je += k * PDb
        out.append(Je)
        kap.append(ks)
    return torch.stack(out), kap


def fit_A5(ctx: Ctx, fit_idx, l: int, den: torch.Tensor):
    its, pairs = fit_rows(ctx, fit_idx)
    L = logits_of(ctx, torch.einsum("edk,nk->ned", den, ctx.prep.resid[its, l].float()))
    r = fit_weights(L, pairs, ctx.excl)
    del L
    return r, combine(den, r.w)


@torch.no_grad()
def recall_layers(ctx: Ctx, maps: dict, items, layers) -> dict:
    """Held-out recall@10 / MRR per layer (item-weighted, macro over the five macro evals) — the
    harness's M1 definition, for a subset of layers (used for tuning and LOTO)."""
    prep = ctx.prep
    items = [i for i in items if prep.items[i].eval in MACRO]
    pos = {i: j for j, i in enumerate(items)}
    rows = [(i, w) for i, w in ctx.rows if i in pos]
    out = {}
    for l in layers:
        Z = ctx.masked(ctx.unembed(prep.resid[items, l].float() @ maps[l].float().T))
        rec, mrr = {}, {}
        by_item = {}
        for i, w in rows:
            c = prep.cands[i][w]
            r = int((Z[pos[i]] > Z[pos[i], c].max()).sum()) + 1
            by_item.setdefault(i, []).append(r)
        for i, rs in by_item.items():
            ev = prep.items[i].eval
            rec.setdefault(ev, []).append(float(np.mean([r <= 10 for r in rs])))
            mrr.setdefault(ev, []).append(float(np.mean([1.0 / r for r in rs])))
        out[l] = {"recall10": float(np.mean([np.mean(rec[e]) for e in MACRO if e in rec])),
                  "mrr": float(np.mean([np.mean(mrr[e]) for e in MACRO if e in mrr])),
                  "per_eval": {e: float(np.mean(rec[e])) for e in MACRO if e in rec}}
    return out


@torch.no_grad()
def failure_cases(ctx: Ctx, ro_a: dict, ro_b: dict, maps_a: dict, maps_b: dict, names=("A0", "A1"), limit=12):
    """Held-out (item, intermediate) rows whose early-layer hit status differs between two lenses, with
    the top-5 readout tokens (Readout Filtering on) under each."""
    prep, tok = ctx.prep, ctx.model.tokenizer
    out = []
    for l in EARLY:
        for (i, w), ra, rb in zip(ro_a["rows"], ro_a["ranks"][l], ro_b["ranks"][l]):
            if prep.items[i].eval not in MACRO or (ra <= 10) == (rb <= 10):
                continue
            rec = {"layer": l, "item": i, "eval": prep.items[i].eval, "prompt": prep.items[i].prompt[-120:],
                   "intermediate": w, f"rank_{names[0]}": int(ra), f"rank_{names[1]}": int(rb)}
            for nm, maps in zip(names, (maps_a, maps_b)):
                z = ctx.masked(ctx.unembed(prep.resid[[i], l].float() @ maps[l].float().T))[0]
                rec[f"top5_{nm}"] = [tok.decode([int(t)]) for t in z.topk(5).indices]
            out.append(rec)
    out.sort(key=lambda r: -abs(math.log(r[f"rank_{names[0]}"]) - math.log(r[f"rank_{names[1]}"])))
    return out[:limit]


def load_json(name):
    p = os.path.join(RESULTS, name)
    return json.load(open(p)) if os.path.exists(p) else None


# ====================================================================== stages
def stage_tune(ctx: Ctx):
    res = load_json("exp1_tune.json") or {"lambdas": LAMBDAS, "layers": EARLY, "per_seed": {}}
    for seed in TUNE_SEEDS:
        if str(seed) in res["per_seed"]:
            continue
        fit_idx, held_idx = ctx.split(seed)
        row = {"A0": {}, "A1": {str(lam): {} for lam in LAMBDAS}}
        maps0, maps1 = {}, {str(lam): {} for lam in LAMBDAS}
        for l in EARLY:
            r0, maps0[l] = fit_A0(ctx, fit_idx, l)
            row["A0"][str(l)] = {"w": r0.w.tolist(), "train": r0.train_exact}
            bf = BandFitter(ctx, fit_idx, l, pooled_bands(ctx, l), r0.w)
            for lam in LAMBDAS:
                r1, maps1[str(lam)][l] = bf.fit(lam)
                row["A1"][str(lam)][str(l)] = {"train": r1.train_exact, "init": r1.init, "step": r1.step,
                                               "dist_to_ref": float(((r1.w - bf.u_ref) ** 2).sum())}
            bf.free()
            log(f"tune seed {seed} layer {l}: A0 train {r0.train_exact:.3f}; A1 train " +
                " ".join(f"{lam}:{row['A1'][str(lam)][str(l)]['train']:.3f}" for lam in LAMBDAS))
        rc0 = recall_layers(ctx, maps0, held_idx, EARLY)
        row["A0"]["held"] = {str(l): rc0[l]["recall10"] for l in EARLY}
        row["A0"]["held_early"] = float(np.mean([rc0[l]["recall10"] for l in EARLY]))
        for lam in LAMBDAS:
            rc = recall_layers(ctx, maps1[str(lam)], held_idx, EARLY)
            row["A1"][str(lam)]["held"] = {str(l): rc[l]["recall10"] for l in EARLY}
            row["A1"][str(lam)]["held_early"] = float(np.mean([rc[l]["recall10"] for l in EARLY]))
        log(f"tune seed {seed}: held early A0 {row['A0']['held_early']:.3f} | A1 " +
            " ".join(f"{lam}:{row['A1'][str(lam)]['held_early']:.3f}" for lam in LAMBDAS))
        res["per_seed"][str(seed)] = row
        save_json(res, "exp1_tune.json")
    means = {str(lam): float(np.mean([res["per_seed"][str(s)]["A1"][str(lam)]["held_early"] for s in TUNE_SEEDS]))
             for lam in LAMBDAS}
    a0 = float(np.mean([res["per_seed"][str(s)]["A0"]["held_early"] for s in TUNE_SEEDS]))
    best = max(means.values())
    lam_star = max(lam for lam in LAMBDAS if means[str(lam)] >= best - 0.002)   # pre-registered tie rule
    res["selection"] = {"mean_held_early_A1": means, "mean_held_early_A0": a0, "lambda_star": lam_star}
    save_json(res, "exp1_tune.json")
    log(f"lambda* = {lam_star} (means {means}; A0 {a0:.4f})")
    runlog_append("E1-tune", "select the A1 ridge lambda on tuning splits 0-5 (early layers), pre-registered rule",
                  f"A0 (J++) mean held-out early recall@10 {a0:.4f}",
                  "A1 by lambda: " + ", ".join(f"{k}: {v:.4f}" for k, v in means.items()),
                  "rule: best mean, largest lambda within 0.2 pt",
                  f"lambda* = {lam_star}; tuning splits only, not a test")
    return lam_star


def stage_confirm(ctx: Ctx, lam: float, seeds=CONF_SEEDS):
    res = load_json("exp1_confirm.json") or {"lambda_star": lam, "per_seed": {}}
    assert res["lambda_star"] == lam
    den_cache, kap_cache = {}, {}
    for seed in seeds:
        if str(seed) in res["per_seed"]:
            continue
        fit_idx, held_idx = ctx.split(seed)
        maps = {k: {} for k in ("A0", "A1", "A2", "A3", "A5")}
        wl = {k: {} for k in maps}
        secondary = seed in SECONDARY_SEEDS
        for l in LAYERS:
            r0, maps["A0"][l] = fit_A0(ctx, fit_idx, l)
            wl["A0"][l] = {"w": r0.w.tolist(), "train": r0.train_exact}
            bands = pooled_bands(ctx, l)
            bf = BandFitter(ctx, fit_idx, l, bands, r0.w)
            r1, maps["A1"][l] = bf.fit(lam)
            bf.free()
            wl["A1"][l] = {"w": r1.w.tolist(), "train": r1.train_exact, "init": r1.init,
                           "dist_to_ref": float(((r1.w - bf.u_ref) ** 2).sum())}
            if l in EARLY:
                bf2 = BandFitter(ctx, fit_idx, l, Bands(haar(ctx.A.shape[1], 1000 + l)), r0.w)
                r2, maps["A2"][l] = bf2.fit(lam)
                bf2.free()
                wl["A2"][l] = {"w": r2.w.tolist(), "train": r2.train_exact, "init": r2.init,
                               "dist_to_ref": float(((r2.w - bf2.u_ref) ** 2).sum())}
            else:
                maps["A2"][l] = maps["A0"][l]          # A2 is fitted at the early layers only
            if secondary:
                r3, maps["A3"][l] = fit_A3(ctx, fit_idx, l, bands)
                wl["A3"][l] = {"w": r3.w.tolist(), "train": r3.train_exact}
                if l not in den_cache:
                    den_cache[l], kap_cache[l] = denoised_experts(ctx, l, bands)
                r5, maps["A5"][l] = fit_A5(ctx, fit_idx, l, den_cache[l])
                wl["A5"][l] = {"w": r5.w.tolist(), "train": r5.train_exact}
            log(f"seed {seed} layer {l}: train A0 {r0.train_exact:.3f} A1 {r1.train_exact:.3f}" +
                (f" A2 {wl['A2'][l]['train']:.3f}" if l in EARLY else "") +
                (f" A3 {wl['A3'][l]['train']:.3f} A5 {wl['A5'][l]['train']:.3f}" if secondary else ""))
        arms = ["A0", "A1", "A2"] + (["A3", "A5"] if secondary else [])
        summ, ros = {}, {}
        for arm in arms:
            summ[arm], ros[arm] = evaluate(ctx, LensSpec(arm, maps=maps[arm]), held_idx, seed, wiki=True,
                                           transport=(arm != "A2"))
        fit_summ = {arm: evaluate(ctx, LensSpec(arm, maps=maps[arm]), fit_idx, seed, wiki=False, transport=False)[0]
                    for arm in ("A0", "A1")}
        fails = failure_cases(ctx, ros["A0"], ros["A1"], maps["A0"], maps["A1"])
        res["per_seed"][str(seed)] = {
            "results": summ, "fit_split": {a: {s: fit_summ[a][s]["recall10"] for s in STRATA} for a in fit_summ},
            "weights": {a: {str(l): v for l, v in d.items()} for a, d in wl.items() if d},
            "failure_cases_A1_vs_A0": fails, "n_fit": len(fit_idx), "n_held": len(held_idx)}
        if secondary and "kappa_A5" not in res:
            res["kappa_A5"] = {str(l): kap_cache[l] for l in kap_cache}
        log(f"seed {seed}: early recall " + " ".join(f"{a} {summ[a]['early']['recall10']:.3f}" for a in arms) +
            f" | d(A1-A0) {summ['A1']['early']['recall10'] - summ['A0']['early']['recall10']:+.3f}")
        if seed == seeds[0]:
            torch.save({a: {l: m.half() for l, m in maps[a].items()} for a in ("A0", "A1")},
                       os.path.join(RESULTS, f"maps_exp1_seed{seed}.pt"))
        save_json(res, "exp1_confirm.json")
    return res


def stage_loto(ctx: Ctx, lam: float):
    res = load_json("exp1_loto.json") or {"lambda_star": lam, "folds": {}}
    prep = ctx.prep
    scored = [i for i in range(len(prep.items)) if ctx.keep[i]]
    for k, ev in enumerate(MACRO):
        if ev in res["folds"]:
            continue
        rng = np.random.default_rng(200 + k)
        others = [i for i in scored if prep.items[i].eval != ev]
        by_ev = {}
        for i in others:
            by_ev.setdefault(prep.items[i].eval, []).append(i)
        # 47 fitting items, stratified over the other evals in proportion to their sizes
        tot = len(others)
        fit_idx = []
        for e in sorted(by_ev):
            n_e = max(1, round(47 * len(by_ev[e]) / tot))
            fit_idx += [by_ev[e][j] for j in rng.permutation(len(by_ev[e]))[:n_e]]
        test = [i for i in scored if prep.items[i].eval == ev]
        m0, m1 = {}, {}
        for l in EARLY:
            r0, m0[l] = fit_A0(ctx, fit_idx, l)
            bf = BandFitter(ctx, fit_idx, l, pooled_bands(ctx, l), r0.w)
            _, m1[l] = bf.fit(lam)
            bf.free()
        rc0, rc1 = recall_layers(ctx, m0, test, EARLY), recall_layers(ctx, m1, test, EARLY)
        res["folds"][ev] = {"n_fit": len(fit_idx), "n_test": len(test),
                            "A0": {str(l): rc0[l]["recall10"] for l in EARLY},
                            "A1": {str(l): rc1[l]["recall10"] for l in EARLY},
                            "A0_early": float(np.mean([rc0[l]["recall10"] for l in EARLY])),
                            "A1_early": float(np.mean([rc1[l]["recall10"] for l in EARLY]))}
        log(f"LOTO {ev}: A0 {res['folds'][ev]['A0_early']:.3f} A1 {res['folds'][ev]['A1_early']:.3f}")
        save_json(res, "exp1_loto.json")
    return res


def stage_verdict():
    res = load_json("exp1_confirm.json")
    seeds = [s for s in CONF_SEEDS if str(s) in res["per_seed"]]
    R = {s: res["per_seed"][str(s)]["results"] for s in seeds}

    def d(a, b, stratum, key):
        if stratum == "best7":
            return [R[s][a]["best7_recall10"] - R[s][b]["best7_recall10"] for s in seeds]
        return [R[s][a][stratum][key] - R[s][b][stratum][key] for s in seeds]

    def stat(x):
        x = np.array(x)
        return {"mean": float(x.mean()), "min": float(x.min()), "max": float(x.max()),
                "wins": int((x > 0).sum()), "ties": int((x == 0).sum()), "losses": int((x < 0).sum()), "n": len(x)}
    out = {"n_splits": len(seeds), "lambda_star": res["lambda_star"]}
    prim = stat(d("A1", "A0", "early", "recall10"))
    fpm = stat(d("A1", "A0", "early", "fp10_matched"))
    fpw = stat(d("A1", "A0", "early", "fp10_wiki"))
    spec = stat(d("A1", "A2", "early", "recall10"))
    out.update({"primary_early_recall_A1_minus_A0": prim, "early_fp10_matched_A1_minus_A0": fpm,
                "early_fp10_wiki_A1_minus_A0": fpw, "early_recall_A1_minus_A2": spec,
                "early_recall_A2_minus_A0": stat(d("A2", "A0", "early", "recall10"))})
    c1 = prim["mean"] >= 0.03
    c2 = prim["wins"] >= 16
    c3 = fpm["mean"] <= 0.005 and fpw["mean"] <= 0.005
    c4 = spec["mean"] >= 0.015 and spec["wins"] >= 14
    refute = prim["mean"] < 0.01 or prim["wins"] <= 12 or fpm["mean"] > 0.01
    verdict = ("CONFIRMED" if (c1 and c2 and c3 and c4) else "REFUTED" if refute else
               "INCONCLUSIVE (parameter-count effect, not spectral)" if (c1 and c2 and c3) else "INCONCLUSIVE")
    out["criteria"] = {"C1_mean_ge_3pt": c1, "C2_wins_ge_16": c2, "C3_no_fp_increase": c3, "C4_spectral_specific": c4,
                       "refute_condition": refute}
    out["verdict"] = verdict
    guard = {"mid_recall": stat(d("A1", "A0", "mid", "recall10")), "late_recall": stat(d("A1", "A0", "late", "recall10")),
             "early_mrr": stat(d("A1", "A0", "early", "mrr")), "early_tr_cos": stat(d("A1", "A0", "early", "tr_cos")),
             "early_tr_nmse": stat(d("A1", "A0", "early", "tr_nmse")), "best7": stat(d("A1", "A0", "best7", None))}
    out["guardrails"] = guard
    out["guardrail_flags"] = {"mid_ge_-1pt": guard["mid_recall"]["mean"] >= -0.01,
                              "late_ge_-1pt": guard["late_recall"]["mean"] >= -0.01,
                              "mrr_same_sign": np.sign(guard["early_mrr"]["mean"]) == np.sign(prim["mean"]),
                              "cos_ge_-0.02": guard["early_tr_cos"]["mean"] >= -0.02,
                              "nmse_le_+0.02": guard["early_tr_nmse"]["mean"] <= 0.02}
    out["guardrail_flags"] = {k: bool(v) for k, v in out["guardrail_flags"].items()}
    sec = [s for s in SECONDARY_SEEDS if str(s) in res["per_seed"] and "A3" in R[s]]
    out["secondary"] = {a: {"early": stat([R[s][a]["early"]["recall10"] - R[s]["A0"]["early"]["recall10"] for s in sec]),
                            "best7": stat([R[s][a]["best7_recall10"] - R[s]["A0"]["best7_recall10"] for s in sec])}
                        for a in ("A3", "A5")} if sec else {}
    fit_gap = {a: float(np.mean([res["per_seed"][str(s)]["fit_split"][a]["early"] - R[s][a]["early"]["recall10"]
                                 for s in seeds])) for a in ("A0", "A1")}
    out["fit_minus_heldout_early"] = fit_gap
    means = {a: {k: float(np.mean([R[s][a]["early"][k] for s in seeds if a in R[s]]))
                 for k in ("recall10", "mrr", "fp10_matched", "fp10_wiki", "auc_matched")} for a in ("A0", "A1", "A2")}
    out["means_early"] = means
    save_json(out, "exp1_verdict.json")
    print(json.dumps(out, indent=1))
    runlog_append("E1-confirm", "H1: per-band expert weights (pooled-SVD bands, ridge to J++) raise held-out early recall@10 by >3 pt without more FPs",
                  f"A0 (J++) early recall@10 {means['A0']['recall10']:.4f}, FP@10 matched {means['A0']['fp10_matched']:.4f}",
                  f"A1 {means['A1']['recall10']:.4f}; d = {prim['mean']:+.4f} [{prim['min']:+.4f},{prim['max']:+.4f}], "
                  f"W/T/L {prim['wins']}/{prim['ties']}/{prim['losses']}; dFP@10 matched {fpm['mean']:+.4f}, wiki {fpw['mean']:+.4f}",
                  f"A2 random basis {means['A2']['recall10']:.4f}; A1-A2 {spec['mean']:+.4f} (W {spec['wins']}/{spec['n']}); "
                  f"guardrails {out['guardrail_flags']}",
                  f"{verdict} (lambda*={res['lambda_star']}, {len(seeds)} fresh splits)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["tune", "confirm", "loto", "verdict"])
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    if a.stage == "verdict":
        stage_verdict()
        sys.exit(0)
    ctx = Ctx(threads=a.threads)
    if a.stage == "tune":
        stage_tune(ctx)
    else:
        tune = load_json("exp1_tune.json")
        lam = tune["selection"]["lambda_star"]
        stage_confirm(ctx, lam) if a.stage == "confirm" else stage_loto(ctx, lam)
    log("done")
