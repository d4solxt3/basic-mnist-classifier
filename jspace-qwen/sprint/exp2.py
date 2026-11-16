"""Experiment 2 (pre-registered in sprint/prereg_exp2.md): self-supervised KL alignment and an
anti-false-positive term for J++'s expert weights.

Objective of arm B2: Surr(w) - gamma * FalseSurr(w) - beta * KL(w), on J++'s 8 LRP experts per layer.
  Surr       J++'s sigmoid-margin surrogate of pair-level macro pass@10 (fitting pairs)
  FalseSurr  the same surrogate on matched-control pairs inside the fitting items (false hits)
  KL         mean KL(softmax(unembed(h_27)) || softmax(unembed(J(w) h_l))) at 256 WikiText positions
Arms: B0 J++ (= Exp 1's A0 on the same seeds), B1 KL-only (label-free), B2 primary. The secondary arms
B3 (Surr - beta*KL) and B4 (Surr - gamma*FalseSurr) were dropped by Amendment 2 to save compute. Search: J++'s Adam (200 steps, lr 0.05, exact score every 5 steps, starts
uniform / best single expert / top-3), surrogate gradients analytic, KL gradient by autograd.

Stages: --stage tune (beta on seeds 0-2), confirm (seeds 100-109), loto (20 fold-replicates), verdict.
The loops are layer-major so each layer's KL inputs (1.2 GB) are built once.
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
from jppsearch import Pairs, fit_weights, macro, reference_starts, single_atom_scores  # noqa: E402
from sprintfit import combine, fit_rows  # noqa: E402

from jspace import paths as P  # noqa: E402

BETAS = [0.03, 0.1, 0.3]
ARMS = ("B0", "B1", "B2")          # B3/B4 dropped by Amendment 2 (compute)
GAMMA = 1.0
N_KL = 256
TUNE_SEEDS = [0, 1, 2]
CONF_SEEDS = list(range(100, 110))
LOTO_REPS = 4
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)


# ---------------------------------------------------------------------- KL term
class KLTerm:
    """KL(p || q_w) at N WikiText positions, with an autograd gradient.

    Exact decomposition: KL_n = sum_v p log p - (u . m_n) / r_n + logsumexp_v z_nv, where
    z_nv = (u . L_n[:, v]) / r_n, r_n = rms of the combined transport, and m_n = sum_v p_nv L_n[:, v]
    (computed once over the full vocabulary). Only the log-partition needs every token. During the search
    it is estimated from each position's K tokens with the largest expert-logit norm plus R tail tokens
    sampled once per position (seed 0) and weighted by |tail| / R. Validation (sprint/test_exp2_kl2.py) gave
    errors <= 0.004 nats at the J++, uniform and R-lens weightings and <= 0.04 nats over 16 test vectors
    with R = 8192. ``exact`` gives the full-vocabulary KL used for every reported number.
    """

    def __init__(self, ctx: Ctx, l: int, H: torch.Tensor, Y: torch.Tensor, K: int = 8192, R: int = 16384):
        E = ctx.experts[l]
        T = torch.einsum("edk,nk->ned", E.float(), H.float())          # [N, E, d]
        self.G = T @ T.transpose(1, 2)                                  # [N, E, E]
        Lfull = T @ ctx.A.T                                             # [N, E, V] pre-norm logits
        self.d, self.eps = T.shape[-1], ctx.eps
        N, Ed, V = Lfull.shape
        with torch.no_grad():
            logp = torch.log_softmax(ctx.unembed(Y.float()), dim=-1)
            p = logp.exp()
            self.negH = (p * logp).sum(-1)                             # [N] sum p log p
            self.m = torch.einsum("nv,nev->ne", p, Lfull)               # [N, E] exact over the vocabulary
            top = Lfull.norm(dim=1).topk(K, dim=-1).indices             # [N, K]
            mask = torch.ones(N, V, dtype=torch.bool)
            mask.scatter_(1, top, False)
            g = torch.Generator().manual_seed(0)
            tail = torch.stack([torch.nonzero(mask[n])[:, 0][torch.randperm(V - K, generator=g)[:R]] for n in range(N)])
            idx = torch.cat([top, tail], dim=1)                         # [N, K + R]
            self.L = Lfull.gather(2, idx[:, None, :].expand(-1, Ed, -1)).contiguous()
            self.logw = torch.cat([torch.zeros(K), torch.full((R,), math.log((V - K) / R))])   # [K + R]
        self.Lfull = Lfull
        self.Hp = float(self.negH.mean())

    def _kl(self, u: torch.Tensor, L: torch.Tensor, logw: torch.Tensor | None) -> torch.Tensor:
        r = (torch.einsum("e,nef,f->n", u, self.G, u) / self.d + self.eps).sqrt()
        z = torch.einsum("e,nev->nv", u, L) / r[:, None]
        if logw is not None:
            z = z + logw[None, :]
        return (self.negH - (self.m @ u) / r + torch.logsumexp(z, dim=-1)).mean()

    def value_grad(self, u: torch.Tensor) -> tuple[float, torch.Tensor]:
        u = u.detach().clone().requires_grad_(True)
        kl = self._kl(u, self.L, self.logw)
        (g,) = torch.autograd.grad(kl, u)
        return float(kl.detach()), g

    def value(self, u: torch.Tensor) -> float:
        with torch.no_grad():
            return float(self._kl(u.float(), self.L, self.logw))

    def exact(self, u: torch.Tensor) -> float:
        with torch.no_grad():
            return float(self._kl(u.float(), self.Lfull, None))

    def validate(self, us: list[torch.Tensor]) -> list[tuple[float, float]]:
        return [(self.value(u), self.exact(u)) for u in us]


def kl_positions(ctx: Ctx, l: int) -> tuple[torch.Tensor, torch.Tensor]:
    cal = torch.load(os.path.join(P.LENSES, "corpus_acts.pt"), weights_only=False, mmap=True)
    g = torch.Generator().manual_seed(11)
    idx = torch.randperm(cal["target"].shape[0], generator=g)[:N_KL]
    return cal["acts"][l][idx].float(), cal["target"][idx].float()


# ---------------------------------------------------------------------- pair tables
def control_pairs(ctx: Ctx, its: list[int], seed: int) -> Pairs | None:
    """Matched-control pairs inside the fitting items: (control row j, candidates of (i, x), eval of i)."""
    prep = ctx.prep
    pos = {i: k for k, i in enumerate(its)}
    rng = np.random.default_rng(20_000 + seed)
    rows_item, rows_cands, rows_eval = [], [], []
    by_ev = {}
    for i in its:
        by_ev.setdefault(prep.items[i].eval, []).append(i)
    for ev, ids in by_ev.items():
        if ev not in MACRO:
            continue
        for i in ids:
            ci = {t for v in prep.cands[i].values() for t in v}
            words = [w.lower() for w in prep.cands[i]]
            ok = [j for j in ids if j != i and not (ci & {t for v in prep.cands[j].values() for t in v})
                  and not any(w in prep.items[j].prompt.lower() for w in words)]
            rng.shuffle(ok)
            for j in ok[:5]:
                for x, c in prep.cands[i].items():
                    rows_item.append(pos[j])
                    rows_cands.append(c)
                    rows_eval.append(ev)
    return Pairs.build(rows_item, rows_cands, rows_eval) if rows_item else None


# ---------------------------------------------------------------------- the search
def _surr_terms(z, zm_topk, pairs: Pairs, spread, L, temperature=0.1):
    """Surrogate value, exact hit rate and analytic gradient (w.r.t. u) for one pair table."""
    kth_val, kth_id = zm_topk
    cz = z[pairs.item].gather(1, pairs.cands)
    bpos = cz.argmax(1)
    best_val = cz.gather(1, bpos[:, None])[:, 0]
    best_id = pairs.cands.gather(1, bpos[:, None])[:, 0]
    margin = best_val - kth_val[pairs.item]
    sq = spread[pairs.item]
    s = torch.sigmoid(margin / (temperature * sq))
    gw = pairs.group_weights()
    c = gw * s * (1 - s) / (temperature * sq)
    D = L[pairs.item, :, best_id] - L[pairs.item, :, kth_id[pairs.item]]
    return float((s * gw).sum()), macro((margin >= 0).float(), pairs), (c[:, None] * D).sum(0)


def fit_multi(L: torch.Tensor, hit: Pairs, excl: torch.Tensor, *, false: Pairs | None = None, gamma: float = 0.0,
              kl: KLTerm | None = None, beta: float = 0.0, starts=None, k=10, steps=200, lr=0.05,
              exact_every=5, top_n=3):
    """Maximise Surr - gamma*FalseSurr - beta*KL over unit-L2 weights; return the iterate with the best
    (exact_hit - gamma*exact_false - beta*KL, surrogate objective)."""
    P_, A, V = L.shape
    Lf = L.permute(1, 0, 2).reshape(A, P_ * V)
    if starts is None:
        starts = reference_starts(single_atom_scores(L, hit, excl, k), top_n)
    best = None
    for name, w0 in starts.items():
        param = torch.nn.Parameter((w0.float() / w0.float().norm()).clone())
        opt = torch.optim.Adam([param], lr=lr)
        for step in range(steps + 1):
            with torch.no_grad():
                th = param.detach()
                nrm = th.norm()
                u = th / nrm
                z = (u @ Lf).view(P_, V)
                spread = z.std(dim=1)
                top = z.masked_fill(excl[None, :], float("-inf")).topk(k, dim=1)
                tk = (top.values[:, -1], top.indices[:, -1])
                s_hit, e_hit, g = _surr_terms(z, tk, hit, spread, L)
                obj_s, obj_e = s_hit, e_hit
                e_false = 0.0
                if false is not None and gamma > 0:
                    s_f, e_false, g_f = _surr_terms(z, tk, false, spread, L)
                    obj_s, obj_e, g = obj_s - gamma * s_f, obj_e - gamma * e_false, g - gamma * g_f
            klv = 0.0
            if kl is not None and beta > 0:
                klv, g_kl = kl.value_grad(u)
                obj_s, obj_e, g = obj_s - beta * klv, obj_e - beta * klv, g - beta * g_kl
            if step % exact_every == 0 or step == steps:
                cand = (obj_e, obj_s)
                if best is None or cand > best[0]:
                    best = (cand, {"w": u.clone(), "train_hit": e_hit, "train_false": e_false, "kl_search": klv,
                                   "init": name, "step": step})
            if step == steps:
                break
            gth = (g - (g @ u) * u) / nrm
            opt.zero_grad()
            param.grad = -gth.clone()
            opt.step()
    out = best[1]
    out["kl"] = kl.exact(out["w"]) if kl is not None else None
    return out


def fit_kl_only(kl: KLTerm, counts: torch.Tensor, steps=200, lr=0.05, every=5):
    E = kl.L.shape[1]
    single = []
    for e in range(E):
        w = torch.zeros(E)
        w[e] = 1.0
        single.append(kl.value(w))
    starts = {"uniform": torch.ones(E), "best_single": torch.eye(E)[int(np.argmin(single))], "rlens": counts.float()}
    best = None
    for name, w0 in starts.items():
        param = torch.nn.Parameter((w0 / w0.norm()).clone())
        opt = torch.optim.Adam([param], lr=lr)
        for step in range(steps + 1):
            th = param.detach()
            nrm = th.norm()
            u = th / nrm
            v, g = kl.value_grad(u)
            if step % every == 0 or step == steps:
                if best is None or v < best[0]:
                    best = (v, {"w": u.clone(), "kl_search": v, "init": name, "step": step})
            if step == steps:
                break
            gth = (g - (g @ u) * u) / nrm          # gradient of KL; we minimise it
            opt.zero_grad()
            param.grad = gth.clone()
            opt.step()
    out = best[1]
    out["kl"] = kl.exact(out["w"])
    return out


# ---------------------------------------------------------------------- scoring helpers
@torch.no_grad()
def task_scores(ctx: Ctx, maps: dict, items) -> dict:
    """Recall@10 per stratum and best-of-7 on a set of items (macro over the macro evals present)."""
    prep = ctx.prep
    items = [i for i in items if prep.items[i].eval in MACRO]
    pos = {i: k for k, i in enumerate(items)}
    rows = [(i, w) for i, w in ctx.rows if i in pos]
    ranks = {}
    for l in LAYERS:
        Z = ctx.masked(ctx.unembed(prep.resid[items, l].float() @ maps[l].float().T))
        ranks[l] = np.array([int((Z[pos[i]] > Z[pos[i], prep.cands[i][w]].max()).sum()) + 1 for i, w in rows])

    def rec(rk):
        by = {}
        for (i, _), r in zip(rows, rk):
            by.setdefault(i, []).append(r)
        per = {}
        for i, rs in by.items():
            per.setdefault(prep.items[i].eval, []).append(float(np.mean([r <= 10 for r in rs])))
        return float(np.mean([np.mean(v) for v in per.values()]))
    out = {s: float(np.mean([rec(ranks[l]) for l in ls])) for s, ls in STRATA.items()}
    out["best7"] = rec(np.min(np.stack([ranks[l] for l in LAYERS]), axis=0))
    return out


def load_json(name):
    p = os.path.join(RESULTS, name)
    return json.load(open(p)) if os.path.exists(p) else None


def fit_inputs(ctx: Ctx, fit_idx, l: int, seed: int):
    its, hit = fit_rows(ctx, fit_idx)
    L = torch.einsum("edk,nk->ned", ctx.experts[l].float(), ctx.prep.resid[its, l].float()) @ ctx.A.T
    return its, hit, control_pairs(ctx, its, seed), L


def loto_folds():
    out = []
    for k in range(len(MACRO)):
        for rep in range(LOTO_REPS):
            out.append((k, rep))
    return out


def loto_fit_items(ctx: Ctx, k: int, rep: int):
    prep = ctx.prep
    ev = MACRO[k]
    scored = [i for i in range(len(prep.items)) if ctx.keep[i]]
    others = [i for i in scored if prep.items[i].eval != ev]
    by_ev = {}
    for i in others:
        by_ev.setdefault(prep.items[i].eval, []).append(i)
    rng = np.random.default_rng(300 + 10 * k + rep)
    fit_idx = []
    for e in sorted(by_ev):
        n_e = max(1, round(47 * len(by_ev[e]) / len(others)))
        fit_idx += [by_ev[e][j] for j in rng.permutation(len(by_ev[e]))[:n_e]]
    return fit_idx, [i for i in scored if prep.items[i].eval == ev]


# ====================================================================== stages
def stage_tune(ctx: Ctx):
    res = load_json("exp2_tune.json") or {"betas": BETAS, "gamma": GAMMA, "weights": {}}
    for l in LAYERS:
        if str(l) in res["weights"]:
            continue
        H, Y = kl_positions(ctx, l)
        kl = KLTerm(ctx, l, H, Y)
        res["weights"][str(l)] = {}
        for seed in TUNE_SEEDS:
            fit_idx, _ = ctx.split(seed)
            its, hit, false, L = fit_inputs(ctx, fit_idx, l, seed)
            ent = {}
            for beta in BETAS:
                r = fit_multi(L, hit, ctx.excl, false=false, gamma=GAMMA, kl=kl, beta=beta)
                ent[str(beta)] = {"w": r["w"].tolist(), "train_hit": r["train_hit"], "kl": r["kl"]}
            res["weights"][str(l)][str(seed)] = ent
            del L
            log(f"tune layer {l} seed {seed}: " + " ".join(f"b{b}: hit {ent[str(b)]['train_hit']:.3f} kl {ent[str(b)]['kl']:.2f}"
                                                           for b in BETAS))
        del kl
        save_json(res, "exp2_tune.json")
    scores = {}
    for seed in TUNE_SEEDS:
        _, held = ctx.split(seed)
        for beta in BETAS:
            maps = {l: combine(ctx.experts[l], torch.tensor(res["weights"][str(l)][str(seed)][str(beta)]["w"]))
                    for l in LAYERS}
            scores.setdefault(str(beta), []).append(task_scores(ctx, maps, held)["best7"])
    means = {b: float(np.mean(v)) for b, v in scores.items()}
    best = max(means.values())
    beta_star = max(float(b) for b in means if means[b] >= best - 0.002)
    res["selection"] = {"mean_held_best7_B2": means, "beta_star": beta_star}
    save_json(res, "exp2_tune.json")
    log(f"beta* = {beta_star} ({means})")
    runlog_append("E2-tune", "select beta for B2 (Surr - FalseSurr - beta*KL) on tuning splits 0-2, all layers",
                  "n/a", "B2 mean held-out best-of-7 recall@10 by beta: " + ", ".join(f"{k}: {v:.4f}" for k, v in means.items()),
                  "rule: best mean, largest beta within 0.2 pt", f"beta* = {beta_star}; tuning splits only")
    return beta_star


def stage_confirm(ctx: Ctx, beta: float):
    """Layer-major fitting of B1 and B2 on seeds 100-109 (B0 = Exp 1's A0), then evaluation per seed."""
    res = load_json("exp2_confirm.json") or {"beta_star": beta, "gamma": GAMMA, "weights": {}, "per_seed": {}}
    e1 = load_json("exp1_confirm.json")
    for l in LAYERS:
        if str(l) in res["weights"]:
            continue
        H, Y = kl_positions(ctx, l)
        kl = KLTerm(ctx, l, H, Y)
        b1 = fit_kl_only(kl, ctx.counts[l])
        ent = {"B1": {"w": b1["w"].tolist(), "kl": b1["kl"]}, "seeds": {}}
        for seed in CONF_SEEDS:
            fit_idx, _ = ctx.split(seed)
            its, hit, false, L = fit_inputs(ctx, fit_idx, l, seed)
            w0 = torch.tensor(e1["per_seed"][str(seed)]["weights"]["A0"][str(l)]["w"])
            r2 = fit_multi(L, hit, ctx.excl, false=false, gamma=GAMMA, kl=kl, beta=beta)
            # B3/B4 (secondary attribution arms) dropped by Amendment 2 to save compute
            ent["seeds"][str(seed)] = {"B0": {"w": w0.tolist(), "kl": kl.exact(w0 / w0.norm())},
                                       "B2": {k: (v.tolist() if torch.is_tensor(v) else v) for k, v in r2.items()}}
            del L
            log(f"layer {l} seed {seed}: train hit B2 {r2['train_hit']:.3f} false {r2['train_false']:.3f}; "
                f"KL B0 {ent['seeds'][str(seed)]['B0']['kl']:.2f} B1 {b1['kl']:.2f} B2 {r2['kl']:.2f}")
        res["weights"][str(l)] = ent
        del kl
        save_json(res, "exp2_confirm.json")
    for seed in CONF_SEEDS:
        if str(seed) in res["per_seed"]:
            continue
        fit_idx, held_idx = ctx.split(seed)
        summ, fitg = {}, {}
        for arm in ARMS:
            maps = {l: combine(ctx.experts[l], torch.tensor(res["weights"][str(l)]["B1"]["w"] if arm == "B1"
                                                            else res["weights"][str(l)]["seeds"][str(seed)][arm]["w"]))
                    for l in LAYERS}
            summ[arm], _ = evaluate(ctx, LensSpec(arm, maps=maps), held_idx, seed, wiki=True, transport=True)
            f = evaluate(ctx, LensSpec(arm, maps=maps), fit_idx, seed, wiki=False, transport=False)[0]
            fitg[arm] = {s: f[s]["recall10"] for s in STRATA} | {"best7": f["best7_recall10"]}
            if seed == CONF_SEEDS[0] and arm in ("B0", "B2"):
                torch.save({arm: {l: m.half() for l, m in maps.items()}},
                           os.path.join(RESULTS, f"maps_exp2_seed{seed}_{arm}.pt"))
        res["per_seed"][str(seed)] = {"results": summ, "fit_split": fitg}
        log(f"seed {seed}: best7 " + " ".join(f"{a} {summ[a]['best7_recall10']:.3f}" for a in summ))
        save_json(res, "exp2_confirm.json")
    return res


def stage_loto(ctx: Ctx, beta: float):
    res = load_json("exp2_loto.json") or {"beta_star": beta, "weights": {}, "scores": {}}
    conf = load_json("exp2_confirm.json")
    n_reps = len(loto_folds())
    for l in LAYERS:
        if len(res["weights"].get(str(l), {})) == n_reps:
            continue
        H, Y = kl_positions(ctx, l)
        kl = KLTerm(ctx, l, H, Y)
        res["weights"].setdefault(str(l), {})
        for k, rep in loto_folds():
            key = f"{k}_{rep}"
            if key in res["weights"][str(l)]:
                continue                  # checkpointed per replicate (container restarts)
            fit_idx, _ = loto_fit_items(ctx, k, rep)
            its, hit, false, L = fit_inputs(ctx, fit_idx, l, 900 + 10 * k + rep)
            r0 = fit_weights(L, hit, ctx.excl)
            r2 = fit_multi(L, hit, ctx.excl, false=false, gamma=GAMMA, kl=kl, beta=beta)
            res["weights"][str(l)][key] = {"B0": r0.w.tolist(), "B2": r2["w"].tolist()}
            del L
            save_json(res, "exp2_loto.json")
        del kl
        log(f"LOTO layer {l} fitted")
    for k, rep in loto_folds():
        key = f"{k}_{rep}"
        if key in res["scores"]:
            continue
        _, test = loto_fit_items(ctx, k, rep)
        sc = {}
        for arm in ("B0", "B2", "B1"):
            maps = {l: combine(ctx.experts[l], torch.tensor(conf["weights"][str(l)]["B1"]["w"] if arm == "B1"
                                                            else res["weights"][str(l)][key][arm])) for l in LAYERS}
            sc[arm] = task_scores(ctx, maps, test)
        res["scores"][key] = {"eval": MACRO[k], "rep": rep, **sc}
        log(f"LOTO {MACRO[k]} rep {rep}: best7 B0 {sc['B0']['best7']:.3f} B2 {sc['B2']['best7']:.3f} B1 {sc['B1']['best7']:.3f}")
        save_json(res, "exp2_loto.json")
    return res


def stage_verdict():
    conf, loto = load_json("exp2_confirm.json"), load_json("exp2_loto.json")
    seeds = [s for s in CONF_SEEDS if str(s) in conf["per_seed"]]
    R = {s: conf["per_seed"][str(s)] for s in seeds}

    def stat(x):
        x = np.array(x)
        return {"mean": float(x.mean()), "min": float(x.min()), "max": float(x.max()), "wins": int((x > 0).sum()),
                "ties": int((x == 0).sum()), "losses": int((x < 0).sum()), "n": len(x)}
    keys = sorted(loto["scores"])
    lo = {arm: stat([loto["scores"][k][arm]["best7"] - loto["scores"][k]["B0"]["best7"] for k in keys]) for arm in ("B2", "B1")}
    lo_strata = {s: stat([loto["scores"][k]["B2"][s] - loto["scores"][k]["B0"][s] for k in keys]) for s in STRATA}

    def gap(arm, s):
        return float(np.mean([R[sd]["fit_split"][arm][s] - R[sd]["results"][arm][s]["recall10"] for sd in seeds]))
    gaps = {arm: float(np.mean([gap(arm, s) for s in STRATA])) for arm in ARMS}
    best7 = {arm: stat([R[s]["results"][arm]["best7_recall10"] - R[s]["results"]["B0"]["best7_recall10"] for s in seeds])
             for arm in ARMS if arm != "B0"}
    fpm = stat([R[s]["results"]["B2"]["early"]["fp10_matched"] - R[s]["results"]["B0"]["early"]["fp10_matched"] for s in seeds])
    c1 = lo["B2"]["mean"] >= 0.02 and lo["B2"]["wins"] >= 14
    c2 = gaps["B2"] <= 0.5 * gaps["B0"]
    c3 = best7["B2"]["mean"] >= -0.005
    c4 = fpm["mean"] <= 0.005
    refute = lo["B2"]["mean"] < 0.005 or lo["B2"]["wins"] <= 10 or best7["B2"]["mean"] < -0.02
    verdict = "CONFIRMED" if (c1 and c2 and c3 and c4) else "REFUTED" if refute else "INCONCLUSIVE"
    if verdict != "CONFIRMED" and c2 and not c1:
        verdict += " (reduces overfitting without generalisation gain)"
    means = {}
    for arm in ARMS:
        means[arm] = {"best7": float(np.mean([R[s]["results"][arm]["best7_recall10"] for s in seeds])),
                      **{f"{st}_recall10": float(np.mean([R[s]["results"][arm][st]["recall10"] for s in seeds])) for st in STRATA},
                      "early_fp10_matched": float(np.mean([R[s]["results"][arm]["early"]["fp10_matched"] for s in seeds])),
                      "early_fp10_wiki": float(np.mean([R[s]["results"][arm]["early"]["fp10_wiki"] for s in seeds])),
                      "early_tr_kl": float(np.mean([R[s]["results"][arm]["early"]["tr_kl"] for s in seeds])),
                      "early_tr_cos": float(np.mean([R[s]["results"][arm]["early"]["tr_cos"] for s in seeds])),
                      "gap": gaps[arm]}
    out = {"beta_star": conf["beta_star"], "gamma": GAMMA, "n_splits": len(seeds), "n_loto": len(keys),
           "loto_best7_vs_B0": lo, "loto_strata_B2_vs_B0": lo_strata, "standard_best7_vs_B0": best7,
           "early_fp10_matched_B2_minus_B0": fpm, "overfit_gap": gaps, "means": means,
           "criteria": {"C1_loto": c1, "C2_gap_halved": c2, "C3_no_loss": c3, "C4_fp": c4, "refute": refute},
           "verdict": verdict}
    save_json(out, "exp2_verdict.json")
    print(json.dumps(out, indent=1))
    runlog_append("E2-confirm", "H2: Surr - FalseSurr - beta*KL generalises to a held-out task and removes overfitting",
                  f"J++ (B0): LOTO best-of-7 {np.mean([loto['scores'][k]['B0']['best7'] for k in keys]):.4f}; gap {gaps['B0']:.4f}",
                  f"B2: LOTO d {lo['B2']['mean']:+.4f} (W {lo['B2']['wins']}/{lo['B2']['n']}); gap {gaps['B2']:.4f}; "
                  f"standard best-of-7 d {best7['B2']['mean']:+.4f}; dFP@10 matched {fpm['mean']:+.4f}",
                  f"B1 KL-only LOTO d {lo['B1']['mean']:+.4f}, standard d {best7['B1']['mean']:+.4f} (B3/B4 dropped, Amendment 2)",
                  f"{verdict} (beta*={conf['beta_star']}, gamma={GAMMA})")
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
        beta = load_json("exp2_tune.json")["selection"]["beta_star"]
        stage_confirm(ctx, beta) if a.stage == "confirm" else stage_loto(ctx, beta)
    log("done")
