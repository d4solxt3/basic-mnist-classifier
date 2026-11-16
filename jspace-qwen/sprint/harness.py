"""Evaluation harness for the J++ improvement sprint (Qwen3-0.6B, cached shards).

Every experiment reports four metric families, always out of sample:

M1  Readout. Held-out recall@10 and MRR of the latent intermediate, per layer, aggregated by
    stratum (early = layers 4, 7; mid = 11, 14; late = 18, 21, 24) and best-of-7 (J++'s
    protocol). Item-weighted, macro over the five macro evals (multihop, multilingual, typo,
    association, poetry). Readout Filtering on (non-semantic ids excluded, candidates spared).
M2  False positives. (a) matched controls: the held-out item's intermediate ranked under up to 5
    other held-out items of the same eval (no shared candidates, intermediate not in the control
    prompt): FP@10 and softmax mass on the intermediate; plus the specificity AUC
    P(rank under own prompt < rank under control). (b) generic controls: every held-out
    intermediate ranked at 256 held-out WikiText positions: FP@10 and mass.
M3  Transport alignment on 4,096 held-out WikiText positions (records 7000-7127): cosine between
    the transported residual M h_l and the final residual h_27, NMSE after one least-squares scale
    fitted on 2,048 calibration positions (records 5000-5255), and KL(model next-token distribution
    || lens distribution) on 512 positions. NB: J-lens-family maps average influence on all later
    positions, so they are not designed to predict h_27 at the same position.
M4  Causal swap (top candidates only): sprint/swap_check.py.

All lenses are maps x -> M_l x (or callables for input-conditional lenses); readout logits are
unembed(M_l h) = A (M_l h) / rms(M_l h), A = W_U diag(g).
"""

from __future__ import annotations

import datetime
import json
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "sprint"))
from jspace import evals as E  # noqa: E402
from jspace import paths as P  # noqa: E402
from jspace.jpp import finalize_experts, split_items  # noqa: E402
from jspace.lens import Lens  # noqa: E402
from jspace.model import load_model  # noqa: E402

LAYERS = [4, 7, 11, 14, 18, 21, 24]
STRATA = {"early": [4, 7], "mid": [11, 14], "late": [18, 21, 24]}
MACRO = E.MACRO_EVALS
RESULTS = os.path.join(ROOT, "sprint", "results")
RUNLOG = os.path.join(ROOT, "sprint", "RUNLOG.md")
N_CONTROLS = 5
N_WIKI_FP = 256
N_KL = 512
N_CALIB = 2048


@dataclass
class LensSpec:
    """A lens = per-layer maps (d x d, x -> M x) or a per-layer callable H[n,d] -> X[n,d]."""
    name: str
    maps: dict[int, torch.Tensor] | None = None
    fn: Callable[[torch.Tensor, int], torch.Tensor] | None = None
    bias: dict[int, torch.Tensor] = field(default_factory=dict)

    def transport(self, H: torch.Tensor, l: int) -> torch.Tensor:
        H = H.float()
        if self.fn is not None:
            return self.fn(H, l)
        X = H @ self.maps[l].float().T
        if l in self.bias:
            X = X + self.bias[l].float()
        return X


class Ctx:
    """Shared, split-independent state."""

    def __init__(self, threads: int = 4):
        torch.set_num_threads(threads)
        self.model = load_model("Qwen/Qwen3-0.6B")
        m = self.model
        self.A = (m.lm_head.weight.float() * m.final_norm.weight.float()[None, :]).contiguous()   # [V, d]
        self.eps = float(getattr(m.final_norm, "variance_epsilon", 1e-6))
        self.V = self.A.shape[0]
        self.prep = torch.load(os.path.join(P.LENSES, "readout_prep.pt"), weights_only=False)
        self.keep = E.scored_item_mask(self.prep)
        self.rows = E.row_index(self.prep)
        excl = E.non_semantic_ids(m.tokenizer, m.vocab_size).clone()
        for c in self.prep.cands:                      # spare every candidate of every item (J++'s rule)
            for ids in c.values():
                excl[ids] = False
        self.excl = excl
        ex = torch.load(os.path.join(P.LENSES, "lrp_experts_final.pt"), weights_only=True)
        self.experts, self.pooled, self.counts = {}, {}, {}
        for l in LAYERS:
            self.counts[l] = ex["counts"][l].float()
            self.experts[l], self.pooled[l], _ = finalize_experts(ex["experts"][l], ex["counts"][l])
        del ex
        self.jlens = Lens.load(os.path.join(P.LENSES, "exact_final.pt"))
        self.rlens = Lens.load(os.path.join(P.LENSES, "lrp_final.pt"))
        self.regress = Lens.load(os.path.join(P.LENSES, "regress.pt"))
        wiki = torch.load(os.path.join(P.LENSES, "heldout_wiki_acts.pt"), weights_only=False)
        self.wiki = {l: wiki["acts"][l].float() for l in LAYERS}
        self.wiki_y = wiki["target"].float()
        g = torch.Generator().manual_seed(7)
        nw = self.wiki_y.shape[0]
        self.wiki_fp_idx = torch.randperm(nw, generator=g)[:N_WIKI_FP]
        self.wiki_kl_idx = torch.randperm(nw, generator=g)[:N_KL]
        cal = torch.load(os.path.join(P.LENSES, "corpus_acts.pt"), weights_only=False, mmap=True)
        ci = torch.randperm(cal["target"].shape[0], generator=g)[:N_CALIB]
        self.cal = {l: cal["acts"][l][ci].float() for l in LAYERS}
        self.cal_y = cal["target"][ci].float()
        del cal
        with torch.no_grad():
            self.wiki_kl_logp = torch.log_softmax(self.unembed(self.wiki_y[self.wiki_kl_idx]), dim=-1)

    # ------------------------------------------------------------------ basics
    def unembed(self, X: torch.Tensor) -> torch.Tensor:
        """True logits [n, V] of transported residuals X [n, d] (RMSNorm then head)."""
        X = X.float()
        rms = (X.pow(2).mean(-1, keepdim=True) + self.eps).sqrt()
        return (X / rms) @ self.A.T

    def masked(self, Z: torch.Tensor) -> torch.Tensor:
        return Z.masked_fill(self.excl[None, :], float("-inf"))

    def split(self, seed: int) -> tuple[list[int], list[int]]:
        return split_items(self.prep.items, self.keep, seed=seed)

    def standard_lenses(self) -> dict[str, LensSpec]:
        d = self.A.shape[1]
        return {"logit": LensSpec("logit", maps={l: torch.eye(d) for l in LAYERS}),
                "J-lens": LensSpec("J-lens", maps={l: self.jlens.maps[l].float() for l in LAYERS}),
                "R-lens": LensSpec("R-lens", maps={l: self.rlens.maps[l].float() for l in LAYERS})}

    # ------------------------------------------------------------------ M1 + M2a
    @torch.no_grad()
    def readout(self, lens: LensSpec, held: Sequence[int], seed: int) -> dict:
        """Per layer: ranks of held-out (item, intermediate) rows, recall@10/MRR, matched-control FPs."""
        prep = self.prep
        held = [i for i in held if prep.items[i].eval in MACRO + ("order-ops",)]
        pos = {i: j for j, i in enumerate(held)}
        rows = [(i, w) for i, w in self.rows if i in pos]
        controls = self.matched_controls(held, seed)
        out = {"per_layer": {}, "ranks": {}}
        for l in LAYERS:
            Z = self.masked(self.unembed(lens.transport(prep.resid[held, l], l)))       # [n, V]
            lse = torch.logsumexp(Z, dim=1)
            S = Z.sort(dim=1).values                                                     # ascending
            ranks, rr = [], []
            for i, w in rows:
                c = prep.cands[i][w]
                best = Z[pos[i], c].max()
                ranks.append(int((Z[pos[i]] > best).sum()) + 1)
            out["ranks"][l] = ranks
            # matched controls: rank/mass of item i's intermediate under control j's readout
            fp_rows = []
            for i, w in rows:
                c = prep.cands[i][w]
                for j in controls.get(i, []):
                    best = Z[pos[j], c].max()
                    rank = int(self.V - torch.searchsorted(S[pos[j]], best.reshape(1), right=True)) + 1
                    mass = float(torch.exp(Z[pos[j], c] - lse[pos[j]]).sum())
                    fp_rows.append((i, w, j, rank, mass))
            out["per_layer"][l] = {"fp_rows": fp_rows,
                                   "own_mass": [float(torch.exp(Z[pos[i], prep.cands[i][w]] - lse[pos[i]]).sum())
                                                for i, w in rows]}
            del Z, S
        out["rows"] = rows
        return out

    def matched_controls(self, held: Sequence[int], seed: int) -> dict[int, list[int]]:
        prep = self.prep
        rng = np.random.default_rng(10_000 + seed)
        by_ev: dict[str, list[int]] = {}
        for i in held:
            by_ev.setdefault(prep.items[i].eval, []).append(i)
        out = {}
        for ev, ids in by_ev.items():
            if ev not in MACRO:
                continue
            for i in ids:
                ci = {t for v in prep.cands[i].values() for t in v}
                words = [w.lower() for w in prep.cands[i]]
                ok = []
                for j in ids:
                    if j == i:
                        continue
                    cj = {t for v in prep.cands[j].values() for t in v}
                    if ci & cj:
                        continue
                    pj = prep.items[j].prompt.lower()
                    if any(w in pj for w in words):
                        continue
                    ok.append(j)
                rng.shuffle(ok)
                out[i] = ok[:N_CONTROLS]
        return out

    # ------------------------------------------------------------------ M2b
    @torch.no_grad()
    def wiki_fp(self, lens: LensSpec, held: Sequence[int]) -> dict:
        prep = self.prep
        rows = [(i, w) for i, w in self.rows if i in set(held) and prep.items[i].eval in MACRO]
        out = {}
        for l in LAYERS:
            Z = self.masked(self.unembed(lens.transport(self.wiki[l][self.wiki_fp_idx], l)))   # [256, V]
            lse = torch.logsumexp(Z, dim=1)
            S = Z.sort(dim=1).values
            hits, mass = {}, {}
            for i, w in rows:
                c = prep.cands[i][w]
                best = Z[:, c].max(1).values                                                 # [256]
                rank = self.V - torch.searchsorted(S, best[:, None], right=True)[:, 0] + 1
                ev = prep.items[i].eval
                hits.setdefault(ev, []).append(float((rank <= 10).float().mean()))
                mass.setdefault(ev, []).append(float(torch.exp(Z[:, c] - lse[:, None]).sum(1).mean()))
            out[l] = {"fp10": float(np.mean([np.mean(hits[e]) for e in MACRO if e in hits])),
                      "mass": float(np.mean([np.mean(mass[e]) for e in MACRO if e in mass]))}
            del Z, S
        return out

    # ------------------------------------------------------------------ M3
    @torch.no_grad()
    def transport_metrics(self, lens: LensSpec) -> dict:
        out = {}
        for l in LAYERS:
            Xc = lens.transport(self.cal[l], l)
            alpha = float((Xc * self.cal_y).sum() / (Xc * Xc).sum().clamp_min(1e-12))
            X = lens.transport(self.wiki[l], l)
            Y = self.wiki_y
            cos = torch.nn.functional.cosine_similarity(X, Y, dim=1)
            nmse = float(((alpha * X - Y) ** 2).sum() / (Y ** 2).sum())
            Xk = X[self.wiki_kl_idx]
            logq = torch.log_softmax(self.unembed(Xk), dim=-1)
            kl = float((self.wiki_kl_logp.exp() * (self.wiki_kl_logp - logq)).sum(-1).mean())
            out[l] = {"cos": float(cos.mean()), "nmse": nmse, "alpha": alpha, "kl": kl}
        return out


# ---------------------------------------------------------------------- summaries
def summarize(ctx: Ctx, ro: dict, wfp: dict | None, tr: dict | None) -> dict:
    """Per-layer and per-stratum M1/M2/M3 numbers from the raw outputs of one lens on one split."""
    prep = ctx.prep
    rows = ro["rows"]
    per_layer = {}
    for l in LAYERS:
        ranks = np.array(ro["ranks"][l])
        by_item: dict[int, list[int]] = {}
        for (i, _), r in zip(rows, ranks):
            by_item.setdefault(i, []).append(int(r))
        rec, mrr = {}, {}
        for i, rs in by_item.items():
            ev = prep.items[i].eval
            if ev not in MACRO:
                continue
            rec.setdefault(ev, []).append(float(np.mean([r <= 10 for r in rs])))
            mrr.setdefault(ev, []).append(float(np.mean([1.0 / r for r in rs])))
        fp_hits, fp_mass, auc = {}, {}, {}
        own_rank = {(i, w): r for (i, w), r in zip(rows, ranks)}
        for i, w, j, rank, mass in ro["per_layer"][l]["fp_rows"]:
            ev = prep.items[i].eval
            fp_hits.setdefault(ev, []).append(float(rank <= 10))
            fp_mass.setdefault(ev, []).append(mass)
            r_own = own_rank[(i, w)]
            auc.setdefault(ev, []).append(1.0 if r_own < rank else (0.5 if r_own == rank else 0.0))
        own_mass = {}
        for (i, w), m in zip(rows, ro["per_layer"][l]["own_mass"]):
            ev = prep.items[i].eval
            if ev in MACRO:
                own_mass.setdefault(ev, []).append(m)

        def mac(d):
            return float(np.mean([np.mean(d[e]) for e in MACRO if e in d]))
        per_layer[l] = {"recall10": mac(rec), "mrr": mac(mrr), "fp10_matched": mac(fp_hits),
                        "mass_matched": mac(fp_mass), "own_mass": mac(own_mass), "auc_matched": mac(auc),
                        "recall10_per_eval": {e: float(np.mean(rec[e])) for e in MACRO if e in rec}}
        if wfp is not None:
            per_layer[l].update({"fp10_wiki": wfp[l]["fp10"], "mass_wiki": wfp[l]["mass"]})
        if tr is not None:
            per_layer[l].update({f"tr_{k}": v for k, v in tr[l].items()})
    # best-of-7 recall (J++'s headline protocol)
    best_rank = np.min(np.stack([np.array(ro["ranks"][l]) for l in LAYERS]), axis=0)
    by_item = {}
    for (i, _), r in zip(rows, best_rank):
        by_item.setdefault(i, []).append(int(r))
    rec = {}
    for i, rs in by_item.items():
        ev = prep.items[i].eval
        if ev in MACRO:
            rec.setdefault(ev, []).append(float(np.mean([r <= 10 for r in rs])))
    out = {"per_layer": per_layer, "best7_recall10": float(np.mean([np.mean(rec[e]) for e in MACRO if e in rec]))}
    for s, ls in STRATA.items():
        out[s] = {k: float(np.mean([per_layer[l][k] for l in ls])) for k in per_layer[ls[0]]
                  if isinstance(per_layer[ls[0]][k], float)}
    return out


def evaluate(ctx: Ctx, lens: LensSpec, held: Sequence[int], seed: int, *, wiki: bool = True,
             transport: bool = True) -> tuple[dict, dict]:
    ro = ctx.readout(lens, held, seed)
    wfp = ctx.wiki_fp(lens, held) if wiki else None
    tr = ctx.transport_metrics(lens) if transport else None
    return summarize(ctx, ro, wfp, tr), ro


def runlog_append(experiment_id: str, hypothesis: str, metric_before: str, metric_after: str, control_check: str,
                  interpretation: str) -> None:
    """Append one row to sprint/RUNLOG.md (append-only; never rewritten)."""
    date = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M")
    cells = [date, experiment_id, hypothesis, metric_before, metric_after, control_check, interpretation]
    cells = [str(c).replace("|", "/").replace("\n", " ") for c in cells]
    with open(RUNLOG, "a") as f:
        f.write("| " + " | ".join(cells) + " |\n")


def save_json(obj, name: str) -> str:
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
    return path
