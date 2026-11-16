"""J++'s expert-weight search (jpp_lens ``fit_expert_weights``), re-implemented for speed with the
same algorithm, plus two extensions used by the sprint (a ridge penalty toward a reference
weighting and caller-supplied starting points).

Reference: safety-research/jpp_lens @ d7efc0d, ``workspace_lens/fitting/condense_experts``.
What is identical to the reference: the objective (per-layer pair-level macro pass@k with
Readout Filtering), the surrogate (sigmoid of (best candidate logit - k-th largest masked
logit) / (temperature * unbiased std of the *unmasked* logits), std detached), unit-L2 scale-free
weights, Adam (lr 0.05) for 200 steps from each start, the exact macro evaluated every 5 steps
and at the end, and the returned iterate = the best (exact macro, surrogate) over all starts and
evaluated steps. Starts default to the reference's three (uniform, best single atom, top-3).

What differs (speed only): the combined logits are one GEMV instead of a loop over experts, the
top-k is taken once per item rather than once per pair, and the gradient is computed
analytically. The surrogate depends on the weights only through the best-candidate and k-th
largest entries of each row (the spread is detached), so its gradient is
    sum_q c_q (L[item_q, :, best_q] - L[item_q, :, kth_q]),
exactly what autograd returns. Float summation order differs from the reference, so near-ties can
resolve differently; ``validate()`` checks agreement on real fixtures.

Extensions: ``penalty`` = lambda * ||u - u_ref||^2 on the unit weights u, subtracted from both the
surrogate (gradient) and the exact score used to choose the returned iterate. lambda = 0 recovers
the reference exactly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass
class Pairs:
    item: torch.Tensor        # [Q] long, index into the P items
    cands: torch.Tensor       # [Q, C] long, padded by repeating the first id
    ev: torch.Tensor          # [Q] long, eval group index
    n_groups: int

    @classmethod
    def build(cls, row_item: Sequence[int], row_cands: Sequence[Sequence[int]], row_eval: Sequence[str]):
        names = sorted(set(row_eval))
        maxc = max(len(c) for c in row_cands)
        return cls(torch.tensor(list(row_item), dtype=torch.long),
                   torch.tensor([list(c) + [c[0]] * (maxc - len(c)) for c in row_cands], dtype=torch.long),
                   torch.tensor([names.index(e) for e in row_eval], dtype=torch.long), len(names))

    def group_weights(self) -> torch.Tensor:
        """Per-pair weight of the macro mean: 1 / (n_groups * size of the pair's group)."""
        cnt = torch.bincount(self.ev, minlength=self.n_groups).float()
        return 1.0 / (self.n_groups * cnt[self.ev])


def macro(values_q: torch.Tensor, pairs: Pairs) -> float:
    return float((values_q * pairs.group_weights()).sum())


def single_atom_scores(L: torch.Tensor, pairs: Pairs, excl: torch.Tensor, k: int = 10) -> torch.Tensor:
    """[A] stand-alone exact macro pass@k of each atom (the reference's expert_pair_pass_scores)."""
    out = []
    for a in range(L.shape[1]):
        z = L[:, a].masked_fill(excl[None], float("-inf"))
        zq = z[pairs.item]
        best = zq.gather(1, pairs.cands).amax(1)
        kth = z.topk(k, dim=1).values[:, -1][pairs.item]
        out.append(macro((best - kth >= 0).float(), pairs))
    return torch.tensor(out)


def reference_starts(scores: torch.Tensor, top_n: int = 3) -> dict[str, torch.Tensor]:
    A = scores.numel()
    order = sorted(range(A), key=lambda a: (-float(scores[a]), a))
    sb = torch.zeros(A)
    sb[order[0]] = 1.0
    mt = torch.zeros(A)
    mt[order[:top_n]] = 1.0
    return {"uniform": torch.ones(A), "select_best": sb, "mean_top": mt}


@dataclass
class Result:
    w: torch.Tensor           # unit L2
    train_exact: float        # exact macro pass@k on the fitting pairs (without penalty)
    objective: float          # exact - penalty, the selection score
    init: str
    step: int


def fit_weights(L: torch.Tensor, pairs: Pairs, excl: torch.Tensor, *, starts: dict[str, torch.Tensor] | None = None,
                k: int = 10, steps: int = 200, lr: float = 0.05, temperature: float = 0.1, exact_every: int = 5,
                top_n: int = 3, lam: float = 0.0, w_ref: torch.Tensor | None = None, loop_sum: bool = False) -> Result:
    """L: [P, A, V] fp32 rank-equivalent atom logits (A = experts, or expert x band atoms).
    excl: [V] bool, the Readout Filtering mask with every candidate id spared."""
    P, A, V = L.shape
    Lf = None if loop_sum else L.permute(1, 0, 2).reshape(A, P * V)   # [A, P*V] for one GEMV per step
    gw = pairs.group_weights()
    excl_row = excl[None, :]
    if starts is None:
        starts = reference_starts(single_atom_scores(L, pairs, excl, k), top_n)
    if lam > 0:
        assert w_ref is not None
        u_ref = w_ref.float() / w_ref.float().norm()
    best: tuple[tuple[float, float], Result] | None = None
    for name, w0 in starts.items():
        theta = (w0.float() / w0.float().norm()).clone()
        opt_param = torch.nn.Parameter(theta)
        opt = torch.optim.Adam([opt_param], lr=lr)
        for step in range(steps + 1):
            with torch.no_grad():
                th = opt_param.detach()
                nrm = th.norm()
                u = th / nrm
                if loop_sum:   # the reference's summation order (bitwise-identical logits)
                    z = torch.zeros(P, V)
                    for a in range(A):
                        z = z + u[a] * L[:, a]
                else:
                    z = (u @ Lf).view(P, V)
                spread = z.std(dim=1)                          # unbiased, unmasked, detached
                zm = z.masked_fill(excl_row, float("-inf"))
                top = zm.topk(k, dim=1)
                kth_val, kth_id = top.values[:, -1], top.indices[:, -1]
                cz = z[pairs.item].gather(1, pairs.cands)          # candidates are never excluded
                bpos = cz.argmax(1)
                best_val = cz.gather(1, bpos[:, None])[:, 0]
                best_id = pairs.cands.gather(1, bpos[:, None])[:, 0]
                margin = best_val - kth_val[pairs.item]
                sq = spread[pairs.item]
                s = torch.sigmoid(margin / (temperature * sq))
                surr = float((s * gw).sum())
                pen = float(lam * ((u - u_ref) ** 2).sum()) if lam > 0 else 0.0
                if step % exact_every == 0 or step == steps:
                    ex = macro((margin >= 0).float(), pairs)
                    cand = (ex - pen, surr - pen)
                    if best is None or cand > best[0]:
                        best = (cand, Result(u.clone(), ex, ex - pen, name, step))
                if step == steps:
                    break
                # analytic gradient of (surrogate - penalty) w.r.t. u, then through u = th/|th|
                c = gw * s * (1 - s) / (temperature * sq)                         # [Q]
                D = L[pairs.item, :, best_id] - L[pairs.item, :, kth_id[pairs.item]]   # [Q, A]
                gu = (c[:, None] * D).sum(0)                                        # [A]
                if lam > 0:
                    gu = gu - 2 * lam * (u - u_ref)
                gth = (gu - (gu @ u) * u) / nrm
            opt.zero_grad()
            opt_param.grad = -gth.clone()
            opt.step()
    assert best is not None
    return best[1]


def validate(fixture_dir: str, layers=(4, 21)) -> dict:
    """Compare with the reference implementation on saved fixtures (sprint/fidelity)."""
    import os
    import sys
    sys.path.insert(0, "/home/user/safety-research/jpp_lens/src")
    from workspace_lens.fitting.condense_experts.fitter import (DEFAULT_CONDENSE_CONFIG,
                                                                build_search_starting_points, fit_expert_weights)
    from workspace_lens.fitting.condense_experts.rank_objective import PairTable, expert_pair_pass_scores
    out = {}
    for l in layers:
        d = torch.load(os.path.join(fixture_dir, f"inputs_{l}.pt"), weights_only=False)
        Z = d["Z"].float()
        pairs = Pairs.build(d["row_item"], d["row_cands"], d["row_eval"])
        excl = d["exclude"].clone()
        for c in d["row_cands"]:
            excl[list(c)] = False
        import time
        t0 = time.time()
        mine = fit_weights(Z, pairs, excl)
        t_mine = time.time() - t0
        ev = tuple(sorted(set(d["row_eval"])))
        pt = PairTable(pairs.item, pairs.cands, pairs.ev, ev)
        t0 = time.time()
        sc = expert_pair_pass_scores(Z, pt, excluded_mask_Bool_V=excl, k=10)
        ref = fit_expert_weights(Z, pt, excluded_mask_Bool_V=excl,
                                 initial_weightings=build_search_starting_points(sc, top_n=3),
                                 condense_config=DEFAULT_CONDENSE_CONFIG)
        t_ref = time.time() - t0
        mine_sc = single_atom_scores(Z, pairs, excl)
        out[str(l)] = {"train_mine": mine.train_exact, "train_ref": ref.train_macro_pass,
                       "init_mine": mine.init, "step_mine": mine.step, "init_ref": ref.init_name,
                       "step_ref": ref.step, "cos_w": float(mine.w @ ref.weights_E),
                       "single_atom_scores_equal": bool(torch.allclose(mine_sc, sc)),
                       "sec_mine": round(t_mine, 1), "sec_ref": round(t_ref, 1)}
    return out
