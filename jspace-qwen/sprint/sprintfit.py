"""Fitting helpers shared by the sprint's experiments: J++'s fitting inputs (rank-equivalent atom
logits L[n, a, :] = A (M_a h_n), as jpp_lens caches them) and the fitted per-layer maps."""
from __future__ import annotations

import torch

from harness import LAYERS, Ctx
from jppsearch import Pairs, Result, fit_weights


def fit_rows(ctx: Ctx, item_ids) -> tuple[list[int], Pairs]:
    """Fitting pairs over every scoreable eval (J++ fits on all six evals' items)."""
    s = set(item_ids)
    its = sorted({i for i, _ in ctx.rows if i in s})
    pos = {i: j for j, i in enumerate(its)}
    fr = [(i, w) for i, w in ctx.rows if i in s]
    pairs = Pairs.build([pos[i] for i, _ in fr], [ctx.prep.cands[i][w] for i, w in fr],
                        [ctx.prep.items[i].eval for i, _ in fr])
    return its, pairs


@torch.no_grad()
def atom_logits(ctx: Ctx, atoms: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """[n, A, V]: A (M_a h_n) for atoms M_a [A, d, d] (pre-norm; ranks ignore the positive rms factor)."""
    X = torch.einsum("adk,nk->nad", atoms.float(), H.float())
    return X @ ctx.A.T


def fit_layer(ctx: Ctx, atoms: torch.Tensor, fit_idx, l: int, **kw) -> Result:
    its, pairs = fit_rows(ctx, fit_idx)
    L = atom_logits(ctx, atoms, ctx.prep.resid[its, l])
    res = fit_weights(L, pairs, ctx.excl, **kw)
    del L
    return res


def combine(atoms: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.einsum("a,adk->dk", w.float(), atoms.float())


def fit_jpp(ctx: Ctx, fit_idx, layers=LAYERS) -> tuple[dict[int, torch.Tensor], dict]:
    maps, log = {}, {}
    for l in layers:
        r = fit_layer(ctx, ctx.experts[l], fit_idx, l)
        maps[l] = combine(ctx.experts[l], r.w)
        log[l] = {"w": [round(x, 5) for x in r.w.tolist()], "train_exact": r.train_exact, "init": r.init,
                  "step": r.step}
    return maps, log
