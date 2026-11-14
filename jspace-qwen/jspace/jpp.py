"""J++ Lens components (Ayonrinde & Lindsey, 2026) and our routed variant.

* ``fit_expert_weights``: J++'s per-layer signed expert weights, fitted on a
  labelled *fitting split* of the readout evals by Adam on a smooth surrogate
  of pair-level macro pass@k (sigmoid of the margin between the best candidate
  logit and the k-th largest logit, in units of the logit spread), from three
  starts (uniform, best single expert, top-3 indicator); the iterate with the
  best *exact* macro pass@k is kept. Weights are unit-L2 and may be negative.
* ``RoutedLens`` (ours): instead of collapsing experts into one map, read each
  residual out through the expert Jacobian of the cluster it falls in,
  ``unembed(J_{c(h)} h)``. Fully unsupervised; costs one nearest-centroid lookup
  in a 64-d space at inference.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

from jspace.lens import Lens
from jspace.model import LensModel


def split_items(items, keep_mask, *, seed: int = 0, fit_fraction: float = 0.2):
    """J++'s stratified split over the scored items: one rng over evals in
    sorted order; per eval, the first max(1, floor(f*n)) of a permutation fit."""
    by_eval: dict[str, list[int]] = {}
    for i, it in enumerate(items):
        if keep_mask[i]:
            by_eval.setdefault(it.eval, []).append(i)
    rng = np.random.default_rng(seed)
    fit = set()
    for ev in sorted(by_eval):
        idx = by_eval[ev]
        perm = rng.permutation(len(idx)).tolist()
        for j in perm[:max(1, int(fit_fraction * len(idx)))]:
            fit.add(idx[j])
    fit_idx = sorted(fit)
    held_idx = sorted(i for ids in by_eval.values() for i in ids if i not in fit)
    return fit_idx, held_idx


@torch.no_grad()
def expert_logits(model: LensModel, experts: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
    """Pre-norm expert logits  Z[n, e, :] = A (J_e h_n),  A = W_U diag(g). Ranks
    under any positive rescaling are unchanged, so the RMS factor is omitted."""
    A = model.lm_head.weight.float() * model.final_norm.weight.float()[None, :]
    X = torch.einsum("edk,nk->ned", experts.float(), H.float())          # [n, E, d]
    return X @ A.T                                                      # [n, E, V]


def _pair_tables(rows, evals_of_rows):
    ev_names = sorted(set(evals_of_rows))
    ev_idx = torch.tensor([ev_names.index(e) for e in evals_of_rows])
    return ev_names, ev_idx


def fit_expert_weights(Z: torch.Tensor, row_item: Sequence[int], row_cands: Sequence[Sequence[int]],
                       row_eval: Sequence[str], *, k: int = 10, steps: int = 200, lr: float = 0.05,
                       temperature: float = 0.1, exclude: torch.Tensor | None = None, top_n: int = 3,
                       exact_every: int = 5) -> tuple[torch.Tensor, float]:
    """Z: [n_items, E, V] expert logits of the fitting items; rows are (item,
    intermediate) pairs. Returns (unit-L2 weights [E], exact train macro)."""
    n, E, V = Z.shape
    Z = Z.clone()
    if exclude is not None:
        spare = torch.zeros(V, dtype=torch.bool)
        for c in row_cands:
            spare[list(c)] = True
        Z[:, :, exclude & ~spare] = -1e4
    ev_names, ev_idx = _pair_tables(None, row_eval)
    item = torch.tensor(row_item)
    maxc = max(len(c) for c in row_cands)
    cidx = torch.zeros(len(row_cands), maxc, dtype=torch.long)
    cmask = torch.zeros(len(row_cands), maxc, dtype=torch.bool)
    for r, c in enumerate(row_cands):
        cidx[r, :len(c)] = torch.tensor(list(c))
        cmask[r, :len(c)] = True
    valid = (Z[:, 0] > -1e3).float()                       # [n, V] (exclusion is expert-independent)
    Zf = Z.permute(1, 0, 2).reshape(E, n * V).contiguous()  # [E, n*V]: one GEMV per step
    del Z

    def logits(w):
        return (w @ Zf).view(n, V)

    def macro(score_rows):
        return torch.stack([score_rows[ev_idx == j].mean() for j in range(len(ev_names))]).mean()

    def exact(w):
        with torch.no_grad():
            z = logits(w)[item]
            best = torch.where(cmask, z.gather(1, cidx), torch.full_like(cidx, -1e9, dtype=z.dtype)).max(1).values
            rank = (z > best[:, None]).sum(1) + 1
        return float(macro((rank <= k).float()))

    single = []
    for e in range(E):
        w = torch.zeros(E)
        w[e] = 1.0
        single.append(exact(w))
    order = sorted(range(E), key=lambda e: (-single[e], e))
    starts = {"uniform": torch.ones(E), "select_best": torch.eye(E)[order[0]],
              "mean_top": torch.zeros(E).index_fill_(0, torch.tensor(order[:top_n]), 1.0)}
    best_w, best_score = None, -1.0
    for name, w0 in starts.items():
        theta = (w0 / w0.norm()).clone().requires_grad_(True)
        opt = torch.optim.Adam([theta], lr=lr)
        for step in range(steps + 1):
            w = theta / theta.norm()
            if step % exact_every == 0 or step == steps:
                sc = exact(w.detach())
                if sc > best_score:
                    best_score, best_w = sc, w.detach().clone()
            if step == steps:
                break
            # Full-vocab logits without autograd; the surrogate's gradient only
            # involves the best-candidate and k-th-largest entries of each row
            # (exactly what autograd through max/topk gives), so we gather those
            # two columns of Z and differentiate a tiny [rows, E] graph. The
            # per-row logit spread is treated as a constant within a step.
            with torch.no_grad():
                zi = logits(w)[item]
                cand_logits = torch.where(cmask, zi.gather(1, cidx), torch.full_like(cidx, -1e9, dtype=zi.dtype))
                best_col = cidx.gather(1, cand_logits.argmax(1, keepdim=True))[:, 0]
                kth_col = zi.topk(k, dim=1).indices[:, -1]
                vi = valid[item]
                mu = (zi * vi).sum(1) / vi.sum(1)
                spread = (((zi - mu[:, None]) ** 2 * vi).sum(1) / vi.sum(1)).sqrt()
                Zv = Zf.view(E, n, V)
                D = Zv[:, item, best_col] - Zv[:, item, kth_col]          # [E, rows]
            soft = torch.sigmoid((w @ D) / (temperature * spread))
            loss = -macro(soft)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return best_w, best_score


def combine(experts: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.einsum("e,edk->dk", w.float(), experts.float())


class RoutedLens(Lens):
    """Hard-routed expert lens: unembed(J_{c(h)} h)."""

    def __init__(self, experts: dict[int, torch.Tensor], router, name: str = "routed"):
        super().__init__("jlens", {l: e.mean(0) for l, e in experts.items()}, name=name)
        self.experts = {l: e.float() for l, e in experts.items()}
        self.router = router

    def transport(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        h = h.float()
        shp = h.shape
        hf = h.reshape(-1, shp[-1])
        a = self.router.assign(hf, layer)
        out = torch.einsum("ndk,nk->nd", self.experts[layer][a], hf)
        return out.reshape(shp)
