"""Clamp-swap interventions along lens directions (J++ / Gurnee et al. protocol).

At every band layer and every prompt position except 0 (the attention sink),
with unit-normalised directions u_s (source concept) and u_t (target concept)
and basis V = [u_s, u_t]:

    c   = V^+ h_clean                      (clean-pass least-squares coordinates)
    c*  = scale * (c_t, c_s)               (coordinates exchanged)
    h  <- h + V (c* - V^+ h)

The component of h orthogonal to span(V) is untouched. Directions come from
``Lens.vectors``: (g * W_U[t]) J_l for the J-lens and g * W_U[t] for the logit
lens, so the logit lens is a matched null that differs *only* in the Jacobian
transport.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from jspace.lens import Lens
from jspace.model import Editor, LensModel, Recorder

NEAR_PARALLEL = 0.999


@dataclass
class SwapResult:
    success_rank: int
    baseline_rank: int
    success_logprob: float
    baseline_logprob: float
    top: str
    n_layers_edited: int
    edit_norm_ratio: float


def _min_rank(logits: torch.Tensor, ids: Sequence[int], mask: torch.Tensor | None = None) -> int:
    z = logits.clone()
    if mask is not None:
        m = mask.clone()
        m[list(ids)] = False
        z[m] = -float("inf")
    best = z[list(ids)].max()
    return int((z > best).sum()) + 1


def _logprob(logits: torch.Tensor, ids: Sequence[int]) -> float:
    lp = torch.log_softmax(logits.float(), -1)
    return float(torch.logsumexp(lp[list(ids)], 0))


@torch.no_grad()
def clean_pass(model: LensModel, ids: torch.Tensor, layers: Sequence[int]):
    with Recorder(model, layers) as rec:
        out = model.forward(ids)
    acts = {l: rec.acts[l][0].float().clone() for l in layers}
    logits = model.lm_head(out.last_hidden_state[0, -1]).float()
    return logits, acts


@torch.no_grad()
def swap(model: LensModel, ids: torch.Tensor, lens: Lens, layers: Sequence[int], src_id: int, tgt_id: int,
         success_ids: Sequence[int], baseline_ids: Sequence[int], *, scale: float = 1.0,
         clean=None, word_mask: torch.Tensor | None = None) -> SwapResult:
    logits0, acts = clean if clean is not None else clean_pass(model, ids, layers)
    vs = lens.vectors(model, src_id, layers)
    vt = lens.vectors(model, tgt_id, layers)
    T = ids.shape[1]
    pos = torch.arange(1, T)
    edits = {}
    n_edit = 0
    for l in layers:
        us, ut = vs[l] / vs[l].norm(), vt[l] / vt[l].norm()
        if float((us @ ut).abs()) > NEAR_PARALLEL:
            continue
        V = torch.stack([us, ut], 1)                       # [d, 2]
        P = torch.linalg.pinv(V)                           # [2, d]
        c = acts[l][pos] @ P.T                             # [P, 2]
        cstar = scale * c[:, [1, 0]]

        def f(h, V=V, P=P, cstar=cstar):
            h2 = h.clone()
            sel = h2[:, pos, :].float()
            cur = sel @ P.T
            h2[:, pos, :] = (sel + (cstar - cur) @ V.T).to(h.dtype)
            return h2

        edits[l] = f
        n_edit += 1
    rec_layers = list(layers)
    with Editor(model, edits), Recorder(model, rec_layers) as rec:
        out = model.forward(ids)
    logits = model.lm_head(out.last_hidden_state[0, -1]).float()
    ratios = []
    for l in rec_layers:
        a = rec.acts[l][0, -1].float()
        b = acts[l][-1]
        ratios.append(float((a - b).norm() / b.norm()))
    ratios.sort()
    return SwapResult(
        success_rank=_min_rank(logits, success_ids, word_mask),
        baseline_rank=_min_rank(logits, baseline_ids, word_mask),
        success_logprob=_logprob(logits, success_ids),
        baseline_logprob=_logprob(logits, baseline_ids),
        top=model.tokenizer.decode([int(logits.argmax())]) if model.tokenizer else "",
        n_layers_edited=n_edit,
        edit_norm_ratio=ratios[len(ratios) // 2] if ratios else 0.0,
    )
