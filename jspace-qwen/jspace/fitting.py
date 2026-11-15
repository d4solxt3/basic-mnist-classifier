"""Estimating the average Jacobian  J_l = E[ d h_final / d h_l ].

Exact estimator (identical to ``anthropics/jacobian-lens``)
-----------------------------------------------------------
For a prompt with valid positions P (positions >= 16, excluding the last), let

    J_n = (1/|P|) sum_{p in P} sum_{p' in P} d h_L[p'] / d h_l[p]     (d x d)

computed row-block by row-block: a one-hot cotangent e_i is placed at *every*
valid target position, one VJP gives row i of the target-summed Jacobian at
every source position, and rows are averaged over source positions. This takes
d_model VJPs per prompt. The lens is J = mean_n J_n.

Note J_n = I + Delta_n exactly: the p' = p term carries the identity through
the residual stream, and every other term is a derivative of block updates.

Sketch estimators (new)
-----------------------
With K << d probes per prompt, draw a Haar-random orthonormal Q (d x K), set
cotangents C = W Q for a symmetric positive-definite "shaping" matrix W, and
obtain G = C^T J_n with K VJPs. Then

    Jhat_n = J0 + (d/K) W^{-1} Q (G - C^T J0)

is unbiased for E[J_n] whenever J0 is independent of (Q_n, prompt n), because
(d/K) E[Q Q^T] = I. Choices:

* ``cv="none"``     J0 = 0        (plain sketch)
* ``cv="identity"`` J0 = I        (the exactly known residual path)
* ``cv="online"``   J0 = running mean of the previous estimates (martingale
  control variate: conditionally unbiased term by term). Unstable for
  n < (d/K - 2)/2: the running-mean error is re-amplified by (d/K - 1) at every
  step (paper, Proposition 3); its error peaks near n = d/K / 2.
* ``cv="shrink"``   J0 = I + lambda_n (running mean - I), lambda_n = n / (n + kappa),
  kappa = d/K by default: the online control variate shrunk toward the identity
  until the running mean is better than I. Never worse than J0 = I under the
  variance recursion; still unbiased.
* ``W = I``         uniform probes;  ``W = (A^T A + eps I)^{1/4}`` with
  A = W_U diag(g) concentrates probes on the output directions the unembedding
  reads ("readout-aware" importance sketch).

For W = I and orthonormal Q,  E||(d/K) Q Q^T M - M||_F^2 = (d/K - 1) ||M||_F^2,
so the per-prompt variance is  sigma^2 + (d/K - 1) E||J_n - J0||^2.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch

from jspace.model import LensModel, Recorder

SKIP_FIRST = 16


def valid_positions(seq_len: int, skip_first: int = SKIP_FIRST) -> torch.Tensor:
    if seq_len - 1 - skip_first <= 0:
        raise ValueError(f"prompt too short ({seq_len} tokens)")
    return torch.arange(skip_first, seq_len - 1)


def split_positions(seq_len: int, skip_first: int = SKIP_FIRST, targets: str = "all"
                    ) -> tuple[torch.Tensor, torch.Tensor]:
    """(source positions, target positions). ``"all"``: the reference estimator,
    sources = targets = the valid positions, so each source sums its influence
    over targets p' >= p, *including p itself* (identity path + own next-token
    prediction). ``"later"`` (broadcast lens, ours): sources are the first half
    of the valid positions and targets the second half, so every term has
    p' > p and only attention-carried influence on later positions remains."""
    pos = valid_positions(seq_len, skip_first)
    if targets == "all":
        return pos, pos
    if targets == "later":
        m = len(pos) // 2
        return pos[:m], pos[m:]
    raise ValueError(targets)


class PromptGraph:
    """One forward pass on ``input_ids`` replicated ``batch`` times, graph
    retained so that many VJP blocks can be pulled from it."""

    def __init__(self, model: LensModel, input_ids: torch.Tensor, source_layers: Sequence[int],
                 target_layer: int, batch: int, skip_first: int = SKIP_FIRST, targets: str = "all"):
        self.model = model
        self.source_layers = list(source_layers)
        self.target_layer = target_layer
        self.batch = batch
        self.seq_len = input_ids.shape[1]
        self.pos, self.tpos = split_positions(self.seq_len, skip_first, targets)
        self._rec = Recorder(model, [*self.source_layers, target_layer], root=min(self.source_layers))
        with self._rec, torch.enable_grad():
            model.forward(input_ids.expand(batch, -1))
        self.target = self._rec.acts[target_layer]
        self.sources = [self._rec.acts[l] for l in self.source_layers]

    def vjp_rows(self, cot_KD: torch.Tensor, *, retain: bool = True) -> dict[int, torch.Tensor]:
        """For each cotangent row c_k (K <= batch), return  c_k^T J_n  for every
        source layer, fp32 ``[K, d]``."""
        K, d = cot_KD.shape
        assert K <= self.batch
        cot = torch.zeros_like(self.target)
        c = cot_KD.to(cot.dtype)
        cot[:K, self.tpos, :] = c[:, None, :]
        with torch.enable_grad():
            grads = torch.autograd.grad(self.target, self.sources, grad_outputs=cot, retain_graph=retain)
        out = {}
        for l, g in zip(self.source_layers, grads):
            out[l] = g[:K, self.pos, :].float().mean(dim=1)
        return out


def batched_vjp_rows(model: LensModel, input_ids: torch.Tensor, cot_KD: torch.Tensor, source_layers: Sequence[int],
                     target_layer: int, skip_first: int = SKIP_FIRST) -> dict[int, torch.Tensor]:
    """Same quantity as ``PromptGraph(...).vjp_rows`` but with ONE batch-1
    forward and a vmapped backward (``is_grads_batched``), so the forward is
    not replicated K times. Equal to the replicated version up to bf16
    rounding (~6e-3 relative)."""
    T = input_ids.shape[1]
    pos = valid_positions(T, skip_first)
    rec = Recorder(model, [*source_layers, target_layer], root=min(source_layers))
    with rec, torch.enable_grad():
        model.forward(input_ids)
    tgt = rec.acts[target_layer]
    K = cot_KD.shape[0]
    cot = torch.zeros((K, *tgt.shape), dtype=tgt.dtype)
    cot[:, 0, pos, :] = cot_KD.to(tgt.dtype)[:, None, :]
    with torch.enable_grad():
        grads = torch.autograd.grad(tgt, [rec.acts[l] for l in source_layers], grad_outputs=cot,
                                    is_grads_batched=True)
    return {l: g[:, 0, pos, :].float().mean(1) for l, g in zip(source_layers, grads)}


def exact_prompt_jacobian(model: LensModel, input_ids: torch.Tensor, source_layers: Sequence[int],
                          target_layer: int, *, dim_batch: int = 16, skip_first: int = SKIP_FIRST,
                          router=None, cl_sum: dict | None = None, cl_count: dict | None = None,
                          lrp: bool = False, targets: str = "all") -> dict[int, torch.Tensor]:
    """Exact per-prompt J_n (reference reduction). If ``router`` is given, also
    adds per-position gradient rows into per-cluster sums ``cl_sum[l]``
    ``[E, d, d]`` (J++ expert Jacobians = cl_sum / cl_count). ``lrp`` records
    the graph under the R-Lens/J++ LRP rules."""
    from jspace.lrp import lrp_rules

    d = model.d_model
    with lrp_rules(model, lrp):
        g = PromptGraph(model, input_ids, source_layers, target_layer, dim_batch, skip_first, targets)
    J = {l: torch.zeros(d, d) for l in source_layers}
    onehot = {}
    if router is not None:
        for l in source_layers:
            a = router.assign(g.sources[source_layers.index(l)][0, g.pos].detach(), l)
            onehot[l] = torch.nn.functional.one_hot(a, router.E).float()      # [P, E]
            cl_count[l] += onehot[l].sum(0)
    starts = list(range(0, d, dim_batch))
    with torch.enable_grad():
        for i, s in enumerate(starts):
            k = min(dim_batch, d - s)
            cot = torch.zeros_like(g.target)
            cot[torch.arange(k)[:, None], g.tpos[None, :], (s + torch.arange(k))[:, None]] = 1.0
            grads = torch.autograd.grad(g.target, g.sources, grad_outputs=cot, retain_graph=i < len(starts) - 1)
            for l, gr in zip(source_layers, grads):
                rows_kPd = gr[:k, g.pos, :].float()
                J[l][s:s + k] = rows_kPd.mean(1)
                if router is not None:
                    cl_sum[l][:, s:s + k, :] += torch.einsum("kpd,pe->ekd", rows_kPd, onehot[l])
    return J


def haar_frame(d: int, K: int, gen: torch.Generator) -> torch.Tensor:
    """Haar-random d x K matrix with orthonormal columns."""
    X = torch.randn(d, K, generator=gen, dtype=torch.float64)
    Q, R = torch.linalg.qr(X)
    Q = Q * torch.sign(torch.diagonal(R))[None, :]
    return Q.float()


def readout_shaping(A: torch.Tensor, *, power: float = 0.25, eps_rel: float = 1e-3) -> tuple[torch.Tensor, torch.Tensor]:
    """W = (A^T A + eps I)^power and W^{-1}, for the readout-aware sketch."""
    M = (A.double().T @ A.double())
    evals, evecs = torch.linalg.eigh(M)
    evals = evals.clamp_min(0) + eps_rel * evals.max()
    W = (evecs * evals.pow(power)) @ evecs.T
    Winv = (evecs * evals.pow(-power)) @ evecs.T
    return W.float(), Winv.float()


@dataclass
class FitState:
    source_layers: list[int]
    target_layer: int
    d: int
    method: str
    K: int
    cv: str
    sum_: dict[int, torch.Tensor] = field(default_factory=dict)
    n: int = 0
    vjp_rows: int = 0          # total cotangent rows used (x tokens = backward cost)
    tokens_bwd: int = 0        # sum over prompts of rows * seq_len
    seconds: float = 0.0
    log: list = field(default_factory=list)
    cl_sum: dict | None = None
    cl_count: dict | None = None

    def mean(self) -> dict[int, torch.Tensor]:
        return {l: s / max(self.n, 1) for l, s in self.sum_.items()}


def fit(model: LensModel, prompts_ids: Sequence[torch.Tensor], *, source_layers: Sequence[int] | None = None,
        target_layer: int | None = None, method: str = "exact", K: int = 16, cv: str = "online",
        W: torch.Tensor | None = None, Winv: torch.Tensor | None = None, J_pilot: dict | None = None,
        kappa: float | None = None, dim_batch: int = 16, seed: int = 0, snapshots: Sequence[int] = (), snapshot_dir: str | None = None,
        save_per_prompt_layers: Sequence[int] = (), per_prompt_path: str | None = None,
        tag: str = "", verbose: bool = True, router=None, lrp: bool = False, targets: str = "all") -> FitState:
    L, d = model.n_layers, model.d_model
    target_layer = L - 1 if target_layer is None else target_layer
    source_layers = list(range(target_layer)) if source_layers is None else sorted(source_layers)
    st = FitState(source_layers, target_layer, d, method, K, cv)
    st.sum_ = {l: torch.zeros(d, d) for l in source_layers}
    if router is not None:
        st.cl_sum = {l: torch.zeros(router.E, d, d) for l in source_layers}
        st.cl_count = {l: torch.zeros(router.E) for l in source_layers}
    gen = torch.Generator().manual_seed(seed)
    eye = torch.eye(d)
    per_prompt: dict[int, list] = {l: [] for l in save_per_prompt_layers}
    snaps = set(snapshots)
    if snapshot_dir:
        os.makedirs(snapshot_dir, exist_ok=True)

    for idx, ids in enumerate(prompts_ids):
        t0 = time.perf_counter()
        T = ids.shape[1]
        if method == "exact":
            Jn = exact_prompt_jacobian(model, ids, source_layers, target_layer, dim_batch=dim_batch,
                                       router=router, cl_sum=st.cl_sum, cl_count=st.cl_count, lrp=lrp,
                                       targets=targets)
            rows = d
        elif method == "sketch":
            Q = haar_frame(d, K, gen)
            C = Q if W is None else W @ Q
            G = batched_vjp_rows(model, ids, C.T.contiguous(), source_layers, target_layer)
            Jn = {}
            for l in source_layers:
                if cv == "none":
                    J0 = None
                elif cv == "identity":
                    J0 = eye
                elif cv == "online":
                    J0 = eye if st.n == 0 else st.sum_[l] / st.n
                elif cv == "shrink":
                    kap = d / K if kappa is None else kappa
                    lam = st.n / (st.n + kap)
                    J0 = eye if st.n == 0 else eye + lam * (st.sum_[l] / st.n - eye)
                elif cv == "pilot":
                    J0 = J_pilot[l]
                else:
                    raise ValueError(cv)
                resid = G[l] if J0 is None else G[l] - C.T @ J0
                upd = (d / K) * (Q @ resid)
                if Winv is not None:
                    upd = Winv @ upd
                Jn[l] = upd if J0 is None else J0 + upd
            rows = K
        else:
            raise ValueError(method)

        for l in source_layers:
            st.sum_[l] += Jn[l]
        for l in save_per_prompt_layers:
            per_prompt[l].append(Jn[l].half())
        st.n += 1
        st.vjp_rows += rows
        st.tokens_bwd += rows * T
        dt = time.perf_counter() - t0
        st.seconds += dt
        st.log.append({"prompt": idx, "seq_len": T, "rows": rows, "sec": dt})
        if verbose and (idx < 3 or (idx + 1) % 10 == 0):
            print(f"[{tag}] prompt {idx + 1}/{len(prompts_ids)} T={T} rows={rows} {dt:.1f}s "
                  f"total={st.seconds / 60:.1f}min", flush=True)
        if snapshot_dir and st.n in snaps:
            save_lens_state(st, os.path.join(snapshot_dir, f"{tag}_n{st.n}.pt"))
            if st.cl_sum is not None:
                save_experts(st, os.path.join(snapshot_dir, f"{tag}_experts_n{st.n}.pt"))

    if per_prompt_path and save_per_prompt_layers:
        torch.save({l: torch.stack(v) for l, v in per_prompt.items()}, per_prompt_path)
    return st


def save_lens_state(st: FitState, path: str) -> None:
    tmp = path + ".tmp"
    torch.save({
        "J": {l: (s / st.n).half() for l, s in st.sum_.items()},
        "n": st.n, "method": st.method, "K": st.K, "cv": st.cv,
        "vjp_rows": st.vjp_rows, "tokens_bwd": st.tokens_bwd, "seconds": st.seconds,
        "source_layers": st.source_layers, "target_layer": st.target_layer, "d": st.d,
    }, tmp)
    os.replace(tmp, path)


def save_experts(st: FitState, path: str) -> None:
    """Per-cluster expert Jacobians (position-weighted means), fp16."""
    tmp = path + ".tmp"
    torch.save({"experts": {l: (s / st.cl_count[l].clamp_min(1)[:, None, None]).half() for l, s in st.cl_sum.items()},
                "counts": st.cl_count, "n": st.n}, tmp)
    os.replace(tmp, path)


def load_lens_state(path: str) -> dict:
    s = torch.load(path, map_location="cpu", weights_only=True)
    s["J"] = {int(l): J.float() for l, J in s["J"].items()}
    return s


def rel_frobenius(Ja: dict, Jb: dict, layers: Sequence[int] | None = None) -> dict[int, float]:
    layers = layers or sorted(Jb)
    return {l: float((Ja[l] - Jb[l]).norm() / Jb[l].norm()) for l in layers}


def readout_rel_error(Ja: dict, Jb: dict, A: torch.Tensor, layers: Sequence[int] | None = None) -> dict[int, float]:
    """Relative error of the J-space dictionary  A J  (what readouts and swap
    vectors actually use)."""
    layers = layers or sorted(Jb)
    out = {}
    for l in layers:
        Mb = A @ Jb[l]
        out[l] = float((A @ Ja[l] - Mb).norm() / Mb.norm())
    return out


def n_math(d: int, K: int) -> float:
    return math.sqrt(d / K - 1)
