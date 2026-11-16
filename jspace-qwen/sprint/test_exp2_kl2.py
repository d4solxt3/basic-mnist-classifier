"""Where does the truncated log-partition fail, and does a sampled tail correction fix it?"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp2 import KLTerm, fit_inputs, kl_positions  # noqa: E402
from harness import Ctx  # noqa: E402
from jppsearch import fit_weights  # noqa: E402

ctx = Ctx(threads=2)
l = 4
fit_idx, _ = ctx.split(0)
its, hit, false, L = fit_inputs(ctx, fit_idx, l, 0)
w_jpp = fit_weights(L, hit, ctx.excl).w
del L
H, Y = kl_positions(ctx, l)
kl = KLTerm(ctx, l, H, Y, K=8192)
Lfull = kl._Lfull_for_check
N, E, V = Lfull.shape
g = torch.Generator().manual_seed(0)
names = ["J++", "uniform", "R-lens"] + [f"rand{i}" for i in range(5)] + [f"e{e}" for e in range(8)]
us = [w_jpp, torch.ones(8) / 8 ** 0.5, ctx.counts[l] / ctx.counts[l].norm()] + \
     [(lambda x: x / x.norm())(torch.randn(8, generator=g)) for _ in range(5)] + [torch.eye(8)[e] for e in range(8)]
# tail-sampling variant: top-K by norm plus R random tail tokens per position, weighted by |tail| / R
K, R = 8192, 8192
idx_top = Lfull.norm(dim=1).topk(K, dim=-1).indices
mask = torch.ones(N, V, dtype=torch.bool)
mask.scatter_(1, idx_top, False)
tail_idx = torch.stack([torch.nonzero(mask[n])[:, 0][torch.randperm(int(mask[n].sum()), generator=g)[:R]] for n in range(N)])
L_tail = Lfull.gather(2, tail_idx[:, None, :].expand(-1, E, -1))
w_tail = float(V - K) / R


def kl_tail(u):
    r = (torch.einsum("e,nef,f->n", u, kl.G, u) / kl.d + kl.eps).sqrt()
    z_top = torch.einsum("e,nev->nv", u, kl.L) / r[:, None]
    z_tail = torch.einsum("e,nev->nv", u, L_tail) / r[:, None] + torch.log(torch.tensor(w_tail))
    lse = torch.logsumexp(torch.cat([z_top, z_tail], dim=1), dim=1)
    return float((kl.negH - (kl.m @ u) / r + lse).mean())


with torch.no_grad():
    for nm, u in zip(names, us):
        a, b = kl.validate([u])[0]
        print(f"{nm:>8}: full {b:7.3f}  top-K {a:7.3f} (err {a - b:+.3f})  top-K+tail {kl_tail(u):7.3f} (err {kl_tail(u) - b:+.4f})")
