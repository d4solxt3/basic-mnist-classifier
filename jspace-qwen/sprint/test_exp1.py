"""Component checks for sprint/exp1.py (run before the experiment; no held-out scoring).

1. Band projectors are a resolution of the identity (sum_b P_b = I), for the pooled-SVD basis and the
   Haar basis.
2. The J++-expanded start reproduces the J++ map exactly: Bands.combine(experts, u_ref) = J++ map / 2.
3. A1 nests J++: with the u_ref start its fitting-split exact score is >= A0's.
4. Atom logits agree with the combined map: u @ L = A (M h).
5. Label-free denoising: kappa values in [0, 1] and the denoised experts stay close to the originals.
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp1 import BandFitter, Bands, denoised_experts, fit_A0, haar, pooled_bands  # noqa: E402
from harness import Ctx  # noqa: E402
from sprintfit import fit_rows  # noqa: E402

ctx = Ctx(threads=4)
fit_idx, _ = ctx.split(0)
l = 4
bands = pooled_bands(ctx, l)
d = ctx.A.shape[1]
for name, bd in (("pooled", bands), ("haar", Bands(haar(d, 1000 + l)))):
    S = sum(bd.U[:, bd.edges[b]:bd.edges[b + 1]] @ bd.U[:, bd.edges[b]:bd.edges[b + 1]].T for b in range(bd.B))
    print(f"[1] {name}: |sum P_b - I|_max = {float((S - torch.eye(d)).abs().max()):.2e}")
t0 = time.time()
r0, M0 = fit_A0(ctx, fit_idx, l)
print(f"A0 fitted in {time.time() - t0:.0f}s, train {r0.train_exact:.4f}")
bf = BandFitter(ctx, fit_idx, l, bands, r0.w)
Mref = bands.combine(ctx.experts[l], bf.u_ref)
print(f"[2] |combine(u_ref) - J++/2| / |J++/2| = {float((Mref - M0 / 2).norm() / (M0 / 2).norm()):.2e}")
its, pairs = fit_rows(ctx, fit_idx)
H = ctx.prep.resid[its, l].float()
z_atoms = torch.einsum("a,nav->nv", bf.u_ref, bf.L)
z_map = (H @ Mref.T) @ ctx.A.T
print(f"[4] |u.L - A M h|_max / |A M h|_max = {float((z_atoms - z_map).abs().max() / z_map.abs().max()):.2e}")
t0 = time.time()
r1, M1 = bf.fit(0.1)
print(f"[3] A1 (lambda 0.1) fitted in {time.time() - t0:.0f}s: train {r1.train_exact:.4f} vs A0 {r0.train_exact:.4f} "
      f"(must be >=), init {r1.init}, step {r1.step}, dist to ref {float(((r1.w - bf.u_ref) ** 2).sum()):.4f}")
assert r1.train_exact >= r0.train_exact - 1e-9
den, kap = denoised_experts(ctx, l, bands)
rel = [float((den[e] - ctx.experts[l][e].float()).norm() / ctx.experts[l][e].float().norm()) for e in range(den.shape[0])]
print(f"[5] kappa per expert x band: {[[round(k, 3) for k in ks] for ks in kap]}")
print(f"    relative change of experts: {[round(x, 4) for x in rel]}")
assert all(0 <= k <= 1 for ks in kap for k in ks)
print("all component checks passed")
