"""Component checks for sprint/exp2.py (light: one layer, run alongside Exp 1 with 2 threads).

1. fit_multi with gamma = beta = 0 reproduces fit_weights (J++'s search) exactly.
2. KL at the J++ weights is of the size the harness measures on held-out WikiText (~6.5 nats at early layers).
3. The control-pair table is non-empty and its rows point at fitting items of the same eval.
4. Timing of one B2 search (gamma=1, beta=0.1) and of the KL-only fit.
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp2 import KLTerm, fit_inputs, fit_kl_only, fit_multi, kl_positions  # noqa: E402
from harness import Ctx  # noqa: E402
from jppsearch import fit_weights  # noqa: E402

ctx = Ctx(threads=2)
l, seed = 4, 0
fit_idx, _ = ctx.split(seed)
its, hit, false, L = fit_inputs(ctx, fit_idx, l, seed)
print(f"[3] fitting pairs {hit.item.numel()}, control pairs {false.item.numel() if false else 0}")
t0 = time.time()
r_ref = fit_weights(L, hit, ctx.excl)
t_ref = time.time() - t0
r_m = fit_multi(L, hit, ctx.excl)
print(f"[1] fit_weights train {r_ref.train_exact:.4f} vs fit_multi(0,0) {r_m['train_hit']:.4f}; "
      f"cos w {float(r_ref.w @ r_m['w']):.6f}; {t_ref:.0f}s")
H, Y = kl_positions(ctx, l)
t0 = time.time()
kl = KLTerm(ctx, l, H, Y)
print(f"    KL term built in {time.time() - t0:.0f}s")
print(f"[2] KL at J++ weights {kl.value(r_ref.w):.3f} nats; model entropy {-kl.Hp:.3f}")
t0 = time.time()
r2 = fit_multi(L, hit, ctx.excl, false=false, gamma=1.0, kl=kl, beta=0.1)
print(f"[4] B2 search {time.time() - t0:.0f}s: train hit {r2['train_hit']:.4f} false {r2['train_false']:.4f} kl {r2['kl']:.3f} "
      f"init {r2['init']} step {r2['step']}")
t0 = time.time()
b1 = fit_kl_only(kl, ctx.counts[l])
print(f"[4] KL-only fit {time.time() - t0:.0f}s: kl {b1['kl']:.3f} init {b1['init']} step {b1['step']}")
