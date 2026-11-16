"""Validate exp2.KLTerm (top-K by norm + sampled tail) against the full-vocabulary KL, and time it."""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp2 import KLTerm, fit_inputs, kl_positions  # noqa: E402
from harness import Ctx  # noqa: E402
from jppsearch import fit_weights  # noqa: E402

ctx = Ctx(threads=2)
for l in (4, 21):
    fit_idx, _ = ctx.split(0)
    its, hit, false, L = fit_inputs(ctx, fit_idx, l, 0)
    w_jpp = fit_weights(L, hit, ctx.excl).w
    del L
    H, Y = kl_positions(ctx, l)
    kl = KLTerm(ctx, l, H, Y)
    g = torch.Generator().manual_seed(0)
    us = [w_jpp, torch.ones(8) / 8 ** 0.5, ctx.counts[l] / ctx.counts[l].norm()] + \
         [(lambda x: x / x.norm())(torch.randn(8, generator=g)) for _ in range(5)] + [torch.eye(8)[e] for e in range(8)]
    pairs = kl.validate(us)
    errs = [a - b for a, b in pairs]
    print(f"layer {l}: error at J++/uniform/R-lens {errs[0]:+.4f}/{errs[1]:+.4f}/{errs[2]:+.4f}; "
          f"max |error| over {len(us)} vectors {max(abs(e) for e in errs):.4f} nats (KL range {min(b for _, b in pairs):.2f}-{max(b for _, b in pairs):.2f})")
    t0 = time.time()
    for _ in range(10):
        kl.value_grad(w_jpp)
    print(f"   value+grad {(time.time() - t0) / 10 * 1000:.0f} ms per call (2 threads, shared CPU)")
