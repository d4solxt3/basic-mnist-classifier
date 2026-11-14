"""Correctness checks on a tiny randomly initialised Qwen3 (no downloads).

1. Our exact estimator (the reduction of anthropics/jacobian-lens: one-hot
   cotangents at every valid target, mean over valid sources) equals the same
   reduction applied to a brute-force full autograd Jacobian.
2. Exact J_n == I + Delta_n: the diagonal identity path is present.
3. The sketch estimators are unbiased: averaging many independent sketches of
   one prompt converges to the exact J_n at the predicted rate
   sqrt((d/K - 1) / m) * ||J_n - J0|| / ||J_n||.

Run:  python scripts/test_estimators.py
"""

import math
import sys

import torch

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
from jspace.fitting import PromptGraph, exact_prompt_jacobian, haar_frame, readout_shaping  # noqa: E402
from jspace.model import load_model  # noqa: E402

torch.manual_seed(0)
cfg = dict(vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
           num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=256,
           tie_word_embeddings=True)
model = load_model("qwen3-like", random_init=True, dtype=torch.float32, config_overrides=cfg)
d, L = model.d_model, model.n_layers
ids = torch.randint(0, 512, (1, 48))
src = [0, 1, 2]
tgt = L - 1

J = exact_prompt_jacobian(model, ids, src, tgt, dim_batch=8)
print("exact J computed; ||J_1|| / sqrt(d) =", float(J[1].norm() / math.sqrt(d)))

# --- 1. brute-force ground truth ---------------------------------------------
# Full Jacobian d h_L[all positions] / d h_l[all positions] by autograd on the
# block stack (independent of the VJP bookkeeping), then the reference
# reduction: sum over valid targets, mean over valid sources.
from jspace.fitting import valid_positions  # noqa: E402
from jspace.model import Recorder  # noqa: E402

T = ids.shape[1]
P = valid_positions(T)
for l in src:
    with Recorder(model, [l]) as rec, torch.no_grad():
        model.forward(ids)
    h_l = rec.acts[l].detach().clone()

    def run_from(h, l=l):
        # Replace block l's output with h and return the final block output.
        def edit(module, inputs, output):
            return h if torch.is_tensor(output) else (h, *output[1:])

        hk = model.layers[l].register_forward_hook(edit)
        try:
            with Recorder(model, [tgt]) as r2:
                model.forward(ids)
        finally:
            hk.remove()
        return r2.acts[tgt]

    full = torch.autograd.functional.jacobian(run_from, h_l)  # [1,T,d,1,T,d]
    full = full[0, :, :, 0, :, :]                              # [T_out, d_out, T_in, d_in]
    Jbf = full[P][:, :, P].sum(0).mean(1)                       # sum targets, mean sources -> [d, d]
    err = float((J[l] - Jbf).norm() / Jbf.norm())
    print(f"  layer {l}: rel. diff estimator vs brute-force autograd = {err:.2e}")
    assert err < 1e-4, err
print("[ok] exact estimator == brute-force target-summed / source-averaged Jacobian")

# --- 2. identity path ----------------------------------------------------------
for l in src:
    # the p'=p diagonal block contributes exactly I; check J - I is not ~I-like
    # by verifying the estimator on a model with all blocks zeroed out.
    pass
with torch.no_grad():
    saved = [(p, p.data.clone()) for blk in model.layers for n, p in blk.named_parameters()
             if n.endswith(("o_proj.weight", "down_proj.weight"))]
    for p, _ in saved:
        p.zero_()
Jz = exact_prompt_jacobian(model, ids, src, tgt, dim_batch=8)
for p, v in saved:
    p.data.copy_(v)
for l in src:
    assert torch.allclose(Jz[l], torch.eye(d), atol=1e-6), "identity path missing"
print("[ok] with all block outputs zeroed, J_n == I exactly (identity path)")

# --- 3. sketch unbiasedness -----------------------------------------------------
A = model.unembed_matrix()
W, Winv = readout_shaping(A)
gen = torch.Generator().manual_seed(1)
for K in (4, 16):
    for cv, J0 in (("none", None), ("identity", torch.eye(d))):
        for shaped in (False, True):
            m = 400
            acc = {l: torch.zeros(d, d) for l in src}
            for _ in range(m):
                Q = haar_frame(d, K, gen)
                C = W @ Q if shaped else Q
                g = PromptGraph(model, ids, src, tgt, batch=K)
                G = g.vjp_rows(C.T.contiguous(), retain=False)
                for l in src:
                    resid = G[l] if J0 is None else G[l] - C.T @ J0
                    upd = (d / K) * (Q @ resid)
                    if shaped:
                        upd = Winv @ upd
                    acc[l] += upd if J0 is None else J0 + upd
            l = 1
            est = acc[l] / m
            err = float((est - J[l]).norm() / J[l].norm())
            M = J[l] - (0 if J0 is None else J0)
            pred = math.sqrt((d / K - 1) / m) * float(M.norm() / J[l].norm())
            rerr = float((A @ est - A @ J[l]).norm() / (A @ J[l]).norm())
            tag = f"K={K:>2} cv={cv:<8} shaped={shaped!s:<5}"
            extra = f" predicted(unshaped)={pred:.3f}" if not shaped else ""
            print(f"  {tag} rel.err(J)={err:.3f}  rel.err(AJ)={rerr:.3f}{extra}")
            if not shaped:
                assert err < 2.0 * pred + 1e-3, (err, pred)
print("[ok] sketch estimators unbiased, error matches sqrt((d/K-1)/m) theory")
