"""Weights-only predictor of the benefit of readout-aware probe shaping.

For A = W_U diag(g) (the readout's linear part), Proposition 3 gives the
variance factor of the shaped sketch W = (A^T A)^{1/4} relative to uniform
probes (W = I), under an isotropic prior on J_n - J_0:

    rho(A) = (sum_i s_i)^2 / (d * sum_i s_i^2),   s_i = singular values of A.

The factor for a general SPD shaping W, with Z = W^2, is
    tr(Z) tr(Z^{-1} A^T A) / (d tr(A^T A)),
which we also evaluate for the regularised W that fit.py actually uses
(readout_shaping: (A^T A + eps I)^{1/4}, eps = 1e-3 * max eigenvalue).

python scripts/rho.py
"""

import json
import os
import sys

import torch

ROOT = __file__.rsplit("/scripts/", 1)[0]
sys.path.insert(0, ROOT)
from jspace import paths as P  # noqa: E402
from jspace.model import load_model  # noqa: E402

torch.set_num_threads(int(os.environ.get("RHO_THREADS", "1")))
model = load_model("Qwen/Qwen3-0.6B")
A = model.unembed_matrix().double()
d = A.shape[1]
M = A.T @ A                                   # A^T A, [d, d]
ev = torch.linalg.eigvalsh(M).clamp_min(0)    # squared singular values
s = ev.sqrt()
rho = float(s.sum() ** 2 / (d * (s ** 2).sum()))


def factor(z_eigs):
    """tr(Z) tr(Z^{-1} A^T A) / (d tr(A^T A)) for Z diagonal in A^T A's eigenbasis."""
    return float(z_eigs.sum() * (ev / z_eigs).sum() / (d * ev.sum()))


eps = 1e-3 * ev.max()
out = {
    "d": d,
    "rho_ideal": rho,
    "rho_implemented": factor((ev + eps).sqrt()),   # W = (A^T A + eps I)^{1/4} -> Z = W^2
    "check_identity_factor": factor(torch.ones_like(ev)),  # = 1 by construction
    "singular_values": {"max": float(s.max()), "median": float(s.median()), "min": float(s.min()),
                        "effective_rank": float(s.sum() ** 2 / (s ** 2).sum())},
}
print(json.dumps(out, indent=1))
os.makedirs(P.RESULTS, exist_ok=True)
with open(os.path.join(P.RESULTS, "rho.json"), "w") as f:
    json.dump(out, f, indent=1)
