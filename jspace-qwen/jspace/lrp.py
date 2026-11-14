"""LRP ("R-Lens") backward rules for Qwen3, as detach-based forward surgery.

Reproduces the "rlens" scope of the J++ code (Ayonrinde & Lindsey, 2026; rules
adapted there from the R-Lens of Blank & Bhatia), for Qwen3's dense blocks:

* LN-rule: residual-stream RMSNorms (input_layernorm, post_attention_layernorm)
  detach the rsqrt(mean(x^2)+eps) factor;
* identity-rule: silu(z) = z * sigmoid(z) with the sigmoid detached;
* half-rule: the gate*up product p -> 0.5 p + 0.5 p.detach().

Attention (including its q/k norms) keeps standard gradients. Forward values are
unchanged, so routing and recorded activations are identical; only gradients
differ. The surgery must be live while the graph is *recorded*.
"""

from __future__ import annotations

import contextlib
from types import MethodType

import torch

from jspace.model import LensModel


def _rmsnorm_lrp(self, hidden_states):
    input_dtype = hidden_states.dtype
    hs = hidden_states.to(torch.float32)
    variance = hs.pow(2).mean(-1, keepdim=True)
    hs = hs * torch.rsqrt(variance + self.variance_epsilon).detach()
    return self.weight * hs.to(input_dtype)


def _mlp_lrp(self, x):
    z = self.gate_proj(x)
    act = z * torch.sigmoid(z).detach()
    p = act * self.up_proj(x)
    p = 0.5 * p + 0.5 * p.detach()
    return self.down_proj(p)


@contextlib.contextmanager
def lrp_rules(model: LensModel, enabled: bool = True):
    if not enabled:
        yield
        return
    saved = []
    for blk in model.layers:
        for mod, fn in ((blk.input_layernorm, _rmsnorm_lrp), (blk.post_attention_layernorm, _rmsnorm_lrp),
                        (blk.mlp, _mlp_lrp)):
            assert type(mod).__name__ in ("Qwen3RMSNorm", "Qwen3MLP"), type(mod).__name__
            if type(mod).__name__ == "Qwen3MLP":
                assert getattr(mod.config, "hidden_act", "silu") == "silu"
            saved.append((mod, mod.__dict__.get("forward")))
            object.__setattr__(mod, "forward", MethodType(fn, mod))
    try:
        yield
    finally:
        for mod, prev in reversed(saved):
            if prev is None:
                del mod.__dict__["forward"]
            else:
                object.__setattr__(mod, "forward", prev)
