"""Lenses and readouts.

Three linear lenses share one interface, ``Lens.logits(model, h, layer)``:

* ``logit``   : unembed(h)                       (nostalgebraist, 2020)
* ``jlens``   : unembed(J_l h)                   (Gurnee et al., 2026)
* ``regress`` : unembed(B_l h + c_l)             least-squares map onto the
  *current* final residual -- a cheap tuned-lens-style baseline that, by
  construction, predicts what the model is about to output.

Readout calibration (new): lens logits contain a context-independent prior
(e.g. frequent tokens rank high at every position). ``Calibration`` stores per-
token mean and std of each lens's logits over corpus positions; a calibrated
readout ranks tokens by the z-score  (z_t - m_t) / s_t . It is a fixed affine
map of the logits, so it costs nothing at inference.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from jspace.model import LensModel


class Lens:
    def __init__(self, kind: str, maps: dict[int, torch.Tensor] | None = None,
                 bias: dict[int, torch.Tensor] | None = None, name: str | None = None):
        assert kind in ("logit", "jlens", "regress")
        self.kind = kind
        self.maps = {int(k): v.float() for k, v in (maps or {}).items()}
        self.bias = {int(k): v.float() for k, v in (bias or {}).items()}
        self.name = name or kind

    def has_layer(self, layer: int) -> bool:
        return self.kind == "logit" or layer in self.maps

    def transport(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        h = h.float()
        if self.kind == "logit":
            return h
        out = h @ self.maps[layer].T
        if self.kind == "regress":
            out = out + self.bias[layer]
        return out

    def logits(self, model: LensModel, h: torch.Tensor, layer: int) -> torch.Tensor:
        """Lens logits ``[..., vocab]`` (fp32) through the model's own final
        norm and LM head."""
        return model.unembed_f32(self.transport(h, layer))

    def vectors(self, model: LensModel, token_id: int, layers: Sequence[int]) -> dict[int, torch.Tensor]:
        """Per-layer readout direction for one token: ``v = (g * W_U[t]) @ J_l``
        (identity transport for the logit lens). ``<h, v>`` is the token's
        pre-RMS lens logit."""
        u = (model.lm_head.weight[token_id].float() * model.final_norm.weight.float())
        out = {}
        for l in layers:
            out[l] = u.clone() if self.kind == "logit" else u @ self.maps[l]
        return out

    def save(self, path: str) -> None:
        torch.save({"kind": self.kind, "name": self.name,
                    "maps": {l: m.half() for l, m in self.maps.items()},
                    "bias": {l: b.float() for l, b in self.bias.items()}}, path)

    @classmethod
    def load(cls, path: str) -> Lens:
        s = torch.load(path, map_location="cpu", weights_only=True)
        if "maps" not in s and "J" in s:  # a fitting snapshot
            return cls("jlens", s["J"], name=path)
        return cls(s["kind"], s["maps"], s.get("bias"), name=s.get("name"))


def fit_regression_lens(acts: dict[int, torch.Tensor], target: torch.Tensor, *, ridge: float = 1e-3) -> Lens:
    """Least squares  h_L ~ B h_l + c  over corpus positions.

    ``acts[l]``: ``[N, d]`` residuals at layer l; ``target``: ``[N, d]`` final
    residuals at the same positions."""
    maps, bias = {}, {}
    Y = target.double()
    my = Y.mean(0)
    for l, X in acts.items():
        X = X.double()
        mx = X.mean(0)
        Xc, Yc = X - mx, Y - my
        C = Xc.T @ Xc
        lam = ridge * C.diagonal().mean()
        B = torch.linalg.solve(C + lam * torch.eye(C.shape[0], dtype=C.dtype), Xc.T @ Yc).T  # [d_out, d_in]
        maps[l] = B.float()
        bias[l] = (my - B @ mx).float()
    return Lens("regress", maps, bias, name="regress")


@dataclass
class Calibration:
    """Per-(lens, layer) token statistics of lens logits over corpus positions."""

    mean: dict[int, torch.Tensor]
    std: dict[int, torch.Tensor]

    def apply(self, logits: torch.Tensor, layer: int, mode: str) -> torch.Tensor:
        if mode == "raw":
            return logits
        if mode == "center":
            return logits - self.mean[layer]
        if mode == "z":
            return (logits - self.mean[layer]) / self.std[layer]
        raise ValueError(mode)

    def save(self, path: str) -> None:
        torch.save({"mean": self.mean, "std": self.std}, path)

    @classmethod
    def load(cls, path: str) -> Calibration:
        s = torch.load(path, map_location="cpu", weights_only=True)
        return cls({int(k): v for k, v in s["mean"].items()}, {int(k): v for k, v in s["std"].items()})


@torch.no_grad()
def fit_calibration(model: LensModel, lens: Lens, acts: dict[int, torch.Tensor], *, chunk: int = 512,
                    std_floor_rel: float = 0.05, max_positions: int = 3072, seed: int = 0) -> Calibration:
    """Mean/std of each token's lens logit over (a random subset of) the
    positions in ``acts``."""
    mean, std = {}, {}
    g = torch.Generator().manual_seed(seed)
    for l, X in acts.items():
        if X.shape[0] > max_positions:
            X = X[torch.randperm(X.shape[0], generator=g)[:max_positions]]
        s1 = torch.zeros(model.vocab_size, dtype=torch.float64)
        s2 = torch.zeros(model.vocab_size, dtype=torch.float64)
        for i in range(0, X.shape[0], chunk):
            z = lens.logits(model, X[i:i + chunk], l).double()
            s1 += z.sum(0)
            s2 += (z * z).sum(0)
        n = X.shape[0]
        m = s1 / n
        v = (s2 / n - m * m).clamp_min(0)
        sd = v.sqrt()
        sd = sd.clamp_min(std_floor_rel * sd.median())
        mean[l], std[l] = m.float(), sd.float()
    return Calibration(mean, std)


class AffineLens(Lens):
    """First-order Taylor transport around the corpus mean (ours):
        mode="affine":   h_L ~ mu_L + J (h - mu_l)
        mode="centered": J (h - mu_l)
    The plain J-lens is J h = J (h - mu_l) + J mu_l, i.e. it replaces the
    true final-layer mean mu_L by the transported mean J mu_l. Directions
    (``vectors``) are those of the underlying J-lens."""

    def __init__(self, base: Lens, mu_in: dict[int, torch.Tensor], mu_out: torch.Tensor,
                 mode: str = "affine", alpha: float = 1.0, name: str | None = None):
        maps = base.maps
        if alpha != 1.0:
            maps = {l: torch.eye(J.shape[0]) + alpha * (J - torch.eye(J.shape[0])) for l, J in maps.items()}
        super().__init__("jlens", maps, name=name or f"{base.name}-{mode}")
        self.mu_in = {int(l): m.float() for l, m in mu_in.items()}
        self.mu_out = mu_out.float()
        self.mode = mode

    def transport(self, h: torch.Tensor, layer: int) -> torch.Tensor:
        h = h.float()
        if self.mode == "plain":
            return h @ self.maps[layer].T
        out = (h - self.mu_in[layer]) @ self.maps[layer].T
        if self.mode == "affine":
            out = out + self.mu_out
        return out
