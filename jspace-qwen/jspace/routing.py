"""Per-layer activation router (J++ "Jacobian filtering"): unit-normalise,
centre, project to the top-D principal components, k-means (k-means++ init)."""

from __future__ import annotations

import torch


class Router:
    def __init__(self, mean: dict, comps: dict, cents: dict):
        self.mean, self.comps, self.cents = mean, comps, cents
        self.E = next(iter(cents.values())).shape[0]

    def project(self, X: torch.Tensor, l: int) -> torch.Tensor:
        X = X.float()
        return (X / X.norm(dim=-1, keepdim=True) - self.mean[l]) @ self.comps[l]

    def assign(self, X: torch.Tensor, l: int) -> torch.Tensor:
        return torch.cdist(self.project(X, l), self.cents[l]).argmin(-1)

    def save(self, path):
        torch.save({"mean": self.mean, "comps": self.comps, "cents": self.cents}, path)

    @classmethod
    def load(cls, path):
        s = torch.load(path, map_location="cpu", weights_only=True)
        return cls({int(k): v for k, v in s["mean"].items()}, {int(k): v for k, v in s["comps"].items()},
                   {int(k): v for k, v in s["cents"].items()})


def kmeans(Z: torch.Tensor, E: int, *, iters: int = 100, seed: int = 0, restarts: int = 4) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    best, best_inertia = None, float("inf")
    for _ in range(restarts):
        C = Z[torch.randint(len(Z), (1,), generator=g)]
        for _ in range(E - 1):  # k-means++
            d2 = torch.cdist(Z, C).min(1).values.pow(2)
            C = torch.cat([C, Z[torch.multinomial(d2 / d2.sum(), 1, generator=g)]])
        for _ in range(iters):
            a = torch.cdist(Z, C).argmin(1)
            newC = torch.stack([Z[a == e].mean(0) if (a == e).any() else C[e] for e in range(E)])
            if torch.allclose(newC, C):
                break
            C = newC
        inertia = float(torch.cdist(Z, C).min(1).values.pow(2).sum())
        if inertia < best_inertia:
            best, best_inertia = C, inertia
    return best


def fit_router(acts: dict[int, torch.Tensor], *, E: int = 8, D: int = 64, seed: int = 0) -> Router:
    mean, comps, cents = {}, {}, {}
    for l, X in acts.items():
        U = X.float() / X.float().norm(dim=-1, keepdim=True)
        mu = U.mean(0)
        C = (U - mu).T @ (U - mu) / len(U)
        evals, evecs = torch.linalg.eigh(C.double())
        P = evecs[:, -D:].flip(1).float()
        Z = (U - mu) @ P
        mean[l], comps[l], cents[l] = mu, P, kmeans(Z, E, seed=seed)
    return Router(mean, comps, cents)
