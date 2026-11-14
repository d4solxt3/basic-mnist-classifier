"""Readout evaluation on the six released lens-eval prompt sets.

Conventions follow the released data READMEs and the J++ evaluation code:

* readout position: final prompt token (multihop, multilingual, order-ops,
  association, typo) or the last newline token (poetry);
* an intermediate's rank is the min over its single-token surface forms
  (with/without leading space; original, capitalised, lowercase casing;
  order-ops numbers/operations expanded to synonyms);
* rank = 1 + #(logits strictly greater than the best candidate);
* only items the model answers correctly are scored (multihop, multilingual,
  order-ops via ``target``; poetry via its rhyme word); association and typo
  pose no question and are always kept;
* recall@k for an item = fraction of its intermediates with min-over-layers
  rank <= k; reported per eval and macro-averaged over the five macro evals
  (order-ops is reported separately, as in J++, because Qwen splits digits).
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from jspace.lens import Calibration, Lens
from jspace.model import LensModel, Recorder

ORDER_OPS_SYNONYMS: dict[str, tuple[str, ...]] = {
    "3": ("3", "three"), "4": ("4", "four"), "5": ("5", "five"), "6": ("6", "six"),
    "7": ("7", "seven"), "8": ("8", "eight"), "9": ("9", "nine"), "10": ("10", "ten"),
    "11": ("11", "eleven"), "12": ("12", "twelve", "dozen"), "13": ("13", "thirteen"),
    "15": ("15", "fifteen"), "16": ("16", "sixteen"), "20": ("20", "twenty"),
    "24": ("24", "twenty-four"),
    "addition": ("addition", "add", "plus", "sum", "and", "+"),
    "subtraction": ("subtraction", "subtract", "minus", "difference", "take", "diff", "-"),
    "multiplication": ("multiplication", "multiply", "times", "product", "mult", "prod", "by", "*", "×"),
    "division": ("division", "divide", "quotient", "/", "÷"),
    "mod": ("mod", "modulo", "remainder", "%"),
    "squared": ("squared", "square", "power", "exponent", "^", "**"),
}

SPECS = {  # eval -> (readout rule, expand synonyms, correctness target source)
    "multihop": ("final_token", False, "target"),
    "multilingual": ("final_token", False, "target"),
    "order-ops": ("final_token", True, "target"),
    "poetry": ("last_newline", False, "first_intermediate"),
    "association": ("final_token", False, None),
    "typo": ("final_token", False, None),
}
MACRO_EVALS = ("multihop", "multilingual", "typo", "association", "poetry")


@dataclass(frozen=True)
class Item:
    eval: str
    name: str
    prompt: str
    target: str | None
    intermediates: tuple[str, ...]

    def correctness_target(self) -> str | None:
        src = SPECS[self.eval][2]
        if src == "target":
            return self.target
        if src == "first_intermediate":
            return self.intermediates[0]
        return None


def load_items(data_dir: str, evals: Sequence[str] = tuple(SPECS)) -> list[Item]:
    items = []
    for ev in evals:
        with open(os.path.join(data_dir, f"lens-eval-{ev}.json")) as f:
            for it in json.load(f)["items"]:
                items.append(Item(ev, it["name"], it["prompt"], it.get("target"), tuple(it["intermediates"])))
    return items


def surface_forms(word: str) -> list[str]:
    cased = {word, word.capitalize(), word.lower()}
    return [p + w for w in sorted(cased) for p in ("", " ")]


def single_token_ids(tok, surfaces: Sequence[str]) -> list[int]:
    out = set()
    for s in surfaces:
        ids = tok(s, add_special_tokens=False).input_ids
        if len(ids) == 1:
            out.add(int(ids[0]))
    return sorted(out)


def candidate_ids(tok, word: str, expand: bool) -> list[int]:
    syn = ORDER_OPS_SYNONYMS.get(word, (word,)) if expand else (word,)
    return single_token_ids(tok, [s for w in syn for s in surface_forms(w)])


def resolve_token(tok, concept: str) -> int | None:
    """One concrete id for building a swap direction (J++ convention: prefer
    the leading-space form in the concept's own casing)."""
    c = concept.strip()
    for s in [" " + c, c, " " + c.lower(), c.lower(), " " + c.capitalize(), c.capitalize()]:
        ids = tok(s, add_special_tokens=False).input_ids
        if len(ids) == 1:
            return int(ids[0])
    return None


def readout_position(ids: torch.Tensor, tok, rule: str) -> int:
    T = ids.shape[1]
    if rule == "final_token":
        return T - 1
    nl = [p for p in range(T) if "\n" in tok.decode([int(ids[0, p])])]
    return nl[-1] if nl else T - 1


def next_token_matches(decoded: str, target: str) -> bool:
    s = decoded.lstrip()
    return bool(s) and target.strip().startswith(s)


def non_semantic_ids(tok, vocab_size: int) -> torch.Tensor:
    """Bool mask: tokens with no letter or digit (J++ Readout Filtering)."""
    dec = tok.batch_decode([[i] for i in range(vocab_size)])
    return torch.tensor([not any(ch.isalnum() for ch in s) for s in dec])


@dataclass
class Prepared:
    items: list[Item]
    resid: torch.Tensor            # [N, L, d] residual at the readout position, every layer
    cands: list[dict[str, list[int]]]
    correct: list[bool | None]
    model_top: list[str]


@torch.no_grad()
def prepare(model: LensModel, items: Sequence[Item], *, max_len: int = 512) -> Prepared:
    tok = model.tokenizer
    keep, resid, cands, correct, tops = [], [], [], [], []
    layers = list(range(model.n_layers))
    for it in items:
        rule, expand, _ = SPECS[it.eval]
        c = {w: candidate_ids(tok, w, expand) for w in it.intermediates}
        c = {w: v for w, v in c.items() if v}
        if not c:
            continue
        ids = model.encode(it.prompt, max_length=max_len)
        pos = readout_position(ids, tok, rule)
        with Recorder(model, layers) as rec:
            out = model.forward(ids)
        h = torch.stack([rec.acts[l][0, pos].float() for l in layers])
        logits = model.lm_head(out.last_hidden_state[0, -1]).float()
        nxt = tok.decode([int(logits.argmax())])
        tgt = it.correctness_target()
        correct.append(None if tgt is None else next_token_matches(nxt, tgt))
        keep.append(it)
        resid.append(h)
        cands.append(c)
        tops.append(nxt)
    return Prepared(keep, torch.stack(resid), cands, correct, tops)


@torch.no_grad()
def lens_ranks(model: LensModel, prep: Prepared, lens: Lens, layers: Sequence[int], *,
               calib: Calibration | None = None, mode: str = "raw",
               exclude: torch.Tensor | None = None, chunk: int = 256) -> np.ndarray:
    """Ranks ``[n_rows, n_layers]`` where rows enumerate (item, intermediate)."""
    rows = [(i, w) for i, c in enumerate(prep.cands) for w in c]
    out = np.zeros((len(rows), len(layers)), dtype=np.int64)
    for j, l in enumerate(layers):
        H = prep.resid[:, l]
        z_all = []
        for s in range(0, H.shape[0], chunk):
            z = lens.logits(model, H[s:s + chunk], l)
            if calib is not None and mode != "raw":
                z = calib.apply(z, l, mode)
            z_all.append(z)
        Z = torch.cat(z_all)
        for r, (i, w) in enumerate(rows):
            z = Z[i]
            ids = prep.cands[i][w]
            best = z[ids].max()
            if exclude is not None:
                mask = exclude.clone()
                mask[ids] = False
                gt = (z > best) & ~mask
            else:
                gt = z > best
            out[r, j] = int(gt.sum()) + 1
    return out


def row_index(prep: Prepared) -> list[tuple[int, str]]:
    return [(i, w) for i, c in enumerate(prep.cands) for w in c]


def scored_item_mask(prep: Prepared) -> np.ndarray:
    return np.array([c is None or c for c in prep.correct])


def recall_at_k(prep: Prepared, ranks: np.ndarray, layer_cols: Sequence[int], k: int, *,
                evals: Sequence[str] = MACRO_EVALS, n_boot: int = 2000, seed: int = 0,
                items_subset: set[int] | None = None, weighting: str = "item") -> dict:
    """Per-eval and macro recall@k with item-level bootstrap 95% CIs.

    weighting="item": per item, the fraction of its intermediates hit, averaged
    over items (Anthropic README). weighting="pair": hit rate over (item,
    intermediate) pairs (J++ pair_pass_at_k); bootstrap still resamples items."""
    rows = row_index(prep)
    best = ranks[:, list(layer_cols)].min(axis=1)
    hit = best <= k
    item_scores: dict[str, list[float]] = {}
    item_w: dict[str, list[float]] = {}
    keep = scored_item_mask(prep)
    per_item: dict[int, list[bool]] = {}
    for r, (i, _) in enumerate(rows):
        per_item.setdefault(i, []).append(bool(hit[r]))
    for i, hs in per_item.items():
        if not keep[i] or (items_subset is not None and i not in items_subset):
            continue
        ev = prep.items[i].eval
        item_scores.setdefault(ev, []).append(float(np.mean(hs)))
        item_w.setdefault(ev, []).append(float(len(hs)) if weighting == "pair" else 1.0)
    rng = np.random.default_rng(seed)
    res = {}
    boots = []
    for ev in evals:
        s = np.array(item_scores.get(ev, []))
        if len(s) == 0:
            continue
        w = np.array(item_w[ev])
        idx = rng.integers(0, len(s), size=(n_boot, len(s)))
        bs = (s[idx] * w[idx]).sum(1) / w[idx].sum(1)
        m = float((s * w).sum() / w.sum())
        res[ev] = {"mean": m, "lo": float(np.quantile(bs, 0.025)),
                   "hi": float(np.quantile(bs, 0.975)), "n": int(len(s))}
        boots.append(bs)
    if boots:
        macro_b = np.mean(np.stack(boots), axis=0)
        res["macro"] = {"mean": float(np.mean([res[e]["mean"] for e in evals if e in res])),
                        "lo": float(np.quantile(macro_b, 0.025)), "hi": float(np.quantile(macro_b, 0.975))}
    return res


def paired_bootstrap_diff(prep: Prepared, ranks_a: np.ndarray, ranks_b: np.ndarray, cols_a, cols_b, k: int,
                          evals: Sequence[str] = MACRO_EVALS, n_boot: int = 5000, seed: int = 0,
                          items_subset: set[int] | None = None) -> dict:
    """Macro recall@k difference (a - b), paired over items, with 95% CI and
    the bootstrap probability that a <= b."""
    rows = row_index(prep)
    ha = ranks_a[:, list(cols_a)].min(1) <= k
    hb = ranks_b[:, list(cols_b)].min(1) <= k
    keep = scored_item_mask(prep)
    per_ev: dict[str, list[tuple[float, float]]] = {}
    acc: dict[int, list[tuple[bool, bool]]] = {}
    for r, (i, _) in enumerate(rows):
        acc.setdefault(i, []).append((bool(ha[r]), bool(hb[r])))
    for i, v in acc.items():
        if keep[i] and (items_subset is None or i in items_subset):
            a = np.mean([x for x, _ in v])
            b = np.mean([y for _, y in v])
            per_ev.setdefault(prep.items[i].eval, []).append((a, b))
    rng = np.random.default_rng(seed)
    diffs = []
    point = []
    for ev in evals:
        if ev not in per_ev:
            continue
        arr = np.array(per_ev[ev])
        d = arr[:, 0] - arr[:, 1]
        point.append(d.mean())
        diffs.append(rng.choice(d, size=(n_boot, len(d)), replace=True).mean(1))
    D = np.mean(np.stack(diffs), axis=0)
    return {"diff": float(np.mean(point)), "lo": float(np.quantile(D, 0.025)),
            "hi": float(np.quantile(D, 0.975)), "p_le_0": float((D <= 0).mean())}
