"""Screen transport variants on the J++ FITTING split only (model selection).

python scripts/screen_variants.py lenses/exact_n8.pt
"""
import os, sys, torch
ROOT = __file__.rsplit("/scripts/", 1)[0]; sys.path.insert(0, ROOT)
from jspace import evals as E, paths as P  # noqa
from jspace.jpp import split_items  # noqa
from jspace.lens import AffineLens, Lens  # noqa
from jspace.model import load_model  # noqa
torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", 2)))
m = load_model("Qwen/Qwen3-0.6B")
layers = [4, 7, 11, 14, 18, 21, 24]
cols = list(range(len(layers)))
prep = torch.load(os.path.join(P.LENSES, "readout_prep.pt"), weights_only=False)
fit_f, _ = split_items(prep.items, E.scored_item_mask(prep), seed=0)
prep_all = torch.load(os.path.join(P.LENSES, "readout_prep.pt"), weights_only=False); prep_all.correct = [None] * len(prep_all.correct)
fit_a, _ = split_items(prep_all.items, E.scored_item_mask(prep_all), seed=0)
s = torch.load(os.path.join(P.LENSES, "corpus_acts.pt"), weights_only=True)
mu_in = {int(l): v.float().mean(0) for l, v in s["acts"].items()}
mu_out = s["target"].float().mean(0)
base = Lens.load(P.resolve(sys.argv[1]))
base.maps = {l: base.maps[l] for l in layers}
excl = E.non_semantic_ids(m.tokenizer, m.vocab_size)
variants = {"plain": AffineLens(base, mu_in, mu_out, "plain"), "centered": AffineLens(base, mu_in, mu_out, "centered"),
            "affine": AffineLens(base, mu_in, mu_out, "affine")}
for al in (0.5, 1.5, 2.0):
    variants[f"plain-a{al}"] = AffineLens(base, mu_in, mu_out, "plain", alpha=al)
    variants[f"affine-a{al}"] = AffineLens(base, mu_in, mu_out, "affine", alpha=al)
print(f"||mu_out|| = {mu_out.norm():.1f}; ||J mu_l|| at layer 14 = {(base.maps[14] @ mu_in[14]).norm():.1f}; cos = "
      f"{float(torch.nn.functional.cosine_similarity(base.maps[14] @ mu_in[14], mu_out, dim=0)):.3f}")
for name, lens in variants.items():
    for filt in (False, True):
        r = E.lens_ranks(m, prep, lens, layers, exclude=excl if filt else None)
        a_ = E.recall_at_k(prep, r, cols, 10, items_subset=set(fit_f), n_boot=10)["macro"]["mean"]
        b_ = E.recall_at_k(prep_all, r, cols, 10, items_subset=set(fit_a), n_boot=10)["macro"]["mean"]
        print(f"{name:<14} filter={filt!s:<5}  FIT-split recall@10: filtered-items={a_:.3f}  all-items={b_:.3f}", flush=True)
