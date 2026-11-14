# J-space on Qwen3-0.6B: the Jacobian lens and J++ at small scale

Reproduction and extension of Gurnee et al. (2026), *Verbalizable Representations Form a Global
Workspace in Language Models* ([paper](https://transformer-circuits.pub/2026/workspace/index.html),
[reference code](https://github.com/anthropics/jacobian-lens)), and of the J++ Lens (Ayonrinde &
Lindsey, 2026; [code](https://github.com/safety-research/jpp_lens)), on Qwen3-0.6B under a CPU-only
budget (4 cores, 15 GB RAM, no GPU).

- [`PROMPT.md`](PROMPT.md): the original request, what was underspecified about it, and the
  improved research spec this project executes.
- [`paper/paper.md`](paper/paper.md): the paper draft, assembled from the fragments in `paper/` by
  `python paper/build.py`.
- [`paper/prereg_round2.md`](paper/prereg_round2.md): the pre-registration (and amendment) for
  the round-2 test of improvements over J++, committed before any round-2 number was computed.

## Results so far

**Round 2 (pre-registered, `paper/prereg_round2.md`).** Five fresh stratified splits (seeds 1–5).
On each split every label-dependent weight is refitted on 47 fitting items, and every lens is
scored on 203 held-out items, under the J++ protocol (correctness filter, Readout Filtering, best
of 7 layers). `results/jpp_splits_round2.json`:

| Lens | labels? | mean held-out recall@10 | mean Δ vs J++ [range] | split wins |
|---|---|---|---|---|
| logit lens | no | 0.285 | −0.107 [−0.127, −0.083] | 0/5 |
| J-lens | no | 0.338 | −0.055 [−0.084, −0.036] | 0/5 |
| R-lens (LRP) | no | 0.386 | −0.006 [−0.023, +0.010] | 2/5 |
| Routed LRP experts, adaptive shrinkage (ours) | no | 0.393 | +0.000 [−0.015, +0.014] | 3/5 |
| **J++** | yes | 0.392 | — | — |
| JppB: J++ + broadcast lens (ours) | yes | 0.396 | +0.004 [−0.017, +0.029] | 3/5 |
| Routed J++ (ours) | yes | 0.380 | −0.013 [−0.034, −0.003] | 0/5 |
| **RB-mix: R-lens + broadcast lens, 2 weights/layer (ours)** | yes | 0.398 | **+0.006** [+0.000, +0.023] | **4/5** + 1 tie |
| **Bagged J++ (ours)** | yes | **0.403** | **+0.010** [−0.030, +0.058] | **4/5** |
| broadcast lens alone (ours) | no | 0.210 | −0.183 | 0/5 |

Win criterion fixed in advance: a positive mean difference and at least 4/5 split wins.

- **Two candidates meet the criterion.** Bagged J++ and RB-mix both do, by small margins: per-split
  paired 95% intervals are about ±3–6 points, and all splits share the same 250-item pool.
- **What both do.** Each reduces J++'s overfitting to its 47 labelled items.
- **Where J++'s gain comes from.** At this scale nearly all of J++'s gain over the J-lens comes
  from the LRP backward pass (R-lens).

Round 1 (one split; `results/jpp_compare_main.json`) and the all-items panel are in §5.2 of the
paper.

## Status

| Component | State |
|---|---|
| Exact J-lens estimator (reference reduction) | done; matches brute-force autograd to ~1e-7 |
| Broadcast-lens estimator (targets p′ > p only) | done; matches brute force to ~1e-7 |
| LRP backward (R-Lens / J++ "rlens" rules) | done |
| J++: router, expert Jacobians with empty-expert fallback, weight search (faithful port) | done |
| Readout evals (released sets, recall@k, correctness filter, bootstrap, Readout Filtering) | done |
| Reproduction readouts on Qwen3-0.6B (`results/readout_main.json`, `band.json`) | done |
| J++ head-to-head, round 1 (`results/jpp_compare_main*.json`) | done |
| J++ head-to-head, round 2 (multi-split, pre-registered; `results/jpp_splits_round2.json`) | done |
| Clamp-swap causal evals, ignition | running |
| Sketch estimators: learning curves vs compute | queued |

## Layout

```
jspace/            library
  model.py         HF wrapper, recording/editing hooks, fp32 unembedding
  fitting.py       exact (reference and broadcast) + sketch estimators of E[dh_final/dh_l]
  lrp.py           LRP backward rules (R-Lens / J++ "rlens" scope)
  routing.py       J++ router (unit-normalise, PCA-64, k-means E=8)
  jpp.py           J++ weight search (round-1 port and faithful v2), expert fallback, routed lens
  lens.py          logit / J / regression lenses, calibration, surface-form collapsing
  evals.py         readout evals (recall@k, bootstrap, paired differences)
  interventions.py clamp swap
  corpus.py        WikiText-103 fitting corpus, activation collection
scripts/
  test_estimators.py   correctness tests (no downloads)
  fit.py               fit a lens (exact / broadcast / LRP / sketch), optionally with J++ experts
  readout_eval.py      reproduction readouts; choose_band.py: pre-declared band rule
  jpp_compare.py       round 1 of the J++ head-to-head
  jpp_splits.py        round 2: pre-registered candidates over fresh splits
  swaps.py ignition.py curves.py figures.py
  run_queue*.sh after_*.sh   the compute queues actually run (logs in logs/)
data/              prompt sets from anthropics/jacobian-lens (Apache-2.0, see LICENSE-jacobian-lens)
results/           JSON results (committed); lenses/ holds fitted matrices (not committed)
```

## Reproduce

```bash
pip install torch "transformers>=5.5" datasets numpy scipy matplotlib
python scripts/test_estimators.py                       # correctness, ~1 min
python scripts/collect.py --n 256 --offset 5000         # held-out corpus residuals + regression lens
python -c "import torch; from jspace.routing import fit_router; \
  a = torch.load('lenses/corpus_acts.pt'); fit_router({int(l): v.float() for l, v in a['acts'].items()}).save('lenses/router_E8.pt')"
python scripts/collect_future.py                        # future-token targets for the label-free variants
python scripts/fit.py --method exact --n 32 --dim-batch 8 --router lenses/router_E8.pt \
    --snapshots 1 2 4 8 16 32 --tag exact               # J-lens + plain experts
python scripts/fit.py --method exact --n 32 --dim-batch 8 --router lenses/router_E8.pt --lrp \
    --snapshots 8 16 32 --tag lrp                       # R-lens + LRP experts (J++)
python scripts/fit.py --method exact --n 8 --dim-batch 8 --targets later --tag later   # broadcast lens
python scripts/readout_eval.py --lens J-lens=lenses/exact_final.pt --regress --filter --tag main
python scripts/choose_band.py results/readout_main.json "J-lens|raw"
python scripts/jpp_compare.py --plain exact --lrp lrp --extra B-lens=lenses/later_final.pt \
    --mix JB-mix=lenses/exact_final.pt,lenses/later_final.pt --tag main
python scripts/jpp_splits.py --seeds 0 1 2 3 4 5 --bags 8 --tag round2
```

See `scripts/run_queue*.sh` for the exact commands and order used. Measured cost on this machine
(bf16 with AMX): an exact prompt (1,024 VJPs over 128 tokens) takes about 2.5–3 min; a K = 16
sketch prompt takes about 2.5 s.
