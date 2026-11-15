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

## At a glance: which large-model claims carry over to 0.6B

| Claim (large models) | Qwen3-0.6B (this work) | Verdict |
|---|---|---|
| The J-lens reads latent intermediates better than the logit lens | 36.7% vs 31.9% recall@10; advantage in layers 4–19 and on typos | holds |
| Readable content occupies a contiguous band of layers | readouts peak in layers 20–24, chosen by a pre-declared rule | holds |
| J-lens directions are used causally | argument swaps 28–48% vs 7–8% for logit-lens directions, ≤2% for an unrelated target | holds, narrowly |
| Causal content is confined to that band | every 5-layer window from layer 3 to 19 beats the readout band (43–48% vs 28%; exploratory) | not at 0.6B |
| Ignition-like switching | the network sharpens competing concepts, but progressively with depth and as much through the logit lens | partly |
| Implicit concepts are readable | ≤1% recall on association and rhyme plans | not at 0.6B |
| J++ makes readouts far more faithful | +6.7 points over the J-lens (25 splits), about two-thirds from the LRP backward pass | much smaller |
| J++'s learned expert weights add substantially | +2.2 points over the plain LRP lens; they overfit, and no pre-registered change beats them | modestly |

Figures are in `paper/figures/` (`scripts/figures.py`); the method schematic is
`paper/arxiv/figures/schematic.tex`.

## Results so far

**J++ is not beaten.** Two pre-registered rounds tested improvements on fresh stratified splits.
On each split every label-dependent weight is refitted on 47 fitting items, and every lens is scored
on 203 held-out items, under the J++ protocol.

**Round 3** (`paper/prereg_round3.md`; 20 new splits; win = positive mean difference and ≥16/20
wins; `results/jpp_splits_round3.json`):

| Lens | labels? | mean held-out recall@10 | mean Δ vs J++ [range] | wins/ties/losses |
|---|---|---|---|---|
| Bagged J++ (round-2 winner) | yes | 0.407 | +0.003 [−0.029, +0.054] | 11/1/8 |
| **J++** | yes | 0.404 | — | — |
| RB-mix (round-2 winner) | yes | 0.401 | −0.004 [−0.063, +0.035] | 7/0/13 |
| RLB-mix: R-lens + LRP broadcast lens (new) | yes | 0.400 | −0.004 [−0.072, +0.044] | 9/0/11 |
| routed LRP experts (no labels) | no | 0.385 | −0.019 [−0.080, +0.033] | 3/0/17 |
| R-lens (LRP) | no | 0.378 | −0.026 [−0.088, +0.027] | 3/0/17 |
| J-lens | no | 0.334 | −0.070 [−0.127, −0.036] | 0/0/20 |

- **No candidate meets the criterion.** The two winners of round 2 (`paper/prereg_round2.md`;
  5 splits; Bagged J++ +1.0 points, RB-mix +0.6) were within noise.
- **Where J++'s gain over the J-lens comes from** (25 splits pooled):
  - 4.5 points from the LRP backward pass;
  - 2.2 points from the expert weights.
- **What bagging does.** It halves J++'s fit/held-out gap (9.8 → 4.0 points) without raising
  held-out recall.
- **Out of pool.** On the 289 items the model answers wrongly, the plain R-lens beats J++ (+0.9
  points, 15/20).

Round 1 (one split) and the all-items panel are in §5.2 of the paper.

**Sketch estimators** (§3.4 and §5.7 of the paper; `results/curves.json`, `results/sketch_theory.json`):

- **The planned online control variate diverges.** At 32 prompts its dictionary error is
  2,600–3,700 times the lens's norm, and recall@10 is 0 at every budget. A variance recursion
  (Proposition 3) predicts growth while n < d/2K = 31, and that is where the error peaks. Replaying
  the estimator from stored per-prompt Jacobians reproduces the saved snapshots to within 3%, so this
  is not a code bug.
- **Shrinking the control variate toward the identity fixes it** (post hoc, one run). At 4,096 VJP
  rows it is 6–22% more accurate than the identity control variate.
- **Probe shaping works as predicted.** Sketch-noise variance falls by 0.74–0.81, against the
  weights-only prediction ρ(A) = 0.764. Recall@10 did not improve (29.0% vs 32.0%).
- **No sketch beats the exact estimator at equal cost under independent prompts** (proved, and
  confirmed by Monte Carlo). On WikiText the sketches stay competitive because consecutive records are
  correlated.

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
| Clamp-swap causal evals (`results/swaps_*.json`) | done |
| Ignition with linear-interpolation control (`results/ignition_main*.json`) | done |
| J++ head-to-head, round 3 (pre-registered, 20 splits; `results/jpp_splits_round3.json`) | done |
| Sketch estimators: learning curves, theory checks, shrunk control variate (`results/curves.json`, `sketch_theory.json`) | done |

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

About 11 CPU-hours end to end on 4 cores / 15 GB RAM, no GPU. Run one job at a time.

```bash
pip install torch "transformers>=5.5" datasets numpy scipy matplotlib
python scripts/test_estimators.py                       # correctness, ~1 min
# corpus residuals + regression lens, J++ router, future-token targets
python scripts/collect.py --n 256 --offset 5000
python -c "import torch; from jspace.routing import fit_router; \
  a = torch.load('lenses/corpus_acts.pt'); fit_router({int(l): v.float() for l, v in a['acts'].items()}).save('lenses/router_E8.pt')"
python scripts/collect_future.py
# lenses: J-lens + plain experts (~80 min), R-lens + LRP experts (~80 min), broadcast lens (~21 min)
python scripts/fit.py --method exact --n 32 --dim-batch 8 --router lenses/router_E8.pt --snapshots 1 2 4 8 16 32 --tag exact
python scripts/fit.py --method exact --n 32 --dim-batch 8 --router lenses/router_E8.pt --lrp --snapshots 8 16 32 --tag lrp
python scripts/fit.py --method exact --n 8 --dim-batch 8 --targets later --tag later
# reproduction readouts and the pre-declared band rule
python scripts/readout_eval.py --lens J-lens=lenses/exact_final.pt --regress --filter --tag main
python scripts/choose_band.py results/readout_main.json "J-lens|raw"
# J++ round 1 (one split; then the all-items panel)
EXTRA="--extra J-lens-n8=lenses/exact_n8.pt B-lens=lenses/later_final.pt --mix JB-mix=lenses/exact_final.pt,lenses/later_final.pt"
python scripts/jpp_compare.py --plain exact --lrp lrp $EXTRA --tag main
python scripts/jpp_compare.py --plain exact --lrp lrp $EXTRA --all-items --tag main_all
# J++ round 2 (pre-registered, paper/prereg_round2.md)
python scripts/jpp_splits.py --seeds 0 1 2 3 4 5 --bags 8 --tag round2
# causal swaps and ignition
python scripts/swaps.py --jlens lenses/exact_final.pt --regress --band 20 24 --tag main
python scripts/swaps.py --jlens lenses/exact_final.pt --band 7 20 --scales 1 2 4 --tag midband
scripts/run_windows.sh                                  # exploratory: equal-width 5-layer swap windows
python scripts/swap_stats.py --tags main midband win3_7 win7_11 win11_15 win15_19
python scripts/ignition.py --jlens lenses/exact_final.pt --carriers 10 --controls --tag main
python scripts/ignition.py --jlens lenses/exact_final.pt --carriers 10 --controls --linear-control --tag main
python scripts/ignition_stats.py --tag main
# J++ round 3 (pre-registered, paper/prereg_round3.md)
python scripts/fit.py --method exact --n 8 --dim-batch 8 --targets later --lrp --tag lrp_later
python scripts/jpp_splits.py --seeds $(seq 6 25) --bags 8 --candidates Bagged-J++ RB-mix RLB-mix \
    --lrp-b lenses/lrp_later_final.pt --oop --tag round3
# sketch estimators and learning curves
python scripts/fit.py --method sketch --K 16 --cv online --n 512 --snapshots 16 32 64 128 256 512 --tag sk16 --seed 1
python scripts/fit.py --method sketch --K 16 --cv online --shaped --n 512 --snapshots 16 32 64 128 256 512 --tag sk16w --seed 1
python scripts/fit.py --method sketch --K 16 --cv identity --n 256 --snapshots 16 32 64 128 256 --tag sk16id --seed 1
python scripts/fit.py --method sketch --K 16 --cv none --n 256 --snapshots 16 32 64 128 256 --tag sk16none --seed 1
python scripts/fit.py --method sketch --K 16 --cv identity --shaped --n 256 --snapshots 16 32 64 128 256 --tag sk16idw --seed 1
python scripts/fit.py --method sketch --K 16 --cv shrink --n 256 --snapshots 16 32 64 128 256 --tag sk16sh --seed 1
python scripts/curves.py --tags exact sk16 sk16w sk16id sk16none sk16idw sk16sh
python scripts/sketch_theory.py --seeds 8 --steps 512 --threads 4   # replay, exact noise, Monte Carlo
python scripts/rho.py                                   # weights-only predictor of the shaping gain
# figures and the assembled paper draft
python scripts/figures.py && python paper/build.py
```

See `scripts/run_queue*.sh` for the exact commands and order used. Measured cost on this machine
(bf16 with AMX): an exact prompt (1,024 VJPs over 128 tokens) takes about 2.5–3 min; a K = 16
sketch prompt takes about 2.5 s.
