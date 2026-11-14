# Workspace Lenses at Small Scale: Reproducing the Jacobian Lens and J++ on Qwen3-0.6B, with Label-Free and Broadcast Variants

*Draft. Author list to be added.*

## Abstract

The Jacobian lens (J-lens) reads an intermediate residual vector of a language model through
the corpus-averaged Jacobian to the final layer. The tokens it surfaces in a mid-depth band
have been argued to behave like the contents of a global workspace. J++ makes these readouts
markedly more faithful in large models. We reproduce the J-lens, the R-Lens and J++ on
Qwen3-0.6B on a single CPU, with the released prompt sets and the J++ evaluation protocol.

**Reproduction.** On held-out items the J-lens recovers 33.2% of latent intermediates at
recall@10, against 27.1% for the logit lens. Its advantage sits in layers 4–19 and on typo
correction. The workspace band chosen by a pre-declared rule is layers 20–24.

**J++ at this scale.** Over 25 fresh splits, J++'s gain over the J-lens averages 6.7 points,
against about 20 at 27B. About two-thirds of it comes from the LRP backward pass (4.5 points) and a
third from the learned expert weights (2.2). Without the correctness filter, the plain LRP R-lens is
ahead of J++.

**Label-free and broadcast variants.**
- A label-free lens routes each residual to its own LRP expert, with count-adaptive shrinkage. It
  comes within 1.6 points of J++ over 25 splits.
- A new *broadcast lens* keeps only influence on strictly later positions, the attention-carried
  traffic a workspace reading identifies with availability. It is weak alone (21.7%), and LRP
  rules raise it by 6 points.

**Pre-registered tests: J++ is not beaten.** Two pre-registered rounds tested improvements over
J++.
- **Round 2** fixed four candidates and ran five fresh splits. Two met the win criterion by small
  margins: bagged J++ weights (+1.0 points, 4/5 wins) and a two-weight mix of the R-lens and the
  broadcast lens (+0.6, 4/5 wins and 1 tie).
- **Round 3** re-tested both, plus an LRP-broadcast mix, on twenty more splits, with 16/20 wins
  required. All three tie J++ (+0.3, −0.4 and −0.4 points; 11, 7 and 9 wins).

Bagging halves J++'s overfitting to its 47 labelled items, but does not raise held-out recall.

**Pitfalls.** We document two implementation pitfalls in J++-style evaluations: empty expert
clusters that score a spurious perfect recall, and masking before combining experts.

**Sketch estimators.** We derive unbiased sketch estimators of the J-lens with a weights-only
predictor of the benefit of readout-aware probe shaping. [SKETCH RESULT]

**Causal swaps.** We swap a templated function's argument (e.g. the country in "The capital of
A is") along lens directions:

| Directions | Success | Band |
|---|---|---|
| J-lens | 28% | readout-selected band (layers 20–24) |
| J-lens | 48% | mid-depth layers 7–20 |
| logit lens | 7–8% | either band |
| random-target null | ≤2% | — |
| J-lens | 43–48% | any 5-layer window in layers 3–19 (exploratory) |

The leverage is concentrated in one category (countries). Equal-width windows show that depth,
not the number of edited layers, sets it: every window from layer 3 to 19 beats the
readout-selected band.

**Ignition.** As one input word is morphed into another, the network sharpens the switch between
competing concepts beyond what a straight-line mix of the endpoint activations predicts. For country
pairs the switch is 0.12–0.16 narrower through layers 12–24, against about 0.05 for unrelated nouns.
The sharpening grows steadily to the output and appears equally in logit-lens readouts. It is not
confined to a J-space band.
