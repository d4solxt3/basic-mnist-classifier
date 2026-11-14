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

**J++ at this scale.** J++'s gain over the J-lens averages 5.5 points over five fresh splits
(range 3.6–8.4), against about 20 at 27B. Nearly all of it comes from the LRP backward pass: J++
beats the plain LRP R-lens by only 0.6 points on average. Without the correctness filter, the
R-lens is ahead of J++.

**Label-free and broadcast variants.**
- A label-free lens, LRP experts routed with count-adaptive shrinkage, ties J++: the mean
  difference over five fresh splits is +0.000.
- A new *broadcast lens* keeps only influence on strictly later positions, the attention-carried
  traffic a workspace reading identifies with availability. It is weak alone (21.7%) but
  complementary in mixes.

**Pre-registered round 2.** Four candidate improvements were fixed in advance and tested on five
fresh splits. Two meet the win criterion (positive mean paired difference and at least 4/5 split
wins), by small margins:
- Bagged J++ weights: +1.0 points recall@10, 4/5 wins.
- A two-weight mix of the R-lens and the broadcast lens: +0.6 points, 4 wins and 1 tie.

Both reduce J++'s overfitting to its 47 labelled fitting items. Neither margin is large relative
to per-split noise.

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
