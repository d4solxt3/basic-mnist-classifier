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

**J++ at this scale.** J++ is still the strongest lens (37.4%), but its gain over the J-lens is
about 4 points rather than about 20 at 27B. Most of that gain comes from the LRP backward pass.

**Label-free and broadcast variants.**
- A label-free lens, LRP experts routed with count-adaptive shrinkage, matches J++ to within a
  point.
- A new *broadcast lens* keeps only influence on strictly later positions, the attention-carried
  traffic a workspace reading identifies with availability. It is weak alone (21.7%) but
  complementary in mixes.

**Pre-registered round 2.** On five fresh splits, [ROUND 2 RESULT].

**Pitfalls.** We document two implementation pitfalls in J++-style evaluations: empty expert
clusters that score a spurious perfect recall, and masking before combining experts.

**Sketch estimators.** We derive unbiased sketch estimators of the J-lens with a weights-only
predictor of the benefit of readout-aware probe shaping. [SKETCH RESULT]

[SWAPS/IGNITION RESULT]
