<!-- Draft fragment; merged into paper.md. Numbers are filled from results/ only. -->

## 1 Introduction

Gurnee et al. (2026) introduced the *Jacobian lens* (J-lens): to read what an intermediate
residual-stream vector h_l is disposed to make a language model say, transport it to the final
layer with the corpus-averaged Jacobian J_l = E[∂h_final/∂h_l], then decode it with the model's own
unembedding. The tokens the J-lens surfaces in a contiguous mid-depth band, the *J-space*, behave
like the contents of a global workspace. They are causally used (swapping one concept's lens
coordinate for another's redirects multi-hop answers), selectively and flexibly routed, and show
ignition-like dynamics. These claims were established on Claude models and reproduced on Qwen
3.6 27B. The J++ Lens (Ayonrinde & Lindsey, 2026) then made the readouts markedly more faithful,
raising recall@10 of latent intermediates on Qwen3.6-27B from 35.7% to 55.2%.

Two practical questions remain open. (1) **Does any of this hold in a small model?** If the J-space
is a product of scale, its properties should weaken or vanish at sub-billion scale. If it is a
generic consequence of residual-stream computation, they should persist. (2) **Can the lens be
fitted and read more efficiently?** The reference estimator needs d_model backward passes per
fitting prompt, and J++ additionally needs labelled evaluation items to weight its expert
Jacobians. Both costs are barriers to using workspace lenses as routine tools.

We study both questions on Qwen3-0.6B (28 layers, d_model = 1024) under a deliberately austere
budget: one 4-core CPU, no GPU. Our contributions:

1. **A small-model reproduction with controls.** We reproduce the J-lens, the R-Lens and the J++
   Lens on Qwen3-0.6B with the authors' released prompt sets and scoring conventions. We add a
   ridge-tuned regression-lens baseline, matched-norm random and logit-lens controls for every
   causal swap, and item-level bootstrap intervals throughout. The J-lens's advantage over the
   logit lens is real but concentrated in mid-depth layers and on one evaluation (typos).
2. **J++ at small scale, and two pitfalls.** J++'s gain over the J-lens shrinks from about 20
   points at 27B to about 5.5 (mean over five splits). Nearly all of that comes from the LRP
   backward pass: J++ beats the plain LRP R-lens by only 0.6 points on average. We document two implementation
   pitfalls that silently distort J++-style evaluations. With a small fitting corpus, empty expert
   clusters give all-zero maps, which score a spurious perfect recall under the strict rank
   rule. Masking tokens before combining experts penalises negative weight sums.
3. **Label-free alternatives that match J++.** Routing each residual through its own LRP expert,
   with count-adaptive shrinkage toward the pooled map, needs no labelled data and ties J++'s
   held-out recall over five fresh splits (mean difference +0.000).
4. **The broadcast lens.** We define a lens that keeps only influence on strictly *later*
   positions, the attention-carried traffic that a global-workspace reading identifies with
   availability. It is weak on its own but ranks some intermediates no other lens finds. Mixed with the R-lens
   (two weights per layer), it gains mainly on multihop bridge entities.
5. **A pre-registered, multi-split test of improvements over J++.** Because the released
   evaluations are small, we pre-register four candidate improvements and test them on five
   fresh fitting/held-out splits with a stated win criterion. Two meet it, by small margins:
   bagging J++'s weights (+1.0 points, 4/5 wins) and the R-lens/broadcast mix (+0.6, 4/5 wins
   and one tie). Both work by giving the 47 labelled items less room to overfit.
6. **Unbiased sketch estimators** of the average Jacobian: Haar-random probes, an exactly known
   identity path, an online control variate, and *readout-aware* probe shaping. We prove they
   are unbiased, derive their variance, and give a closed-form, weights-only predictor ρ(A) of the
   gain from shaping.

We report every comparison, including those our methods lose.
