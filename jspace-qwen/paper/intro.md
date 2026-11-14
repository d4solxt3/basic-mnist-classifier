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
   causal swap, and item-level bootstrap intervals throughout.
2. **Unbiased sketch estimators** of the average Jacobian (Haar-random probes, an exactly known
   identity path and an online control variate, and *readout-aware* probe shaping). We prove they
   are unbiased, derive their variance, and give a closed-form, weights-only predictor ρ(A) of the
   gain from shaping.
3. **Unsupervised readout improvements**, evaluated head-to-head against J++ under its own
   protocol: per-token *calibrated* readouts, and a *routed* expert lens that uses J++'s
   clusters at inference without any labelled data.

We report every comparison, including those our methods lose.
