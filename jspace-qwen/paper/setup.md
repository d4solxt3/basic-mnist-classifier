<!-- Draft fragment; merged into paper.md. -->

## 4 Experimental setup

**Model and hardware.** Qwen3-0.6B (post-trained checkpoint `Qwen/Qwen3-0.6B`; 28 decoder blocks,
d_model = 1024, vocabulary 151,936, tied embeddings), in bf16, on one 4-core Intel Xeon CPU with
AMX, 15 GB RAM, no GPU. Chat prompts use the model's template with thinking disabled. Layer l means
the output of block l; the target layer is block 27, before the final RMSNorm.

**Fitting corpus.** WikiText-103 (raw) training records of at least 600 characters, truncated to
128 tokens, as in the reference implementation. Fitting uses records 0–31 (exact) or 0–511
(sketches). Router fitting, calibration statistics and the regression lens use 12,288 positions
(48 random positions ≥ 16 from each of records 5000–5255), disjoint from the fitting prompts.

**Lenses.**
*Logit lens*: identity transport.
*Regression lens*: ridge least squares from h_l to h_27 on corpus positions, with the ridge chosen
per layer on held-out R² (0.21 at layer 0, rising to 0.75 at layer 24). It is a tuned-lens-style
baseline trained on the model's *current* output.
*J-lens*: exact reference estimator over 32 prompts (32 × 1,024 VJPs).
*R-lens*: the same, with the backward graph under the LRP rules of J++ ("rlens" scope: LN-rule on
residual RMSNorms, identity rule on SiLU, half rule on the gate×up product).
*J-filt / J++*: per-layer router (unit-normalise, PCA to 64 dimensions, k-means with E = 8), expert
Jacobians as position-weighted per-cluster means from the same backward passes, and signed
unit-norm expert weights fitted per layer by J++'s smooth-surrogate objective on the fitting split.
J-filt uses plain gradients and J++ uses LRP gradients. As in the reference implementation, an
expert that saw fewer than 50 positions is replaced by the pooled Jacobian. With 32 fitting
prompts this happens for one cluster at each of layers 20–26. Without the fallback, the empty
clusters give all-zero maps whose logits all tie, and under the strict rank rule an all-zero map
scores a spurious recall of 1.0. We hit this bug in a first run and now flag any lens whose
fitting-split recall is 1.0.
*Routed* (ours): each residual is decoded through the expert of its own cluster.
*Weight-search fidelity.* Our round-1 port of J++'s weight search deviated from the reference
search (`workspace_lens.fitting.condense_experts`) in three small ways. It masked excluded tokens
to −10⁴ in each expert before combining, rather than to −∞ after. It took the surrogate's spread
over non-excluded tokens rather than the full vocabulary. It broke ties between iterates by order
rather than by surrogate value. Rounds 2 and 3 (§5.3–5.4) use a faithful port for J++ and for every
candidate, and round 2's seed-0 run gives faithful-port numbers on the round-1 split.
*B-lens* (ours): the broadcast lens of §3.1, fitted on the same first 8 corpus prompts as the
J-lens's n = 8 snapshot (the J-lens's recall is flat from n = 8 to n = 32; §5.1).

**Readout evaluation.** The six released lens-eval prompt sets (multihop, multilingual, poetry,
association, typo, order-ops). Readout positions, single-token surface forms (with/without a leading
space; original, capitalised and lowercase casing; order-ops synonyms) and min-over-surfaces ranks
follow the released READMEs and the J++ code. Items count only if the model's greedy next token
matches the target (multihop, multilingual, order-ops) or the rhyme word (poetry); association and
typo items are always kept. For Qwen3-0.6B this leaves 14/84 multihop, 15/107 multilingual, 8/98
poetry, 99/99 association and 96/96 typo items (and 18/55 order-ops, reported separately). We
therefore also report a pre-declared secondary analysis *without* the correctness filter.
*Recall@k* is computed per item (the fraction of an item's intermediates whose best rank over
layers is ≤ k), macro-averaged over the five macro evals. We also give J++'s pair-level weighting.
Following J++, the head-to-head comparison uses the best rank over seven evenly spaced layers
(4, 7, 11, 14, 18, 21, 24), a stratified 20% fitting / 80% held-out split (seed 0), and scores on
held-out items only. Intervals are 95% item-level bootstrap intervals. Differences between lenses
use a paired bootstrap over items.

**Model selection.** Every readout choice (raw, centred, or z-scored logits, with or without
Readout Filtering) is selected on the *fitting* split, the same labelled items J++ uses to fit its
weights. The held-out split is used only for reporting.
