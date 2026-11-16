<!-- Draft fragment; merged into paper.md. Search-noise note: sprint/results/search_noise.json (sprint/fidelity/search_noise.py), sprint/RUNLOG.md rows E0.1-fidelity-search and E0.1-search-noise. -->

## 6 Limitations

**Scale.** All results are for one 0.6B model. Small models answer few of the released evaluation
items correctly (8–15 items each on multihop, multilingual and poetry under the correctness
filter), so per-eval estimates on those sets are noisy. We mitigate this with a pre-declared
all-items analysis and item-level bootstrap intervals, but we cannot separate "the lens fails"
from "the model never computed the intermediate" on items the model gets wrong.

**Fitting budget.** Our lenses average 32 prompts (32 × 1,024 VJPs, about 1.5 CPU-hours each),
against 1,000 for the released Qwen lenses and J++. J++'s expert Jacobians each see about 1/8 of
the positions, so they are the most budget-sensitive component. J++ may therefore be
disadvantaged relative to its published configuration, and conclusions about J++ at full budget
should not be drawn from our numbers.

**Protocol fidelity.** We could not access the J-space paper's text from our compute environment.
Our implementation follows the authors' released code (`anthropics/jacobian-lens`) and the J++
code (`safety-research/jpp_lens`) line by line where possible: the estimator, the correctness rule,
surface forms, the clamp swap and the expert-weight search. We verified the estimator against a
brute-force Jacobian. Choices not pinned down by released code are stated explicitly: the
seven-layer subset for a 28-layer model, the workspace-band rule, the ignition summary statistic,
and including the final-norm gain in J-lens directions.

**Splits, seeds and the item pool.** Rounds 2 and 3 vary the fitting/held-out split (26 seeds in
all). The router's k-means, the fitting prompts and the probe draws use one seed each, so bootstrap
intervals capture item and split sampling but not variability over fitting prompts or router
initialisations. Part of the split-to-split variation is search noise: J++'s weight search (200 Adam
steps on a sigmoid-margin surrogate, keeping the best exact iterate) is chaotic at the float level,
and in a later check the autograd reference implementation (`safety-research/jpp_lens`) and an
algorithmically identical reimplementation with an analytic gradient started from the same points,
with the same surrogate and gradient at step 0, yet agreed on only 8 of 14 (split, layer) instances
(split seeds 0 and 1). Where they differed, J++'s objective on held-out items (recall@10 at a single
layer) moved by up to 2.1 points, with mean difference −0.05 and SD 0.66 points over all 14
instances, and even on the fitting items, where the search maximises that objective, the two ended
at 0.271 and 0.285 (split seed 0, layer 4).
All splits draw on the same 250 correctness-filtered items. Round 3 shows how easily
five overlapping splits mislead: two candidates that met round 2's criterion tie J++ over twenty
more. The out-of-pool items (289 that the model answers wrongly) are a different population, not a
replication on new items.

**Sketch comparison.** Each sketch variant ran once on the real model, with one probe seed and one
record order. The dictionary errors are measured against an exact lens on only 16 records, whose
noise is shared by every estimator. The theory checks use only records 0–31, the records for which
per-prompt Jacobians were stored. We derived the shrunk control variate after the planned one
diverged. Its real-model run is a single exploratory test, not a pre-registered one.

**What "J-space" means here.** We test operational consequences (readout recall, causal swap
success, switch sharpness) and make no claims about consciousness or access consciousness,
which, as the original authors stress, these experiments cannot settle.
