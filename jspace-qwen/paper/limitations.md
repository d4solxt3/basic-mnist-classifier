<!-- Draft fragment; merged into paper.md. -->

## Limitations

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

**Single seed and split.** The fitting/held-out split, the router's k-means and the probe draws
use one seed each. Bootstrap intervals capture item sampling, but not variability over fitting
prompts or router initialisations.

**What "J-space" means here.** We test operational consequences (readout recall, causal swap
success, switch sharpness) and make no claims about consciousness or access consciousness,
which, as the original authors stress, these experiments cannot settle.
