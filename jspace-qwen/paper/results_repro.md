<!-- Draft fragment; merged into paper.md. Numbers from results/readout_main.json, readout_peek_n8.json, band.json. -->

## 5 Results

### 5.1 The J-lens on a 0.6B model

**Readout recall.** Table 1 reports recall@10 of latent intermediates, using the best rank over all
27 block outputs and correctness-filtered items. The J-lens recovers 36.7% of intermediates
(macro; 95% CI 31.4–41.8%), against 31.9% for the logit lens and 16.0% for the ridge regression
lens. With Readout Filtering the figures are 39.2% for the J-lens and 33.5% for the logit lens. The
advantage is concentrated in one eval: on typo items the J-lens recovers the corrected word 77%
of the time, against 47% for the logit lens. On multihop and multilingual items the two lenses
are statistically indistinguishable at this sample size (14 and 15 items).

**Table 1.** Recall@10 (best of layers 0–26, correctness-filtered items, 95% item bootstrap CI on
the macro).

| Lens | multihop (14) | multilingual (15) | typo (96) | association (99) | poetry (8) | macro |
|---|---|---|---|---|---|---|
| logit lens | 0.750 | 0.378 | 0.469 | 0.000 | 0.000 | 0.319 [0.265, 0.365] |
| regression lens | 0.250 | 0.311 | 0.240 | 0.000 | 0.000 | 0.160 [0.114, 0.212] |
| J-lens (n = 32) | 0.679 | 0.383 | 0.771 | 0.000 | 0.000 | 0.367 [0.314, 0.418] |
| logit lens + RF | 0.750 | 0.417 | 0.500 | 0.010 | 0.000 | 0.335 [0.280, 0.383] |
| J-lens + RF | 0.750 | 0.400 | 0.802 | 0.010 | 0.000 | 0.392 [0.340, 0.441] |

RF = Readout Filtering (non-alphanumeric tokens removed from the ranking).

**Where the two lenses differ.** The all-layer maximum hides the mid-depth picture (Figure 1). In
layers 4–19 the logit lens is near floor (single-layer recall@10 between 0.004 and 0.036). The
J-lens recovers 5–11% there, three to eight times more, and peaks at layer 24 (26.2%, against 19.9%
for the logit lens). In the last layers both lenses read out the imminent answer, which is why
their best-of-layers scores converge on the multihop and multilingual sets, where the
intermediate is often close to the output.

**The workspace band.** Our pre-declared rule takes the contiguous run of layers around the peak
of the J-lens curve on which recall stays above half the peak, excluding the three
output-adjacent layers. It selects layers 20–24, about 71–86% of depth. Restricted to this band,
recall@10 is 23.4% for the J-lens against 18.7% for the logit lens (25.5% vs 20.3% with Readout
Filtering).

**Association and poetry are beyond this model.** No lens recovers more than 1% of association
intermediates, and none recovers a poetry rhyme plan. This is not a scoring artefact. The median
best rank of the correct association concept is about 1,000–2,000 across lenses (10th percentile about 200), and the median
poetry rhyme rank is around 300–440. A 0.6B model evidently does not represent these implicit
concepts at the readout position in a form any linear readout of the unembedding recovers. All
lens differences below therefore come from the multihop, multilingual and typo sets.

**Fitting budget.** The J-lens saturates early. Fitted on 8 prompts (8,192 VJPs, about 25 minutes
on our CPU), it scores 36.9% recall@10 (39.9% with Readout Filtering), the same as with 32 prompts.
At this model size the corpus-averaged Jacobian is well determined by a few thousand source
positions.
