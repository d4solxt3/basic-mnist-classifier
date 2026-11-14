<!-- Draft fragment; merged into paper.md. -->

### Label-free Jacobian filtering

J++'s largest single gain comes from *Jacobian filtering*. Source positions are clustered, one
expert Jacobian J_{l,e} is averaged per cluster, and the lens uses J_l = Σ_e w_{l,e} J_{l,e}. The
signed weights are chosen per layer to maximise recall@10 of known intermediates on a labelled
fitting split of the evaluation prompts. The weights thus depend on labelled data from the same
distribution the lens is scored on. That is a reasonable choice for a benchmark, but a barrier for a
tool meant to read *unknown* content.

We replace the labels with the definition of the J-lens itself. J_l measures how the residual at a
position influences the final-layer state summed over all *future* positions, i.e. what the model is
disposed to say later. On unlabelled text we know what was said later. For held-out corpus
positions p we take as candidates the set of semantic tokens (containing a letter or digit) among
t_{p+1}, …, t_{p+16}, and fit w_l with J++'s exact search and surrogate. We use 200 positions from
WikiText-103 records disjoint from all fitting and evaluation data. Everything else (experts,
router, number of starts and steps, Readout Filtering) is unchanged. We call the result *SS-J++*
(LRP experts) or *SS-J-filt* (plain experts).

### Routed expert lens

J++ collapses its experts into a single map per layer, keeping inference cost at one matrix–vector
product. The router that defines the clusters is itself cheap: a 64-dimensional projection and a
nearest-centroid lookup. The *routed* lens decodes each residual through the expert of its own
cluster, unembed(J_{l,c(h)} h). It needs no labelled data. Because each expert is averaged over
only about 1/E of the positions, we also consider shrinkage toward the pooled Jacobian,
J_{l,e}(λ) = λ J_{l,e} + (1 − λ) J̄_l, with λ ∈ {0.5, 1} selected on the fitting split.

### Ideas that did not work

We report three readout modifications that we expected to help. On the fitting split, two hurt
and one made no difference.

*Per-token calibration.* We ranked tokens by the z-score (or centred value) of their lens logit
relative to corpus statistics, to remove the context-independent prior. For the n = 8 J-lens,
recall@10 over all layers fell from 0.369 (raw) to 0.281 (centred) and 0.267 (z-scored). It fell
for the logit lens too (0.319 → 0.280). The prior that calibration removes evidently carries
signal: many latent intermediates are frequent, generic words.

*Affine Taylor transport.* The J-lens linearises around zero, J h = J(h − μ_l) + J μ_l. A
first-order expansion around the corpus mean instead gives μ_L + J(h − μ_l). On the fitting split
this lowered recall@10 from 0.343 to 0.280 (correctness-filtered items) and from 0.247 to 0.163
(all items). Dropping the offset entirely ("centred") was also worse. ‖μ_L‖ = 459, against
‖J μ_l‖ = 119 at layer 14, with cosine 0.83. Replacing the transported mean by the true
final-layer mean over-weights the generic output direction.

*Surface-form collapsing.* Top-10 lists often spend several slots on one word ("December",
" December", " december"). The evaluation credits a candidate through its best surface form but
makes it compete against every variant of every competitor. We therefore collapsed each class of
tokens that agree after dropping one leading space and lowercasing (37,912 tokens in multi-member
classes) into one canonical entry carrying the class maximum. The canonical member is fixed by the
vocabulary as the first of " "+lower, lower, " "+Capitalised, Capitalised that exists, which is
always one of a candidate's credited forms. We checked this for all 3,930 candidate ids, with no
exception. Fitting-split recall@10 was unchanged (J-lens 0.364 → 0.364; logit lens 0.311 →
0.311). The duplicated variants are almost always variants of the *correct* concept, which
already outranks the candidate's competitors.
