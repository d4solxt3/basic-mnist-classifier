<!-- Draft fragment; merged into paper.md. -->

### 3.2 Label-free Jacobian filtering

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

### 3.3 Routed expert lens

J++ collapses its experts into a single map per layer, keeping inference cost at one matrix–vector
product. The router that defines the clusters is itself cheap: a 64-dimensional projection and a
nearest-centroid lookup. The *routed* lens decodes each residual through the expert of its own
cluster, unembed(J_{l,c(h)} h). It needs no labelled data. Because each expert is averaged over
only about 1/E of the positions, we also consider shrinkage toward the pooled Jacobian,
J_{l,e}(λ) = λ J_{l,e} + (1 − λ) J̄_l. We consider λ ∈ {0.5, 1} and a count-adaptive
λ_e = n_e/(n_e + κ) with κ = 200, where n_e is the number of fitting positions expert e saw. The
variant is selected on the fitting split.
