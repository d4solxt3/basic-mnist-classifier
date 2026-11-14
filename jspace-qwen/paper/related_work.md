<!-- Draft fragment; merged into paper.md. Citations marked [verify] were seen only
     through secondary sources during drafting and must be checked against the
     primary source before submission. -->

## 2 Related work

**Lenses on the residual stream.** The *logit lens* (nostalgebraist, 2020) decodes an
intermediate residual with the model's own unembedding; it is uninformative in early and middle
layers, where the representation basis has not yet aligned with the output. The *tuned lens*
(Belrose et al., 2023) learns an affine translator per layer, trained to match the final-layer
distribution. Because it is trained on the model's *current* output, it tends to "skip ahead" to
the imminent prediction rather than expose intermediate variables. *Future Lens* (Pal et al.,
2023) and *Patchscopes* (Ghandeharioun et al., 2024) decode hidden states by learned or
prompted transplantation.

**The Jacobian lens and J-space.** Gurnee et al. (2026) replace the identity transport of the logit
lens with the average input–output Jacobian, J_l = E[∂h_final/∂h_l], estimated over
pretraining-like text, and call the span of the resulting per-token directions the *J-space*.
They argue that J-space content has functional properties of a global workspace (Baars, 1988;
Dehaene & Naccache, 2001): it is causally used downstream, confined to a mid-depth band,
flexibly routed, selective, and capacity-limited. A reference implementation and synthetic prompt
sets are public (anthropics/jacobian-lens). A commentary accompanying the paper reports a
reproduction of the core findings on Qwen 3.6 27B [verify].

**Improving the J-lens.** The *R-Lens* (Blank & Bhatia, 2026 [verify]) computes the Jacobian
through a Layer-wise Relevance Propagation backward graph. The *J++ Lens* (Ayonrinde & Lindsey,
2026) adds *Jacobian filtering*: positions are clustered (k-means, eight clusters per layer), one
expert Jacobian is averaged per cluster, and experts are combined with learned signed weights. J++
also uses the LRP backward pass and *Readout Filtering*, which drops tokens with no letter or
digit from the ranking. On Qwen3.6-27B it raises recall@10 of latent intermediates from 35.7%
(J-lens) to 55.2%. These methods change *what* is averaged. Our contributions are orthogonal: we
change *how cheaply* the average is estimated, and *how* lens logits are ranked.

**Randomized estimation.** Our sketch estimators are instances of classical randomized linear
algebra: Hutchinson-style probing (Hutchinson, 1989), random-projection sketches (Halko,
Martinsson & Tropp, 2011; Martinsson & Tropp, 2020), and importance sampling for approximate
matrix products (Drineas, Kannan & Mahoney, 2006). Control variates for Monte Carlo are textbook
(e.g. Owen, 2013). To our knowledge they have not been applied to estimating workspace lenses.
