## 3 Methods

All lenses map a residual h_l at block l to logits unembed(M_l h_l) for a d×d map M_l. The logit lens
uses M_l = I. The J-lens uses M_l = J_l, the corpus-averaged Jacobian of the final residual with
respect to h_l, summed over target positions p′ ≥ p and averaged over source positions p. The R-Lens
computes the same average through an LRP backward graph. J++ fits signed weights over per-cluster
"expert" Jacobians (§4). This section describes what we add.
