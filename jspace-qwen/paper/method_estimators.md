<!-- Draft fragment; merged into paper.md. -->

### 3.4 Sketch estimators of the Jacobian lens

**Setup.** For a prompt n with valid positions P (positions ≥ 16, excluding the last), the
reference estimator computes the target-summed, source-averaged Jacobian

  J_n = |P|⁻¹ Σ_{p∈P} Σ_{p'∈P} ∂h_L[p'] / ∂h_l[p]  ∈ ℝ^{d×d},

one row block at a time. A one-hot cotangent e_i, placed at every target position, returns row i
at every source position in a single vector–Jacobian product (VJP), and does so for all source
layers at once. The lens is J = N⁻¹ Σ_n J_n. Each prompt costs d VJPs: 1,024 for Qwen3-0.6B and
5,120 for a 27B model. This cost dominates fitting.

**Observation 1 (exact identity path).** The p' = p term contains the residual connection, so
J_n = I + Δ_n, where Δ_n collects the derivatives of the block updates. The identity part needs no
estimation.

**Estimator.** Draw a Haar-random Q ∈ ℝ^{d×K} with orthonormal columns. Fix a symmetric
positive-definite *shaping* matrix W and a *control variate* J₀ that is independent of
(Q, prompt n). Compute the K VJPs G = (WQ)ᵀ J_n and set

  Ĵ_n = J₀ + (d/K) W⁻¹ Q (G − (WQ)ᵀ J₀).

**Proposition 1 (unbiasedness).** E[Ĵ_n | J₀] = E[J_n].
*Proof.* E[QQᵀ] = (K/d) I for a Haar K-frame, so E[(d/K) W⁻¹QQᵀW (J_n − J₀)] = J_n − J₀. ∎

This holds for any J₀ built from *previous* prompts and probes. The running mean of
earlier estimates ("online" control variate) therefore keeps every term conditionally unbiased:
the sequence of errors is a martingale difference sequence.

**Proposition 2 (variance, W = I).** For any M ∈ ℝ^{d×d},
E‖(d/K) QQᵀ M − M‖²_F = (d/K − 1) ‖M‖²_F. Hence each prompt contributes variance
σ² + (d/K − 1)·E‖J_n − J₀‖²_F, where σ² = E‖J_n − E J_n‖²_F.

*Consequences.* (i) Without a control variate, the sketch pays (d/K − 1)·‖J_n‖², which is
dominated by the large mean ‖E J_n‖². (ii) J₀ = I removes the identity path exactly. (iii) As the
online J₀ converges, the cost per prompt approaches (d/K)σ². At equal VJP budget this matches the
exact estimator, which pays σ² per prompt but uses K/d as many prompts. So sketching is not a
free lunch in the isotropic, Gaussian-fluctuation regime. What it buys is (a) *anytime*
estimates at any budget, (b) K/d-times-cheaper access to *distinct contexts*, which matters if
J_n is heavy-tailed or the prompt distribution is broad, and (c) the option to *shape* the probes.

**Proposition 3 (readout-aware shaping).** Readouts and swap directions use only the J-space
dictionary A J, where A = W_U diag(g) is the unembedding with the final-norm gain folded in.
For Gaussian probes, the leading term of E‖A(Ĵ_n − J_n)‖²_F is
K⁻¹ tr(W M Mᵀ W) · tr(W⁻¹ AᵀA W⁻¹), with M = J_n − J₀. Over symmetric PD W, with an
isotropic prior on M (MMᵀ ∝ I), this is minimised by W = (AᵀA)^{1/4}. The variance relative to
W = I is reduced by the factor

  ρ(A) = (Σ_i s_i)² / (d Σ_i s_i²) ≤ 1,

where s_i are the singular values of A. Equality holds only for a flat spectrum. *Proof sketch.*
Write Z = W². By the Cauchy–Schwarz inequality for the trace inner product,
tr(Z) tr(Z⁻¹AᵀA) ≥ (tr (AᵀA)^{1/2})², with equality at Z ∝ (AᵀA)^{1/2}. ∎
(Without the isotropy assumption, the optimum is the matrix geometric mean of (MMᵀ)⁻¹ and
AᵀA, which a pilot estimate of MMᵀ can approximate.)

ρ(A) can be computed from the weights alone. It *predicts* the benefit of shaping before any
Jacobian is computed.
