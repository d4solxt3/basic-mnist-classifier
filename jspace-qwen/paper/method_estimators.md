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

This holds for any J₀ built from *previous* prompts and probes, including the running mean of
earlier estimates (the "online" control variate). Unbiasedness says nothing about variance,
however, and Proposition 4 shows that the online control variate's variance explodes.

**Proposition 2 (variance, W = I).** For any M ∈ ℝ^{d×d},
E‖(d/K) QQᵀ M − M‖²_F = (d/K − 1) ‖M‖²_F. Hence each prompt contributes variance
σ² + (d/K − 1)·E‖J_n − J₀‖²_F, where σ² = E‖J_n − E J_n‖²_F.

*Consequences.* (i) Without a control variate, the sketch pays (d/K − 1)·‖J_n‖², which is
dominated by the large mean ‖E J_n‖². (ii) J₀ = I removes the identity path exactly.

**Corollary (no sketch beats the exact estimator at equal cost).** With independent prompts and
a fixed J₀, a budget of R VJP rows buys R/d exact prompts or R/K sketched ones, with variances

  exact: σ² d/R,   sketch: (d/R)·[σ² + (1 − K/d)‖E J_n − J₀‖²_F].

Their ratio is 1 + (1 − K/d)‖E J_n − J₀‖²/σ² ≥ 1. It approaches 1 only as J₀ approaches the lens
itself. For Qwen3-0.6B with J₀ = I, the ratio is 2.1, 3.3 and 5.7 at layers 7, 14 and 21 (§5.7).
Under independent prompts, sketching therefore cannot reduce variance at equal cost. It can only
(a) give *anytime* estimates below the d rows of one exact prompt, (b) spread a fixed budget over
d/K times as many *distinct* prompts, which helps when nearby prompts are correlated (§5.7), and
(c) *shape* the probes (Proposition 3).

**Proposition 4 (the online control variate is unstable).** Let W = I and c = d/K − 1. Use J₀ = I
for the first prompt and J₀ = J̄_n, the mean of the first n estimates, afterwards. Let
v_n = E‖J̄_n − E J_n‖²_F. Then

  v₁ = (c + 1)σ² + c‖E J_n − I‖²_F,   v_{n+1} = [(n² + c)·v_n + (c + 1)σ²] / (n + 1)².

*Proof.* The new estimate's error is (J_{n+1} − E J) + ((d/K)QQᵀ − I)(J_{n+1} − J̄_n).
Conditional on the past, both terms have mean zero and are uncorrelated with each other and with
J̄_n − E J. Proposition 2 gives the second term's variance, c·(σ² + v_n). Averaging into the
running mean gives the recursion. ∎

The coefficient on v_n, (n² + c)/(n + 1)², exceeds 1 while n < (c − 1)/2. The running-mean error
therefore *grows* for the first ≈ d/(2K) prompts: for d/K = 64 it grows through n = 31, by a
factor of 1.8 × 10⁵ in variance (430 in norm) on top of the first prompt's error. Only then does it
decay, roughly as 1/n in variance. The mechanism is that each sketched prompt re-amplifies the
current error of J₀ by (d/K − 1), and early on the running mean holds too few prompts to dilute it.

*Fix (shrunk online control variate).* Use J₀ = I + λ_n(J̄_n − I) with λ_n = n/(n + d/K). The
recursion keeps its form, with c·v_n replaced by c(λ_n² v_n + (1 − λ_n)²‖E J_n − I‖²). The
coefficient (n² + cλ_n²)/(n + 1)² is below 1 for every n, so the error never grows. Like the plain
online control variate, it approaches the exact estimator's cost per row as n grows. We derived
this fix after the online control variate diverged (§5.7), so its test is exploratory.

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
