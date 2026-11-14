<!-- Draft fragment; merged into paper.md. Results are filled in from results/jpp_compare_main.json only. -->

### The broadcast lens: reading what a position sends forward

The reference J-lens averages, over source positions p, the summed sensitivity of the final
residual at every target position p′ ≥ p:

  J_l = E_p [ Σ_{p′ ≥ p} ∂h_L[p′] / ∂h_l[p] ].

The p′ = p term differs in kind from the others. It contains the residual skip path, which
contributes exactly the identity (J_n = I + Δ_n), and the position's own next-token computation.
The terms with p′ > p exist only because later positions *read* position p through attention.
On a global-workspace reading, that cross-position traffic is what makes content "available":
the workspace is defined by broadcast, not by what a position is about to emit itself. The
top-10 lists of the reference lens show the cost of mixing the two. At a sentence-final period,
generic sentence openers (" They", " The", " This", " It") occupy top-10 slots in about a quarter
of fitting-split item–layer cells. These are predictions about the *next token*, not about held
content.

We therefore define the broadcast lens (B-lens) by keeping only strictly later targets:

  B_l = E_p [ Σ_{p′ ∈ T(p)} ∂h_L[p′] / ∂h_l[p] ],  with every p′ ∈ T(p) satisfying p′ > p.

The diagonal cannot be removed from a single target-summed backward pass: one VJP returns, for
every source position, its total over all targets. So we split each record's valid positions
(16 … T−2) into halves. Sources are the first half, and the one-hot cotangent is placed on every
position of the second half. Every term then has p′ > p, the identity path is absent, and the
cost per prompt is the same d VJPs as the reference estimator. We verified the estimator against
the brute-force Jacobian of a small random Qwen3 model (relative error ≈ 1e-7).

Two properties are worth noting. First, lens logits are invariant to rescaling the map, so the
different number of summed targets per source does not matter. Second, the B-lens needs no labels
and no extra machinery. It is a different *definition* of the lens, not a fitted correction.
Because J++'s expert-weight search accepts any set of candidate maps, we also report *JB-mix*,
per-layer signed weights over the Frobenius-normalised J_l and B_l. It comes in a labelled version
(fitting split, as for J++) and a label-free version (future tokens, as for SS-J++).
