# J++ Lens improvement sprint: scope, plan and Experiment 1 pre-registration

Committed before any Experiment 1 training code was written. Changes after commit are appended as
dated amendments at the bottom; nothing above the amendments line is edited afterwards.

## Scope

**Goal.** Find out, with out-of-sample tests, whether any of four cheap changes to the J++ Lens
(Ayonrinde & Lindsey, 2026; `safety-research/jpp_lens` @ `d7efc0d`) improves (i) readout of latent
intermediates and (ii) intervention reliability, without raising false positives. The four changes
are Exp 1–4 below, run one at a time, cheapest first.

**Done when.** Each experiment that is run has:
- a committed pre-registration with confirm/refute thresholds;
- all four metric families (M1–M4 below) on fresh held-out splits, against J++ and against its
  controls;
- a RUNLOG row and a verdict (confirmed / refuted / inconclusive), with raw failure cases next to the
  aggregates.

The sprint stops early if an experiment's premise fails its own cheap pre-check.

**Deliverable.**
- `sprint/RUNLOG.md` (append-only);
- `sprint/results/*.json` (every number, per split);
- per-experiment tables;
- a short write-up of what was confirmed and refuted.

Code is in `sprint/`, built on the cached Qwen3-0.6B shards and the jpp_lens algorithms.

## Setting (decided, with reasons)

- **Model: Qwen3-0.6B.** Not Qwen3.5-0.8B, for three reasons:
  - Cached LRP expert Jacobians (E = 8, PCA-64 k-means router, 32 WikiText prompts) exist for all
    27 source layers.
  - E0.1 verified that our cached-expert estimator is identical to jpp_lens's
    `ExpertJacobianRelPTrainer` (`lrp_mode="rlens"`): relative error 0.0 on a test prompt.
  - A new model would need about 3 CPU-hours of LRP Jacobians before any experiment.
  - We would revisit only if a hypothesis is confirmed and worth replicating.
- **Layers.**
  - The seven J++ layers {4, 7, 11, 14, 18, 21, 24} of 28.
  - Strata: early = {4, 7}, mid = {11, 14}, late = {18, 21, 24}.
  - A stratum score is the mean of its per-layer scores. Each layer is read out on its own; this is
    the quantity J++'s per-layer objective fits.
  - Best-of-7 (J++'s headline protocol) is reported alongside.
- **Items.**
  - 250 correctness-filtered items of the released lens-eval sets: multihop 14, multilingual 15,
    poetry 8, association 99, typo 96, order-ops 18.
  - Recall is macro-averaged over the five macro evals; order-ops is reported separately.
  - Three macro evals have 8–15 items, so per-split estimates are noisy. This is why every claim
    rests on many splits.
- **Splits.**
  - J++'s stratified rule: 20% fitting, 80% held-out per eval, `numpy.default_rng(seed)`.
  - Tuning splits: seeds 0–5. These were used in earlier rounds, and any hyperparameter is chosen
    on them.
  - Confirmation splits: seeds 100–119. They have never been used, and are scored once per
    pre-registered arm.
- **Weight search.**
  - jpp_lens's algorithm (pair-level macro pass@10 surrogate, sigmoid margin, temperature 0.1,
    Adam lr 0.05, 200 steps, exact score every 5 steps, best (exact, surrogate) iterate), in an
    exact-gradient fast implementation (`sprint/jppsearch.py`).
  - E0.1 showed the search is chaotic at the float level: identical algorithms reach different
    optima, with train macro differences up to 1.4 points.
  - E0.1-search-noise measures how much held-out recall moves from that alone. Every arm, J++
    included, runs through the same implementation.

## Evaluation suite (every experiment reports all four)

**M1 readout.**
- Metrics: held-out recall@10 and MRR, per layer and per stratum, plus best-of-7.
- Rank = 1 + number of vocabulary logits strictly above the best surface form.
- Readout Filtering on: non-semantic ids are excluded, and every item's candidates are spared
  (J++'s rule).

**M2 false positives.**
- (a) Matched controls: each held-out intermediate is ranked under up to 5 other held-out items of
  the same eval.
  - A control shares no candidate token with the item, and the control prompt does not contain the
    intermediate.
  - Metrics: FP@10 (fraction ranked in the top 10), mean softmax mass on the intermediate, and the
    specificity AUC P(rank under own prompt < rank under control).
- (b) Generic controls: every held-out intermediate is ranked at 256 held-out WikiText positions;
  FP@10 and mass.

**M3 transport.**
- Positions: 4,096 held-out WikiText positions (records 7000–7127, never used for any fit).
- Cosine between M_l h_l and h_27.
- NMSE = Σ‖α M_l h_l − h_27‖² / Σ‖h_27‖², with one scale α fitted on 2,048 calibration positions
  (records 5000–5255).
- KL(model next-token distribution ‖ lens distribution) on 512 positions.
- Caveat: J-lens-family maps average influence on *all later* positions. They are not estimators of
  h_27 at the same position, so M3 measures a different property than M1. The identity map (logit
  lens) and the ridge regression lens (which optimises NMSE directly) are reported as references.

**M4 causal swap (top candidates only).**
- Flexible-generalisation clamp swaps (`f(A)` → `f(B)`), clean-correct trials only.
- Lens directions at a fixed set of edited layers ({4, 7} and {4, 7, 11, 14}), scales 1 and 2.
- Controls: an unrelated-word null and logit-lens directions.

## Execution plan

1. **Setup.** Done: E0.0 tests, E0.1 fidelity and E0.1 spectral diagnostic. Next: E0.1 search
   noise, then E0.2 baseline (metric_before) on the tuning splits.
2. **Exp 1 (spectral / subspace condensation).** Pre-registered below.
   - Tuning stage: λ only, early layers, tuning splits.
   - Confirmation stage: 20 fresh splits.
   - M4 only if the M1–M2 criteria pass.
3. **Exp 2 (self-supervised KL + false-positive penalty).** The pre-registration will be written
   after Exp 1's verdict, with the same splits machinery.
   - Primary test: leave-one-task-out. Weights are fitted on four macro evals and scored on the
     fifth.
   - Prediction: a next-token KL objective pulls the map toward next-token readouts (a tuned-lens
     objective), which the latent-intermediate evals do not reward.
4. **Exp 3 (input-conditional gating).** Same harness, using the callable-lens path.
   - The routed lens (hard k-means gating) already trailed J++ by 1.6 points over 25 splits, which
     bounds what soft gating can add.
   - A linear router has 8 × 1024 parameters, so it needs a label-free objective (KL / transport) or
     heavy regularisation.
5. **Exp 4 (outlier-robust aggregation).**
   - Cheap version first: prompt-level Winsorised or Huber means of the 32 stored per-prompt
     Jacobians, available at layers 7, 14 and 21.
   - Per-position filtering needs new backward passes (about 1.5 CPU-hours per lens) and only
     follows if the cheap version moves anything.

## Experiment 1 pre-registration: spectral / subspace expert condensation

**Hypothesis H1 (as given).** Scalar expert weights cannot separate clean singular directions from
noisy outlier directions within one expert. Projecting the experts onto singular bands of the pooled
Jacobian and learning a weight per (expert, band), with shrinkage, will raise held-out early-layer
recall@10 by more than 3 points without raising the false-positive rate.

**Pre-check already run (E0.1-spectral-diag, label-free).**
- Expert deviations J_e − J̄ are reproducible across prompt halves in every band (split-half
  reliability 0.83–0.97, signal-to-noise ≥ 9). The "noise" mechanism is therefore not supported.
- The coordinate-aligned outlier directions are the leading singular directions.
- What remains testable is the weaker claim that band-specific expert weighting improves readout.
- **Our prediction before running: refuted or inconclusive** (Δ early < +1 point).

**Construction.** For each layer l:
- Pooled LRP map J̄_l = Σ_e n_e J_{l,e} / Σ_e n_e, with SVD J̄_l = U S Vᵀ.
- Output bands on U: b ∈ {[0,16), [16,64), [64,256), [256,1024)}, fixed a priori with no tuning.
- Projectors P_b = U_b U_bᵀ; Σ_b P_b = I.
- Atoms a_{e,b} = P_b J_{l,e}, 32 per layer.
- Map M_l = Σ_{e,b} W_{e,b} a_{e,b}, with W unit-norm (scale-free, as in J++).

**Arms** (all fitted per layer on each split's fitting items with the same search):
- **A0 J++.** 8 expert weights, reference starts.
- **A1 SB-J++ (primary).** 32 band weights. Objective = J++ surrogate − λ‖u − u_ref‖², where
  u_ref = (w_J++ ⊗ 1_4)/2 is the split's own J++ solution expanded to bands (unit norm).
  - Starts: u_ref plus the three reference starts over atoms.
  - The returned iterate maximises (exact − penalty, surrogate − penalty).
  - A1 contains J++ exactly (u = u_ref, zero penalty), so its fitting-split score is never below
    J++'s.
- **A2 random-basis control (same parameter budget).** As A1, but with U replaced by a Haar-random
  orthogonal matrix (seed 1000 + l). Same band sizes, same λ, same starts. It also contains J++.
- **Secondary arms (exploratory, no confirmatory claim):**
  - A3: 4 weights over pooled-band atoms P_b J̄ (spectral reweighting of the R-lens).
  - A4: A1 with λ = 0 (from the tuning grid; overfitting check).
  - A5: label-free spectral denoising followed by the plain J++ search. Here
    J'_e = J̄ + Σ_b κ_{e,b} P_b (J_e − J̄), with κ_{e,b} = max(0, 1 − noise/total) estimated from the
    two prompt halves.

**Hyperparameter (tuning stage only).**
- λ ∈ {0, 0.03, 0.1, 0.3, 1}.
- Chosen on tuning splits 0–5 as the value with the highest mean held-out early-stratum recall@10
  of A1.
- If several values fall within 0.2 points of the best, choose the largest λ.
- The same λ is used for A1 and A2 on every layer.

**Confirmation stage.**
- Seeds 100–119, all arms. A2 is fitted at the early layers only, the claim's stratum. A3 and A5 run
  on the first 10 seeds only.

**Primary metric.** Paired per-split difference in held-out early-stratum recall@10, A1 − A0.

**Confirm H1** if all four hold:
1. The mean difference is at least +3.0 points.
2. A1 beats A0 on at least 16 of 20 splits. Ties are not wins.
3. False positives do not rise: the mean change in early-stratum FP@10 is at most +0.5 points for
   matched controls and at most +0.5 points for WikiText controls.
4. The gain is specific to the spectral basis: the mean of A1 − A2 is at least +1.5 points, and A1
   beats A2 on at least 14 of 20 splits.

**Refute H1** if any of these hold:
- the mean of A1 − A0 is below +1.0 point;
- A1 wins on 12 or fewer splits;
- the mean early-stratum FP@10 change (matched) exceeds +1.0 point.

**Otherwise, inconclusive.** If criteria 1–3 hold but criterion 4 fails, the result is labelled a
"parameter-count effect, not spectral".

**Guardrails** (reported, and flagged if violated):
- held-out recall changes in the mid and late strata ≥ −1.0 point;
- the early MRR change has the same sign as the early recall change;
- early transport cosine change ≥ −0.02 and NMSE change ≤ +0.02;
- the best-of-7 change.

**M4** runs only if criteria 1–3 hold. It compares A1 and A0 maps fitted on seed 100's fitting
items, at layers {4, 7} and {4, 7, 11, 14}:
- A1's swap success must not fall more than 5 points below A0's;
- the null must stay at 2% or less.

**Secondary analyses (reported, not part of the verdict).**
- Leave-one-task-out: fit on four macro evals plus order-ops, score the held-out fifth.
- The fitting–held-out gap (overfitting).
- Raw failure cases: per split, the held-out items whose early-layer hit status differs between A1
  and A0, with their top-5 readout tokens under both.

## Amendments

(none yet)
